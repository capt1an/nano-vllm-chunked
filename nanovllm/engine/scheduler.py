from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus, SchedulerOutput
from nanovllm.engine.block_manager import BlockManager
from nanovllm.distributed.kv_transfer.config import PDRole
from nanovllm.distributed.kv_transfer.scheduler import PullSchedulerConnector


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.enable_continuous_batching = config.enable_continuous_batching
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.pd_config = config.pd_config
        self.max_model_len = config.max_model_len
        self.connector = PullSchedulerConnector(config.pd_config) if config.pd_config else None
        self.requests = {}
        self.remote_waiting = {}
        self.cancelled = set()
        self.handoffs = deque()
        self.failures = deque()
        self._seen_transfer_ids = set()

    def is_finished(self):
        return not (self.waiting or self.running or self.remote_waiting
                    or (self.connector and self.connector.has_pending_work()))

    def add(self, seq: Sequence):
        if self.connector:
            if seq.request_id in self.requests:
                raise ValueError("Duplicate request_id")
            if not seq.token_ids or seq.max_tokens < 1:
                raise ValueError("Nonempty prompt and positive max_tokens required")
            decode = self.pd_config.role == PDRole.DECODE
            if decode != (seq.kv_transfer_params is not None):
                raise ValueError("Only decode requests require a remote KV handoff")
            if decode and seq.kv_transfer_params.transfer_id in self._seen_transfer_ids:
                raise ValueError("Duplicate transfer attempt")
            total = seq.num_prompt_tokens + (seq.max_tokens if decode else 0)
            if total > self.max_model_len or (total + self.block_size - 1) // self.block_size > len(self.block_manager.blocks):
                raise ValueError("PD request exceeds model length or KV capacity")
            if not decode and not self.enable_chunked_prefill and seq.num_prompt_tokens > self.max_num_batched_tokens:
                raise ValueError("Prompt exceeds token budget with chunked prefill disabled")
            self.requests[seq.request_id] = seq
            if decode:
                self._seen_transfer_ids.add(seq.kv_transfer_params.transfer_id)
        self.waiting.append(seq)

    # ------------------------------------------------------------------ #
    # Public dispatch                                                      #
    # ------------------------------------------------------------------ #

    def schedule(self) -> SchedulerOutput:
        if self.connector:
            return self._schedule_pd()
        if self.enable_continuous_batching:
            return self._schedule_continuous_batching()
        else:
            return self._schedule_legacy()

    def _schedule_pd(self):
        # Reserve full footprints at admission: PD never preempts into a local
        # prompt recomputation. Scan waiting requests so one large request does
        # not block smaller requests that fit while transfers are in flight.
        for _ in range(len(self.waiting)):
            seq = self.waiting.popleft()
            if len(self.running) + len(self.remote_waiting) >= self.max_num_seqs:
                self.waiting.append(seq)
                continue
            is_decode = self.pd_config.role == PDRole.DECODE
            total = seq.num_prompt_tokens + (seq.max_tokens if is_decode else 0)
            count = (total + self.block_size - 1) // self.block_size
            if not self.block_manager.allocate_exclusive(seq, count):
                self.waiting.append(seq)
                continue
            if is_decode:
                tokens, _ = self.connector.get_num_new_matched_tokens(seq, 0)
                seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
                self.remote_waiting[seq.request_id] = seq
                self.connector.update_state_after_alloc(seq, tuple(seq.block_table), tokens)
            else:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)

        prefill, decode = [], []
        budget = self.max_num_batched_tokens
        # Round-robin within each phase; ready decode requests get first choice.
        candidates = sorted(self.running, key=lambda seq: seq.is_prefill)
        for seq in candidates:
            if not budget:
                break
            if not self.enable_continuous_batching and (prefill or decode):
                if seq.is_prefill != bool(prefill):
                    continue
            needed = seq.num_prefill_tokens_remaining if seq.is_prefill else 1
            if seq.is_prefill and not self.enable_chunked_prefill and needed > budget:
                continue
            seq.num_scheduled_tokens = min(needed, budget)
            if seq.is_prefill:
                prefill.append(seq)
            else:
                self.block_manager.may_append(seq)
                decode.append(seq)
            budget -= seq.num_scheduled_tokens
            self.running.remove(seq)
            self.running.append(seq)
        output = SchedulerOutput(prefill + decode, len(prefill),
                                 sum(seq.num_scheduled_tokens for seq in prefill))
        output.connector_metadata = self.connector.build_connector_meta(output)
        return output

    def update_connector_output(self, output):
        if not self.connector:
            return
        failures = {event.transfer_id: event.message for event in output.failures}
        terminal = output.received | output.sent | output.cancelled | failures.keys()
        for transfer_id in terminal:
            request_id = self.connector.pending.get(transfer_id)
            seq = self.requests.get(request_id)
            if seq is None:  # Late/duplicate event from an old attempt.
                continue
            if transfer_id in output.received and request_id not in self.cancelled:
                self.remote_waiting.pop(request_id, None)
                # Full prompt KV has no logits. Recompute its last token on D.
                seq.num_cached_tokens = seq.num_prompt_tokens - 1
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                if transfer_id in failures:
                    self.failures.append((request_id, failures[transfer_id]))
                self._release_pd(seq)
        self.connector.update_connector_output(output)

    def _release_pd(self, seq):
        self.remote_waiting.pop(seq.request_id, None)
        self.requests.pop(seq.request_id, None)
        self.cancelled.discard(seq.request_id)
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)

    def cancel_request(self, request_id):
        if not self.connector:
            raise ValueError("Cancellation interface currently requires PD")
        seq = self.requests.get(request_id)
        if seq is None:
            return
        for queue in (self.waiting, self.running):
            if seq in queue:
                queue.remove(seq)
        if request_id in self.connector.pending.values():
            self.cancelled.add(request_id)
            self.connector.cancel_request(seq)
        else:
            self._release_pd(seq)

    def shutdown(self):
        """Reclaim scheduler ownership only AFTER worker transport shutdown."""
        if self.connector:
            self.waiting.clear()
            self.running.clear()
            for seq in list(self.requests.values()):
                self._release_pd(seq)
            self.connector.pending.clear()
            self.connector.shutdown()

    # ------------------------------------------------------------------ #
    # Continuous batching (new)                                            #
    # ------------------------------------------------------------------ #

    def _schedule_continuous_batching(self) -> SchedulerOutput:
        prefill_seqs: list[Sequence] = []
        decode_seqs: list[Sequence] = []
        num_batched_tokens = 0

        # ── Phase 1: decode（running 队列，优先保障 ITL）──────────────────────
        decode_scheduled: list[Sequence] = []
        not_scheduled: list[Sequence] = []
        while self.running:
            seq = self.running.popleft()
            if len(decode_scheduled) + len(prefill_seqs) >= self.max_num_seqs:
                not_scheduled.append(seq)
                continue
            if num_batched_tokens >= self.max_num_batched_tokens:
                not_scheduled.append(seq)
                continue

            if seq.is_prefill:
                # 仍在 chunked prefill 中：继续推进剩余 prompt tokens
                remaining = self.max_num_batched_tokens - num_batched_tokens
                num_tokens_needed = seq.num_prefill_tokens_remaining
                seq.num_scheduled_tokens = min(num_tokens_needed, remaining)
                num_batched_tokens += seq.num_scheduled_tokens
                prefill_seqs.append(seq)
                decode_scheduled.append(seq)   # 占 seq 槽位
            else:
                # 正常 decode
                while not self.block_manager.can_append(seq):
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        seq = None
                        break
                if seq is None:
                    break
                seq.num_scheduled_tokens = 1
                self.block_manager.may_append(seq)
                num_batched_tokens += 1
                decode_seqs.append(seq)
                decode_scheduled.append(seq)

        # 未调度的 seq 放回队头，保持顺序
        self.running.extendleft(reversed(not_scheduled))
        self.running.extendleft(reversed(decode_scheduled))

        # ── Phase 2: prefill（waiting 队列，用剩余 budget）───────────────────
        while self.waiting:
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if len(prefill_seqs) + len(decode_seqs) >= self.max_num_seqs:
                break

            seq = self.waiting[0]

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                # print(" num cached blocks: ", num_cached_blocks)
                if num_cached_blocks == -1:
                    break
                num_tokens_needed = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens_needed = seq.num_prefill_tokens_remaining

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens_needed, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            # 不管本 chunk 是否结束 prefill，都先移入 running
            # is_prefill 会在 postprocess 后自动由 num_cached_tokens 推导
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            prefill_seqs.append(seq)

        if not prefill_seqs and not decode_seqs:
            raise RuntimeError(
                "Continuous batching scheduler: nothing scheduled. "
                "This usually means the KV cache is exhausted or "
                "enable_chunked_prefill=False and no sequence fits in max_num_batched_tokens."
            )

        all_seqs = prefill_seqs + decode_seqs
        # self.block_manager.print_status()
        return SchedulerOutput(
            seqs=all_seqs,
            num_prefill_seqs=len(prefill_seqs),
            num_prefill_tokens=sum(s.num_scheduled_tokens for s in prefill_seqs),
        )

    # ------------------------------------------------------------------ #
    # Legacy scheduling (original nano-vllm behaviour)                     #
    # ------------------------------------------------------------------ #

    def _schedule_legacy(self) -> SchedulerOutput:
        """Replicate the original nano-vllm scheduler.

        Key differences from ``_schedule_continuous_batching``:

        * A single step is **either** pure prefill **or** pure decode — never a
          mix of both.
        * Prefill is attempted first; decode only runs when no prefill is
          possible.
        * Chunked prefill is governed by ``enable_chunked_prefill``:

          - ``True``  (default): the **first** sequence in the prefill batch
            may be partially scheduled (chunked).  All subsequent sequences
            must fit entirely in the remaining token budget.  This matches the
            original nano-vllm behaviour prior to continuous batching.
          - ``False``: **every** prefill sequence must fit entirely — if the
            next sequence does not fit, the prefill phase ends immediately.
        """
        scheduled_seqs: list[Sequence] = []
        num_batched_tokens = 0

        # ── prefill ──────────────────────────────────────────────────────
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens_needed = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens_needed = seq.num_tokens - seq.num_cached_tokens

            # Chunked-prefill policy for legacy mode
            if self.enable_chunked_prefill:
                # Original behaviour: only the first seq may be partially filled
                if remaining < num_tokens_needed and scheduled_seqs:
                    break
            else:
                # Strict: every seq must fit entirely
                if remaining < num_tokens_needed:
                    break

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens_needed, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            # Original logic: only move to running when prefill is complete
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            # All prefill — even if the first seq was only partially filled,
            # it's still a "prefill batch" (the model uses flash_attn_varlen_func).
            return SchedulerOutput(
                seqs=scheduled_seqs,
                num_prefill_seqs=len(scheduled_seqs),
                num_prefill_tokens=sum(s.num_scheduled_tokens for s in scheduled_seqs),
            )

        # ── decode ───────────────────────────────────────────────────────
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        if not scheduled_seqs:
            raise RuntimeError(
                "Legacy scheduler: nothing scheduled in decode phase. "
                "This usually means the KV cache is exhausted or all prompts "
                "are longer than max_num_batched_tokens with chunked prefill disabled."
            )
        self.running.extendleft(reversed(scheduled_seqs))
        return SchedulerOutput(
            seqs=scheduled_seqs,
            num_prefill_seqs=0,
            num_prefill_tokens=0,
        )

    # ------------------------------------------------------------------ #
    # Bookkeeping                                                          #
    # ------------------------------------------------------------------ #

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.num_cached_tokens = 0
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, output: SchedulerOutput, token_ids: list[int]):
        """Update sequence state after the model has run.

        Parameters
        ----------
        output:
            The :class:`SchedulerOutput` that was used to build this step's
            batch.  We need it to know which seqs are prefill vs decode and
            how many tokens each was scheduled for.
        token_ids:
            One sampled token per sequence in ``output.seqs``.  For prefill
            seqs that have not finished their prompt yet (partial chunk), the
            token is meaningless and will be discarded – only the KV cache
            accounting matters.
        """
        for seq, token_id in zip(output.seqs, token_ids):
            if not self.connector:
                self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            # Prefill seq that still has un-prefilled tokens: skip token emit.
            if seq.is_prefill:
                continue

            if self.connector and self.pd_config.role == PDRole.PREFILL:
                self.running.remove(seq)
                seq.status = SequenceStatus.FINISHED
                _, handoff = self.connector.request_finished(seq, tuple(seq.block_table))
                self.handoffs.append((seq.request_id, handoff))
                continue

            seq.append_token(token_id)
            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                # print("触发deallocate!")
                if self.connector:
                    self._release_pd(seq)
                else:
                    self.block_manager.deallocate(seq)
                # running.remove is O(n) but the list is short in practice.
                if seq in self.running:
                    self.running.remove(seq)
