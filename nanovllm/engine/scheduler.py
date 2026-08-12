from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus, SchedulerOutput
from nanovllm.engine.block_manager import BlockManager


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

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # ------------------------------------------------------------------ #
    # Public dispatch                                                      #
    # ------------------------------------------------------------------ #

    def schedule(self) -> SchedulerOutput:
        if self.enable_continuous_batching:
            return self._schedule_continuous_batching()
        else:
            return self._schedule_legacy()

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
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            # Prefill seq that still has un-prefilled tokens: skip token emit.
            if seq.is_prefill:
                continue

            seq.append_token(token_id)
            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                # print("触发deallocate!")
                self.block_manager.deallocate(seq)
                # running.remove is O(n) but the list is short in practice.
                if seq in self.running:
                    self.running.remove(seq)
