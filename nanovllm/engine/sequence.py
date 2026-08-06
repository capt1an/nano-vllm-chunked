from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class SchedulerOutput:
    """Describes the composition of one scheduled step.

    Replaces the old ``(seqs, is_prefill: bool)`` tuple.  A step may now
    contain *both* prefill tokens and decode tokens (mixed batch), so a
    single boolean is no longer sufficient.

    Attributes
    ----------
    seqs:
        All sequences participating in this step, prefill sequences first,
        decode sequences after.  The ordering matches the token layout that
        ``prepare_inputs`` will build.
    num_prefill_seqs:
        Number of leading entries in ``seqs`` that are still in the prefill
        phase.  ``seqs[num_prefill_seqs:]`` are all decode sequences.
    num_prefill_tokens:
        Total token count contributed by the prefill sequences in this step
        (sum of ``seq.num_scheduled_tokens`` for prefill seqs).  This is the
        split point that the attention kernel and ``run_model`` use to decide
        whether CUDA graphs can be used.
    """
    seqs: list["Sequence"]
    num_prefill_seqs: int
    num_prefill_tokens: int

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                  #
    # ------------------------------------------------------------------ #

    @property
    def prefill_seqs(self) -> list["Sequence"]:
        return self.seqs[:self.num_prefill_seqs]

    @property
    def decode_seqs(self) -> list["Sequence"]:
        return self.seqs[self.num_prefill_seqs:]

    @property
    def has_prefill(self) -> bool:
        return self.num_prefill_tokens > 0
    
    @property
    def has_decode(self) -> bool:
        return len(self.seqs) - self.num_prefill_seqs > 0

    @property
    def is_pure_decode(self) -> bool:
        return self.num_prefill_tokens == 0

    def __len__(self) -> int:
        return len(self.seqs)


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    # ------------------------------------------------------------------ #
    # Prefill-phase helpers                                                #
    # ------------------------------------------------------------------ #

    @property
    def is_prefill(self) -> bool:
        """True while this sequence still has un-prefilled prompt tokens.

        Replaces the old mutable ``self.is_prefill = True/False`` flag with a
        derived property so there is a single source of truth.  A sequence is
        in the prefill phase as long as the number of tokens whose KV entries
        have been computed (``num_cached_tokens``) is less than the full
        prompt length.
        """
        return self.num_cached_tokens < self.num_prompt_tokens

    @property
    def num_computed_tokens(self) -> int:
        """Alias for ``num_cached_tokens``, emphasising the prefill viewpoint.

        ``num_cached_tokens`` tracks how many tokens (prompt + previously
        generated) have their KV cache written.  During chunked prefill this
        equals the number of prompt tokens processed so far, which is exactly
        what the scheduler needs to know to compute the next chunk's slice.
        """
        return self.num_cached_tokens

    @property
    def num_prefill_tokens_remaining(self) -> int:
        """Prompt tokens that have not yet been prefilled."""
        return max(0, self.num_prompt_tokens - self.num_cached_tokens)

    # ------------------------------------------------------------------ #
    # Existing interface (unchanged)                                       #
    # ------------------------------------------------------------------ #

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        # is_prefill is now derived; serialise token_ids when in prefill phase
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            last_state,
        )

    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            last_state,
        ) = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state

    def __repr__(self):
        return (
            f"Sequence("
            f"seq_id={self.seq_id}, "
            f"status={self.status.name}, "
            f"tokens={self.num_tokens}, "
            f"prompt={self.num_prompt_tokens}, "
            f"cached={self.num_cached_tokens}, "
            f"scheduled={self.num_scheduled_tokens}, "
            f"prefill={self.is_prefill}"
            f")"
        )