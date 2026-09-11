"""CPU-only control messages; no tensors, pointers or transport handles.

The initial contract transfers a full prompt between identical KV layouts,
with TP=1 and exclusively allocated destination blocks. Block IDs are ordered
by logical token position. The worker owns handles and in-flight state across
steps; clearing a ConnectorMetadata does not cancel its transfers.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class KVTransferParams:
    """P -> router -> D handoff, also retained for P's pending read.

    transfer_id identifies one handoff attempt, distinct from request_id.
    A retry must use a new transfer_id so late events cannot finish a new task.
    block_size and num_tokens describe the valid prefix; final-block padding
    is never part of the valid KV. Model/dtype/layout compatibility must be
    checked by the worker handshake before any data transfer.
    """
    transfer_id: str
    remote_engine_id: str
    remote_request_id: str
    remote_host: str
    remote_port: int
    remote_block_ids: tuple[int, ...]
    num_tokens: int
    block_size: int

    def __post_init__(self):
        object.__setattr__(self, "remote_block_ids", tuple(self.remote_block_ids))
        if not all(value.strip() for value in (
            self.transfer_id, self.remote_engine_id,
            self.remote_request_id, self.remote_host,
        )):
            raise ValueError("Transfer identifiers and remote_host must be nonempty")
        if not 1 <= self.remote_port <= 65535:
            raise ValueError("remote_port must be between 1 and 65535")
        if self.num_tokens <= 0 or self.block_size <= 0:
            raise ValueError("num_tokens and block_size must be positive")
        _validate_blocks(self.remote_block_ids, self.num_tokens, self.block_size)


def _validate_blocks(block_ids, num_tokens, block_size):
    required = (num_tokens + block_size - 1) // block_size
    if len(block_ids) != required:
        raise ValueError("Block count must match the valid token count")
    if any(block < 0 for block in block_ids) or len(set(block_ids)) != len(block_ids):
        raise ValueError("Block IDs must be nonnegative and unique")


@dataclass(frozen=True, slots=True)
class KVLoadTask:
    """D-side destination allocation paired with the P-side handoff."""
    request_id: str
    remote: KVTransferParams
    local_block_ids: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "local_block_ids", tuple(self.local_block_ids))
        if not self.request_id.strip():
            raise ValueError("request_id must be nonempty")
        _validate_blocks(self.local_block_ids, self.remote.num_tokens, self.remote.block_size)


@dataclass
class ConnectorMetadata:
    """New commands for one scheduler step, not a snapshot of all transfers.

    In pull mode sends register P's retained blocks for remote reading; they
    do not initiate a push. Cancels identify attempts, not entire requests.
    Cancellation submission alone never authorizes block reclamation.
    """
    loads: list[KVLoadTask] = field(default_factory=list)
    sends: list[KVTransferParams] = field(default_factory=list)
    cancels: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class KVTransferFailure:
    transfer_id: str
    message: str


@dataclass
class ConnectorOutput:
    """Terminal events keyed by transfer_id, resolved by the scheduler.

    received means KV is valid and GPU-visible. sent means the remote read
    no longer needs P's blocks. cancelled and failures are terminal only once
    transport accesses have stopped: a timeout alone is not safe to free.
    An attempt must appear in at most one terminal category on each endpoint.
    Worker-local handles and engine-level registrations never travel here.
    """
    received: set[str] = field(default_factory=set)
    sent: set[str] = field(default_factory=set)
    cancelled: set[str] = field(default_factory=set)
    failures: list[KVTransferFailure] = field(default_factory=list)
