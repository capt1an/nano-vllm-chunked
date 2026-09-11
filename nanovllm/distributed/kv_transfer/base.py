"""Scheduler/worker contracts for request-level asynchronous KV transfer.

No transport is implemented here. Scheduler methods run on the scheduling
thread and never access GPU memory. Workers consume each step's commands once
and retain in-flight state independently of the bound metadata object.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from nanovllm.distributed.kv_transfer.config import PDConfig
from nanovllm.distributed.kv_transfer.metadata import (
    ConnectorMetadata, ConnectorOutput, KVTransferParams,
)

if TYPE_CHECKING:
    import torch
    from nanovllm.engine.sequence import SchedulerOutput, Sequence


class SchedulerConnector(ABC):
    def __init__(self, config: PDConfig):
        self.config = config

    @abstractmethod
    def get_num_new_matched_tokens(
        self, seq: "Sequence", num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Query additional remotely available tokens without allocating blocks.

        None means retry later; (0, False) means no load. A positive count
        with True requires asynchronous loading. This query must not enqueue
        duplicate tasks when admission is retried. Initial full-prompt PD
        admission uses num_computed_tokens=0 and exclusive destination blocks.
        """
        raise NotImplementedError

    @abstractmethod
    def update_state_after_alloc(
        self, seq: "Sequence", block_ids: tuple[int, ...], num_external_tokens: int,
    ) -> None:
        """Queue a load after the scheduler reserves destination blocks.

        Zero external tokens must not enqueue a load. Snapshot block IDs;
        never retain a mutable block_table alias. The scheduler must mark the
        request WAITING_FOR_REMOTE_KVS before submitting the resulting metadata.
        This method does not change Sequence or BlockManager state itself.
        """
        raise NotImplementedError

    @abstractmethod
    def build_connector_meta(self, output: "SchedulerOutput") -> ConnectorMetadata:
        """Drain new commands into a per-step object, including empty batches.

        Do not mutate output. The caller attaches the result to its
        connector_metadata field. Draining commands does not finish transfers;
        retain attempt-to-request mappings until terminal events are processed.
        """
        raise NotImplementedError

    @abstractmethod
    def request_finished(
        self, seq: "Sequence", block_ids: tuple[int, ...],
    ) -> tuple[bool, KVTransferParams | None]:
        """Return (delay_free, handoff) after local computation finishes.

        P queues a pending-read registration and returns an immutable handoff
        for the router. Only computed prompt blocks belong in the handoff,
        not an extra output-token block. True tells the scheduler to retain
        blocks even though the request has left the execution queue. The
        worker must ensure producer writes are complete before permitting reads.
        """
        raise NotImplementedError

    @abstractmethod
    def cancel_request(self, seq: "Sequence") -> None:
        """Queue cancellation of tracked attempts; do not release their blocks.

        Handle both not-yet-submitted and active attempts. Completion can race
        cancellation: keep mappings until one terminal outcome is processed.
        """
        raise NotImplementedError

    @abstractmethod
    def update_connector_output(self, output: ConnectorOutput) -> None:
        """Update internal attempt tracking from worker terminal events.

        Deduplicate events and reject/ignore stale attempts. Scheduler remains
        responsible for request state transitions and block reclamation; it
        consumes the same output using its request/transfer ownership records.
        """
        raise NotImplementedError

    @abstractmethod
    def has_pending_work(self) -> bool:
        """Include queued commands and attempts awaiting terminal events."""
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Release control resources after worker transfers have been stopped.

        This is not an implicit permission to reclaim blocks still in use.
        """
        raise NotImplementedError


class WorkerConnector(ABC):
    def __init__(self, config: PDConfig):
        self.config = config
        self._connector_metadata: ConnectorMetadata | None = None

    def bind_connector_metadata(self, metadata: ConnectorMetadata) -> None:
        """Bind one step's commands; previous step must have been cleared."""
        if self._connector_metadata is not None:
            raise RuntimeError("Connector metadata is already bound")
        self._connector_metadata = metadata

    def clear_connector_metadata(self) -> None:
        """Drop only this step's reference, never the in-flight transfer state."""
        self._connector_metadata = None

    def _get_connector_metadata(self) -> ConnectorMetadata:
        if self._connector_metadata is None:
            raise RuntimeError("No connector metadata is bound")
        return self._connector_metadata

    @abstractmethod
    def register_kv_cache(self, kv_cache: "torch.Tensor") -> None:
        """Register ModelRunner's [2, layers, blocks, block_size, heads, dim] cache.

        Retain registration for the cache lifetime. Validate remote model,
        dtype and layout compatibility before transferring into this memory.
        """
        raise NotImplementedError

    @abstractmethod
    def start_load_kv(self) -> None:
        """Consume bound loads, pending reads and cancels without waiting for I/O.

        Called once per binding, even with no model work. Retain task metadata
        and handles across steps. In pull mode sends announce retained P KV,
        not a push operation. Producer-write readiness must be enforced before
        remote reads; CPU command order alone is insufficient.
        """
        raise NotImplementedError

    @abstractmethod
    def get_finished(self) -> ConnectorOutput:
        """Progress/poll transfers without waiting; drain new terminal events.

        Invoke even on steps without a forward pass. received requires valid,
        GPU-visible KV; sent requires completed remote access. Failed/cancelled
        events require stopped memory accesses. A timeout alone cannot satisfy
        that contract. Report each local attempt's terminal outcome once.
        """
        raise NotImplementedError

    @abstractmethod
    def has_pending_work(self) -> bool:
        """Include handshake, transfer, cancellation and pending-read work."""
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Quiesce all memory accesses before unregistering KV and closing transport.

        Must not return successfully while transport can still access the cache.
        """
        raise NotImplementedError
