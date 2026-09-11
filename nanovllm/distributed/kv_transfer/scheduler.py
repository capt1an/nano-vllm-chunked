"""Request-level pull connector bookkeeping, owned by the scheduling thread."""
from uuid import uuid4

from nanovllm.distributed.kv_transfer.base import SchedulerConnector
from nanovllm.distributed.kv_transfer.config import PDRole
from nanovllm.distributed.kv_transfer.metadata import (
    ConnectorMetadata, KVLoadTask, KVTransferParams,
)


class PullSchedulerConnector(SchedulerConnector):
    def __init__(self, config):
        super().__init__(config)
        self.pending = {}  # transfer_id -> request_id
        self._commands = ConnectorMetadata()

    def get_num_new_matched_tokens(self, seq, num_computed_tokens):
        params = seq.kv_transfer_params
        if self.config.role != PDRole.DECODE or params is None:
            return 0, False
        count = max(0, params.num_tokens - num_computed_tokens)
        return count, count > 0

    def update_state_after_alloc(self, seq, block_ids, num_external_tokens):
        if not num_external_tokens:
            return
        params = seq.kv_transfer_params
        if params.transfer_id in self.pending:
            raise ValueError("Transfer attempt already admitted")
        self.pending[params.transfer_id] = seq.request_id
        self._commands.loads.append(KVLoadTask(seq.request_id, params, block_ids))

    def build_connector_meta(self, output):
        commands, self._commands = self._commands, ConnectorMetadata()
        return commands

    def request_finished(self, seq, block_ids):
        if self.config.role != PDRole.PREFILL:
            return False, None
        params = KVTransferParams(
            transfer_id=uuid4().hex, remote_engine_id=self.config.engine_id,
            remote_request_id=seq.request_id, remote_host=self.config.host,
            remote_port=self.config.port, remote_block_ids=block_ids,
            num_tokens=seq.num_prompt_tokens, block_size=seq.block_size,
        )
        self.pending[params.transfer_id] = seq.request_id
        self._commands.sends.append(params)
        return True, params

    def cancel_request(self, seq):
        self._commands.cancels.update(
            key for key, request_id in self.pending.items() if request_id == seq.request_id
        )

    def update_connector_output(self, output):
        for key in output.received | output.sent | output.cancelled | {
            failure.transfer_id for failure in output.failures
        }:
            self.pending.pop(key, None)

    def has_pending_work(self):
        return bool(self.pending or self._commands.cancels)

    def shutdown(self):
        if self.pending:
            raise RuntimeError("Cannot close scheduler connector with active transfers")
        self._commands = ConnectorMetadata()
