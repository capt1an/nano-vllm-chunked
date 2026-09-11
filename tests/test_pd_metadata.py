import pickle
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.config import Config
from nanovllm.engine.sequence import SchedulerOutput, Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.distributed.kv_transfer import (
    ConnectorMetadata, ConnectorOutput, KVLoadTask, KVTransferParams,
    PDConfig, PDRole,
)


class PDMetadataTest(unittest.TestCase):
    def test_sequence_ipc_preserves_pd_identity_without_claiming_kv_ready(self):
        seq = Sequence([3] * 257, SamplingParams(temperature=0.5, max_tokens=8),
                       request_id="request-d", kv_transfer_params=self.handoff())
        self.assertEqual(seq.status, SequenceStatus.WAITING)
        self.assertEqual(seq.num_cached_tokens, 0)
        seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
        seq.block_table = [12, 48]
        restored = pickle.loads(pickle.dumps(seq))
        self.assertEqual(restored.token_ids, seq.token_ids)
        self.assertEqual(restored.request_id, seq.request_id)
        self.assertEqual(restored.seq_id, seq.seq_id)
        self.assertEqual(restored.status, SequenceStatus.WAITING_FOR_REMOTE_KVS)
        self.assertEqual(restored.kv_transfer_params, self.handoff())
        self.assertFalse(restored.is_finished)
        self.assertEqual(restored.temperature, 0.5)
        self.assertEqual(restored.max_tokens, 8)

    def test_decode_ipc_remains_compact_and_preserves_identity(self):
        seq = Sequence([3] * 257, kv_transfer_params=self.handoff())
        seq.num_cached_tokens = 257
        seq.append_token(9)
        seq.status = SequenceStatus.RUNNING
        restored = pickle.loads(pickle.dumps(seq))
        self.assertEqual(restored.token_ids, [])
        self.assertEqual(restored.last_token, 9)
        self.assertEqual(len(restored), 258)
        self.assertEqual(restored.request_id, seq.request_id)
        self.assertEqual(restored.kv_transfer_params, seq.kv_transfer_params)
        self.assertFalse(restored.is_prefill)

    def test_sequence_rejects_incompatible_handoff(self):
        with self.assertRaisesRegex(ValueError, 'full prompt'):
            Sequence([1], kv_transfer_params=self.handoff())
        with self.assertRaisesRegex(ValueError, 'block_size'):
            Sequence([1] * 257, kv_transfer_params=replace(
                self.handoff(), block_size=512, remote_block_ids=(37,)))
        with self.assertRaisesRegex(ValueError, 'request_id'):
            Sequence([1], request_id="")

    def test_transfer_only_output_roundtrip(self):
        metadata = ConnectorMetadata(loads=[KVLoadTask("d", self.handoff(), (12, 48))])
        output = SchedulerOutput([], 0, 0, connector_metadata=metadata)
        restored = pickle.loads(pickle.dumps(output))
        self.assertEqual(restored.connector_metadata, metadata)
        self.assertFalse(restored.has_prefill)
        self.assertFalse(restored.has_decode)
        self.assertIsNone(SchedulerOutput([], 0, 0).connector_metadata)

    def handoff(self):
        return KVTransferParams(
            transfer_id="attempt-1", remote_engine_id="prefill",
            remote_request_id="request-p", remote_host="127.0.0.1",
            remote_port=5600, remote_block_ids=(37, 91),
            num_tokens=257, block_size=256,
        )

    def test_partial_block_mapping_survives_ipc(self):
        blocks = [12, 48]
        task = KVLoadTask("request-d", self.handoff(), blocks)
        blocks[0] = 99
        metadata = ConnectorMetadata(loads=[task])
        restored = pickle.loads(pickle.dumps(metadata))
        metadata.loads.clear()
        self.assertEqual(restored.loads[0].local_block_ids, (12, 48))
        self.assertEqual(restored.loads[0].remote.remote_block_ids, (37, 91))
        self.assertEqual(restored.loads[0].remote.num_tokens, 257)

    def test_rejects_incomplete_or_aliased_block_mapping(self):
        for blocks in ((1,), (1, 1), (-1, 2), (1, 2, 3)):
            with self.subTest(blocks=blocks):
                with self.assertRaises(ValueError):
                    replace(self.handoff(), remote_block_ids=blocks)
                with self.assertRaises(ValueError):
                    KVLoadTask("request-d", self.handoff(), blocks)
        self.assertEqual(len(replace(self.handoff(), num_tokens=256,
                                     remote_block_ids=(37,)).remote_block_ids), 1)

    def test_step_events_do_not_leak_into_next_step(self):
        output = ConnectorOutput(received={"attempt-1"})
        self.assertEqual(ConnectorOutput().received, set())
        self.assertEqual(pickle.loads(pickle.dumps(output)), output)
        first = ConnectorMetadata(cancels={"attempt-1"})
        self.assertFalse(ConnectorMetadata().cancels)
        self.assertEqual(first.cancels, {"attempt-1"})

    def test_config_rejects_invalid_roles_endpoints_and_timeouts(self):
        for kwargs in (dict(role="both"), dict(port=0), dict(port=65536),
                       dict(transfer_timeout=0), dict(transfer_timeout=float('nan')),
                       dict(transfer_timeout=float('inf')), dict(engine_id=""),
                       dict(host=""), dict(connector="unknown")):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    PDConfig(**(dict(role="prefill") | kwargs))
        first, second = PDConfig("prefill"), PDConfig("decode", port=5601)
        self.assertEqual(first.role, PDRole.PREFILL)
        self.assertNotEqual(first.engine_id, second.engine_id)

    @patch("nanovllm.config.AutoConfig.from_pretrained")
    def test_pd_constraints_leave_ordinary_tp_unchanged(self, load_config):
        load_config.return_value = SimpleNamespace(
            max_position_embeddings=4096, quantization_config=None,
        )
        ordinary = Config("unused", tensor_parallel_size=2)
        self.assertIsNone(ordinary.pd_config)
        self.assertEqual(ordinary.world_size, 2)
        pd = PDConfig("decode", port=5601)
        self.assertEqual(Config("unused", pd_config=pd).pd_config, pd)
        for kwargs in (dict(tensor_parallel_size=2), dict(enable_expert_parallel=True)):
            with self.assertRaisesRegex(ValueError, 'TP=1'):
                Config("unused", pd_config=pd, **kwargs)


if __name__ == '__main__':
    unittest.main()
