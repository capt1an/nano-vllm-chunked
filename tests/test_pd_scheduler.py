import unittest
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.distributed.kv_transfer import PDConfig, ConnectorOutput, KVTransferParams, KVTransferFailure
from nanovllm.distributed.kv_transfer.nixl import block_descriptors


class PDSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.old_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old_size

    def scheduler(self, role, blocks=12):
        return Scheduler(SimpleNamespace(
            max_num_seqs=3, max_num_batched_tokens=3, eos=99,
            kvcache_block_size=4, enable_continuous_batching=True,
            enable_chunked_prefill=True, num_kvcache_blocks=blocks,
            pd_config=PDConfig(role), max_model_len=48,
        ))

    def request(self, name, length=5, max_tokens=3, remote=False):
        params = KVTransferParams(name + '-xfer', 'p', name, '127.0.0.1', 5600,
                                  tuple(range((length+3)//4)), length, 4) if remote else None
        return Sequence(list(range(length)), SamplingParams(max_tokens=max_tokens),
                        request_id=name, kv_transfer_params=params)

    def test_prefill_holds_blocks_until_remote_read_finishes(self):
        scheduler = self.scheduler('prefill', blocks=2)
        first, second = self.request('a'), self.request('b')
        scheduler.add(first)
        scheduler.add(second)
        for _ in range(2):
            output = scheduler.schedule()
            scheduler.postprocess(output, [7] * len(output.seqs))
        self.assertEqual(first.status, SequenceStatus.FINISHED)
        self.assertEqual(len(first.block_table), 2)
        self.assertFalse(scheduler.is_finished())
        request_id, handoff = scheduler.handoffs.popleft()
        output = scheduler.schedule()
        self.assertEqual(output.seqs, [])
        self.assertEqual(output.connector_metadata.sends, [handoff])
        scheduler.update_connector_output(ConnectorOutput(sent={handoff.transfer_id}))
        self.assertEqual(first.block_table, [])
        self.assertEqual(scheduler.schedule().seqs, [second])

    def test_waiting_transfer_does_not_block_ready_decode(self):
        scheduler = self.scheduler('decode')
        slow, ready = self.request('slow', remote=True), self.request('ready', remote=True)
        scheduler.add(slow)
        scheduler.add(ready)
        output = scheduler.schedule()
        self.assertEqual(len(output.connector_metadata.loads), 2)
        self.assertEqual(output.seqs, [])
        self.assertEqual(slow.num_cached_tokens, 0)
        scheduler.update_connector_output(ConnectorOutput(received={'ready-xfer'}))
        output = scheduler.schedule()
        self.assertEqual(output.seqs, [ready])
        self.assertEqual(output.num_prefill_tokens, 1)
        scheduler.postprocess(output, [7])
        output = scheduler.schedule()
        self.assertEqual(output.decode_seqs, [ready])
        self.assertEqual(slow.status, SequenceStatus.WAITING_FOR_REMOTE_KVS)
        scheduler.postprocess(output, [7])
        scheduler.update_connector_output(ConnectorOutput(received={'ready-xfer'}))
        self.assertEqual(list(scheduler.running), [ready])

    def test_decode_reservation_crosses_boundary_without_preemption(self):
        scheduler = self.scheduler('decode', blocks=2)
        seq = self.request('a', length=4, max_tokens=4, remote=True)
        scheduler.add(seq)
        scheduler.schedule()
        self.assertEqual(len(seq.block_table), 1)
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 0)
        scheduler.update_connector_output(ConnectorOutput(received={'a-xfer'}))
        for index in range(4):
            output = scheduler.schedule()
            scheduler.postprocess(output, [7])
        self.assertTrue(seq.is_finished)
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 2)
        self.assertFalse(scheduler.block_manager.reserved_block_ids)
        self.assertTrue(scheduler.is_finished())

    def test_cancel_waits_for_terminal_event_and_ignores_late_receive(self):
        scheduler = self.scheduler('decode', blocks=2)
        seq = self.request('a', remote=True)
        scheduler.add(seq)
        scheduler.schedule()
        scheduler.cancel_request('a')
        self.assertTrue(seq.block_table)
        output = scheduler.schedule()
        self.assertEqual(output.connector_metadata.cancels, {'a-xfer'})
        scheduler.update_connector_output(ConnectorOutput(received={'a-xfer'}))
        self.assertFalse(seq.block_table)
        self.assertTrue(scheduler.is_finished())
        scheduler.update_connector_output(ConnectorOutput(received={'a-xfer'}))
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 2)

    def test_receive_failure_releases_unpublished_blocks(self):
        scheduler = self.scheduler('decode', blocks=2)
        seq = self.request('a', remote=True)
        scheduler.add(seq)
        scheduler.schedule()
        scheduler.update_connector_output(ConnectorOutput(failures=[KVTransferFailure('a-xfer', 'failed')]))
        self.assertTrue(scheduler.is_finished())
        self.assertEqual(list(scheduler.failures), [('a', 'failed')])
        self.assertFalse(scheduler.block_manager.hash_to_block_ids)
        with self.assertRaisesRegex(ValueError, 'Duplicate transfer'):
            scheduler.add(self.request('a', remote=True))

    def test_oversized_requests_fail_before_admission(self):
        scheduler = self.scheduler('decode', blocks=1)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            scheduler.add(self.request('a', remote=True))
        self.assertTrue(scheduler.is_finished())

    def test_first_decode_sample_eos_frees_all_reserved_blocks(self):
        scheduler = self.scheduler('decode', blocks=2)
        seq = self.request('a', remote=True)
        scheduler.add(seq)
        scheduler.schedule()
        scheduler.update_connector_output(ConnectorOutput(received={'a-xfer'}))
        scheduler.postprocess(scheduler.schedule(), [99])
        self.assertEqual(seq.completion_token_ids, [99])
        self.assertTrue(scheduler.is_finished())
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 2)

    def test_shutdown_reclaims_retained_ownership_after_worker_quiescence(self):
        scheduler = self.scheduler('decode', blocks=2)
        scheduler.add(self.request('a', remote=True))
        scheduler.schedule()
        scheduler.shutdown()
        self.assertTrue(scheduler.is_finished())
        self.assertFalse(scheduler.requests)
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 2)

    def test_failed_reservation_does_not_consume_remaining_capacity(self):
        scheduler = self.scheduler('decode', blocks=3)
        a, b = self.request('a', remote=True), self.request('b', remote=True)
        scheduler.add(a)
        scheduler.add(b)
        scheduler.schedule()
        self.assertFalse(b.block_table)
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 1)
        self.assertEqual(b.status, SequenceStatus.WAITING)

    def test_descriptors_map_partial_blocks_and_different_pool_capacities(self):
        source = dict(shape=[2, 2, 10, 4, 1, 2], address=1000, itemsize=2, device=0)
        target = dict(source, shape=[2, 2, 20, 4, 1, 2], address=8000, device=1)
        src = block_descriptors(source, (3, 7), 5)
        dst = block_descriptors(target, (11, 2), 5)
        self.assertEqual([length for _, length, _ in src], [16, 4] * 4)
        self.assertEqual(src[0][0], 1000 + 3 * 16)
        self.assertEqual(src[2][0], 1000 + (10 + 3) * 16)
        self.assertEqual(dst[2][0], 8000 + (20 + 11) * 16)
        with self.assertRaises(ValueError):
            block_descriptors(source, (10, 1), 5)


if __name__ == '__main__':
    unittest.main()
