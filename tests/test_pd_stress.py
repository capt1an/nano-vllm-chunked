"""Deterministic pressure and delayed-event tests of scheduler ownership."""
import random
import unittest
from types import SimpleNamespace
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.distributed.kv_transfer import PDConfig, KVTransferParams, ConnectorOutput, KVTransferFailure


class SchedulerPressureTest(unittest.TestCase):
    def test_thousand_requests_delayed_out_of_order_cancel_failure_and_reuse(self):
        old = Sequence.block_size
        Sequence.block_size = 4
        self.addCleanup(setattr, Sequence, 'block_size', old)
        rng = random.Random(20260910)
        s = Scheduler(SimpleNamespace(max_num_seqs=32, max_num_batched_tokens=13,
                      eos=99, kvcache_block_size=4, enable_continuous_batching=True,
                      enable_chunked_prefill=True, num_kvcache_blocks=48,
                      pd_config=PDConfig('decode'), max_model_len=64))
        counts = dict(done=0, cancelled=0, failed=0)
        for wave in range(10):
            for i in range(100):
                rid = f'{wave}:{i}'
                n = rng.randint(1, 31)
                remote = KVTransferParams(rid, 'p', rid, 'localhost', 5600,
                                          tuple(range((n+3)//4)), n, 4)
                s.add(Sequence(list(range(n)), SamplingParams(max_tokens=rng.randint(1, 17)),
                               request_id=rid, kv_transfer_params=remote))
            pending = []
            for step in range(10000):
                if s.is_finished():
                    break
                out = s.schedule()
                self.assertLessEqual(sum(q.num_scheduled_tokens for q in out.seqs), 13)
                self.assertLessEqual(len(s.running)+len(s.remote_waiting), 32)
                self.assertTrue(all(q.request_id not in s.remote_waiting for q in out.seqs))
                for task in out.connector_metadata.loads:
                    kind = rng.choice(['received'] * 8 + ['cancelled', 'failed'])
                    if kind == 'cancelled':
                        s.cancel_request(task.request_id)
                    pending.append((step+rng.randint(1, 15), task.remote.transfer_id, kind))
                s.postprocess(out, [7]*len(out.seqs))
                counts['done'] += sum(q.is_finished for q in out.seqs)
                rng.shuffle(pending)
                for due, tid, kind in pending[:]:
                    if due > step:
                        continue
                    event = (ConnectorOutput(failures=[KVTransferFailure(tid, 'injected')]) if kind == 'failed'
                             else ConnectorOutput(**{kind: {tid}}))
                    s.update_connector_output(event)
                    s.update_connector_output(event)  # stale duplicate must not affect reused blocks
                    if kind != 'received':
                        counts[kind] += 1
                    pending.remove((due, tid, kind))
                b = s.block_manager
                owned = [bid for q in s.requests.values() for bid in q.block_table]
                owned += [bid for reserved in b.reserved_block_ids.values() for bid in reserved]
                self.assertEqual(len(owned), len(set(owned)))
                self.assertEqual(set(owned) | set(b.free_block_ids), set(range(48)))
                self.assertFalse(set(owned) & set(b.free_block_ids))
                self.assertTrue(all(block.ref_count == int(block.block_id in owned) for block in b.blocks))
            else:
                self.fail('scheduler made no bounded progress')
            self.assertFalse(s.requests)
            self.assertFalse(pending)
            self.assertFalse(s.block_manager.reserved_block_ids)
            self.assertEqual(len(s.block_manager.free_block_ids), 48)
            s.failures.clear()
        self.assertEqual(sum(counts.values()), 1000)
        self.assertTrue(all(counts.values()), counts)


if __name__ == '__main__':
    unittest.main()
