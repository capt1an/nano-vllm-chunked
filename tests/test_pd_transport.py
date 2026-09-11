"""Protect lease safety without requiring NIXL or a GPU."""
from queue import SimpleQueue
import threading
import unittest
from unittest.mock import Mock

from nanovllm.distributed.kv_transfer import ConnectorOutput, KVTransferParams
from nanovllm.distributed.kv_transfer.nixl import NixlWorkerConnector


class LeaseTest(unittest.TestCase):
    def worker(self):
        worker = object.__new__(NixlWorkerConnector)
        worker._lock = threading.Lock()
        worker._released = SimpleQueue()
        worker._published = {}
        worker._loads = {}
        worker._cancelled = set()
        worker._events = ConnectorOutput()
        worker.info = dict(model_id='model', dtype='torch.float16', shape=[2, 1, 4, 256, 1, 2])
        params = KVTransferParams('xfer', 'p', 'request', '127.0.0.1', 5600, (1,), 5, 256)
        worker._published['xfer'] = dict(params=params, reader=None, released=False,
                                         cancelled=False, deadline=0)
        return worker

    def acquire(self, worker, reader='d', signature=None):
        return worker._control(dict(op='acquire', transfer_id='xfer', reader=reader,
                                    remote_request_id='request',
                                    signature=signature or worker._signature(worker.info)))

    def test_claimed_blocks_do_not_expire_or_free_on_cancellation(self):
        worker = self.worker()
        self.assertEqual(self.acquire(worker)['status'], 'ok')
        self.assertFalse(worker.get_finished().sent)
        self.assertIn('xfer', worker._published)
        worker._published['xfer']['cancelled'] = True
        self.assertFalse(worker.get_finished().cancelled)
        self.assertIn('xfer', worker._published)
        worker._control(dict(op='release', transfer_id='xfer', reader='d'))
        self.assertEqual(worker.get_finished().cancelled, {'xfer'})
        self.assertFalse(worker.has_pending_work())

    def test_only_claiming_peer_can_release_and_ack_is_idempotent(self):
        worker = self.worker()
        self.acquire(worker)
        self.assertEqual(self.acquire(worker, 'other')['status'], 'error')
        worker._control(dict(op='release', transfer_id='xfer', reader='other'))
        self.assertIn('xfer', worker._published)
        worker._control(dict(op='release', transfer_id='xfer', reader='d'))
        self.assertEqual(worker.get_finished().sent, {'xfer'})
        worker._control(dict(op='release', transfer_id='xfer', reader='d'))
        self.assertFalse(worker.get_finished().sent)

    def test_unclaimed_blocks_expire_and_mismatch_cannot_claim(self):
        worker = self.worker()
        self.assertEqual(self.acquire(worker, signature=['wrong'])['status'], 'error')
        output = worker.get_finished()
        self.assertEqual(output.failures[0].transfer_id, 'xfer')
        self.assertFalse(worker.has_pending_work())

    def test_failed_backend_cancellation_keeps_handle_and_does_not_ack(self):
        worker = self.worker()
        worker.agent = Mock()
        worker.agent.release_xfer_handle.side_effect = RuntimeError('still active')
        worker._pool = Mock()
        handle = object()
        record = dict(handle=handle, task=object(), stage='read')
        with self.assertRaisesRegex(RuntimeError, 'still active'):
            worker._finish_read(record)
        self.assertIs(record['handle'], handle)
        self.assertEqual(record['stage'], 'read')
        worker._pool.submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
