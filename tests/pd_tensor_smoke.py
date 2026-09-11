"""Two-process exact KV copy check; run PYTHONPATH=. python tests/pd_tensor_smoke.py."""
import multiprocessing as mp
import os
import time
import traceback

import torch

from nanovllm.distributed.kv_transfer import PDConfig, ConnectorMetadata, KVTransferParams, KVLoadTask
from nanovllm.distributed.kv_transfer.nixl import NixlWorkerConnector


def worker(pipe, role):
    connector = None
    try:
        torch.cuda.set_device(0)
        capacity = 6 if role == 'prefill' else 7
        cache = torch.full((2, 2, capacity, 256, 2, 4), -1., device='cuda', dtype=torch.float16)
        reference = (torch.arange(2*2*6*256*2*4, device='cuda').reshape(2, 2, 6, 256, 2, 4) % 997).half()
        if role == 'prefill':
            cache.copy_(reference)
        connector = NixlWorkerConnector(PDConfig(role, engine_id=role, port=5610 if role == 'prefill' else 5611), 'tensor-test')
        connector.register_kv_cache(cache)
        params = KVTransferParams('attempt', 'prefill', 'request', '127.0.0.1', 5610, (3, 1), 257, 256)
        commands = ConnectorMetadata(sends=[params]) if role == 'prefill' else ConnectorMetadata(
            loads=[KVLoadTask('d-request', params, (1, 4))])
        connector.bind_connector_metadata(commands)
        connector.start_load_kv()
        connector.clear_connector_metadata()
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            output = connector.get_finished()
            if output.failures:
                raise RuntimeError(output.failures)
            if output.sent or output.received:
                if role == 'decode':
                    torch.testing.assert_close(cache[:, :, 1], reference[:, :, 3], rtol=0, atol=0)
                    torch.testing.assert_close(cache[:, :, 4, :1], reference[:, :, 1, :1], rtol=0, atol=0)
                    self_tail = cache[:, :, 4, 1:]
                    assert torch.all(self_tail == -1)
                    assert torch.all(cache[:, :, 0] == -1)
                pipe.send(('ok', role))
                break
            time.sleep(.001)
        else:
            raise TimeoutError('Tensor transfer timed out')
        pipe.recv()  # Keep both registered agents alive until both sides finish.
    except BaseException:
        pipe.send(('error', traceback.format_exc()))
    finally:
        if connector:
            connector.shutdown()


def main():
    ctx = mp.get_context('spawn')
    workers = []
    previous = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        for index, role in enumerate(('prefill', 'decode')):
            parent, child = ctx.Pipe()
            os.environ['CUDA_VISIBLE_DEVICES'] = str(index)
            process = ctx.Process(target=worker, args=(child, role))
            process.start()
            child.close()
            workers.append((parent, process))
        for pipe, _ in workers:
            assert pipe.poll(120), 'worker timeout'
            status, result = pipe.recv()
            assert status == 'ok', result
        print('PASS: exact GPU KV transfer, remapped blocks, different capacities, untouched padding')
    finally:
        if previous is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = previous
        for pipe, process in reversed(workers):
            if process.is_alive():
                pipe.send('stop')
                process.join(10)
            if process.is_alive():
                process.terminate()
                process.join()


if __name__ == '__main__':
    main()
