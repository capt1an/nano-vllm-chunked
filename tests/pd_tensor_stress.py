"""32 concurrent transfers across 10 waves, exact KV copy check; run PYTHONPATH=. python tests/pd_tensor_smoke.py."""
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
        capacity = 96 if role == 'prefill' else 112
        cache = torch.full((2, 2, capacity, 256, 2, 4), -1., device='cuda', dtype=torch.float16)
        connector = NixlWorkerConnector(PDConfig(role, engine_id=role, port=5610 if role == 'prefill' else 5611), 'tensor-test')
        connector.register_kv_cache(cache)
        for wave in range(10):
            reference = ((torch.arange(2*2*96*256*2*4, device='cuda').reshape(2, 2, 96, 256, 2, 4)
                          + wave * 101) % 997).half()
            cache.fill_(-1)
            if role == 'prefill':
                cache.copy_(reference)
            params = [KVTransferParams(f'{wave}:{i}', 'prefill', str(i), '127.0.0.1', 5610,
                      (2*i+1, 2*i), 257, 256) for i in range(32)]
            commands = ConnectorMetadata(sends=params) if role == 'prefill' else ConnectorMetadata(
                loads=[KVLoadTask(str(i), param, (2*i+7, 2*i+6)) for i, param in enumerate(params)])
            connector.bind_connector_metadata(commands)
            connector.start_load_kv()
            connector.clear_connector_metadata()
            deadline = time.monotonic() + 90
            completed = set()
            while time.monotonic() < deadline:
                output = connector.get_finished()
                if output.failures:
                    raise RuntimeError(output.failures)
                terminal = output.sent | output.received
                assert not completed & terminal
                completed |= terminal
                if len(completed) == 32:
                    break
                time.sleep(.001)
            else:
                raise TimeoutError('Tensor transfer timed out')
            assert not connector.has_pending_work()
            if role == 'decode':
                expected = torch.full_like(cache, -1)
                for i in range(32):
                    expected[:, :, 2*i+7] = reference[:, :, 2*i+1]
                    expected[:, :, 2*i+6, :1] = reference[:, :, 2*i, :1]
                torch.testing.assert_close(cache, expected, rtol=0, atol=0)
            pipe.send(('ok', (role, wave)))
            pipe.recv()  # Barrier before reusing either side's storage.
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
        for wave in range(10):
            for pipe, _ in workers:
                assert pipe.poll(120), 'worker timeout'
                status, result = pipe.recv()
                assert status == 'ok', result
                assert result[1] == wave
            for pipe, _ in workers:
                pipe.send('next')
        print('PASS: 320 exact GPU transfers, 32 concurrent, 10 reuse waves, remapped blocks and untouched padding')
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
            assert process.exitcode == 0, f"worker exit: {process.exitcode}"


if __name__ == '__main__':
    main()
