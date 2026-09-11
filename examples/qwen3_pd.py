"""Independent scheduled P/D engines, with NIXL GPU READ and control-only pipes.

python -m examples.qwen3_pd /path/to/Qwen3 --eager --check
"""
import argparse
import multiprocessing as mp
import os
import time
import traceback

from nanovllm import LLM, SamplingParams
from nanovllm.distributed.kv_transfer import PDConfig


def engine_loop(connection, model, role, port, eager, timeout):
    engine = None
    try:
        engine = LLM(model, pd_config=PDConfig(role, port=port, transfer_timeout=timeout),
                     distributed_init_method=f'tcp://127.0.0.1:{port + 100}',
                     max_model_len=512, max_num_batched_tokens=128, max_num_seqs=8,
                     gpu_memory_utilization=0.3, enforce_eager=eager)
        connection.send(('ready', role))
        ids = {}
        stopping = False
        while not stopping:
            while connection.poll():
                command, data = connection.recv()
                if command == 'stop':
                    stopping = True
                    break
                if command == 'add':
                    request_id, prompt, params, handoff = data
                    engine.add_request(prompt, params, request_id=request_id, kv_transfer_params=handoff)
                    if role == 'decode':
                        ids[engine.scheduler.requests[request_id].seq_id] = request_id
                elif command == 'cancel':
                    engine.cancel_request(data)
            if stopping:
                break
            if not engine.is_finished():
                finished, _, _ = engine.step()
                for request_id, handoff in engine.get_kv_handoffs():
                    connection.send(('handoff', (request_id, handoff)))
                for seq_id, tokens in finished:
                    connection.send(('done', (ids.pop(seq_id), tokens)))
                for failure in engine.get_transfer_failures():
                    connection.send(('error', failure))
            time.sleep(0.001)
    except BaseException:
        connection.send(('error', traceback.format_exc()))
    finally:
        if engine is not None:
            engine.exit()
        connection.close()


def run_pd(model, prompts, params, eager=True, devices=('0', '1'), port=5600, timeout=60):
    """Minimal router: route handoffs, never KV tensors; both engines step independently."""
    context = mp.get_context('spawn')
    workers = []
    try:
        for index, role in enumerate(('prefill', 'decode')):
            parent, child = context.Pipe()
            process = context.Process(target=engine_loop,
                                      args=(child, model, role, port + index, eager, timeout))
            previous = os.environ.get('CUDA_VISIBLE_DEVICES')
            try:
                # Set before spawn: CUDA extensions are imported before engine_loop.
                os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[index])
                process.start()
            finally:
                if previous is None:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = previous
            child.close()
            workers.append((parent, process))
        for connection, _ in workers:
            if not connection.poll(300):
                raise TimeoutError('Engine startup timed out')
            event, value = connection.recv()
            if event != 'ready':
                raise RuntimeError(value)
        for index, prompt in enumerate(prompts):
            workers[0][0].send(('add', (str(index), prompt, params, None)))
        results = {}
        deadline = time.monotonic() + max(300, timeout * 2)
        while len(results) < len(prompts):
            if time.monotonic() > deadline:
                raise TimeoutError('PD generation timed out')
            for connection, process in workers:
                if not process.is_alive():
                    raise RuntimeError('PD worker exited unexpectedly')
                while connection.poll():
                    event, value = connection.recv()
                    if event == 'error':
                        raise RuntimeError(value)
                    if event == 'handoff':
                        request_id, handoff = value
                        workers[1][0].send(('add', (request_id, prompts[int(request_id)], params, handoff)))
                    elif event == 'done':
                        request_id, tokens = value
                        results[int(request_id)] = tokens
            time.sleep(0.001)
        return [results[i] for i in range(len(prompts))]
    finally:
        # Stop D first so its reads/ACKs can finish while P remains available.
        for connection, process in reversed(workers):
            if process.is_alive():
                try:
                    connection.send(('stop', None))
                except (OSError, EOFError):
                    pass
                process.join(timeout=timeout + 5)
            if process.is_alive():
                process.terminate()
                process.join()
            connection.close()


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--prefill-device', default='0')
    parser.add_argument('--decode-device', default='1')
    parser.add_argument('--port', type=int, default=5600)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.prefill_device == args.decode_device:
        parser.error('Use two different GPUs')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    short = tokenizer.encode('The capital of France is')
    question = tokenizer.encode(tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'What is 2 + 3? Answer only the number.'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False))
    prompts = [question, (short * 256)[:256], (short * 257)[:257]]
    params = SamplingParams(temperature=1e-6, max_tokens=8)
    outputs = run_pd(args.model, prompts, params, args.eager,
                     (args.prefill_device, args.decode_device), args.port)
    for tokens in outputs:
        print(tokenizer.decode(tokens))
    if args.check:
        # All PD workers have stopped before baseline allocation.
        baseline = LLM(args.model, enforce_eager=True, max_model_len=512,
                       max_num_batched_tokens=128, max_num_seqs=8, gpu_memory_utilization=0.3)
        try:
            expected = [baseline.generate([p], params, use_tqdm=False)[0]['token_ids'] for p in prompts]
        finally:
            baseline.exit()
        assert outputs == expected, (outputs, expected)
        print('PASS: concurrent scheduled PD matches baseline at short/256/257 token lengths')


if __name__ == '__main__':
    main()
