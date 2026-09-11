"""Real GPU PD stress validation. Run with PYTHONPATH=. .venv/bin/python tests/pd_stress.py MODEL."""
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

from nanovllm import LLM, SamplingParams
from nanovllm.distributed.kv_transfer import PDConfig


def worker(pipe, model, role, options):
    engine = None
    try:
        engine = LLM(model, pd_config=PDConfig(role, port=options['port'] + (role == 'decode'),
                     transfer_timeout=120), distributed_init_method=f"tcp://127.0.0.1:{options['port'] + 100 + (role == 'decode')}",
                     max_model_len=1024, max_num_batched_tokens=128,
                     max_num_seqs=options['concurrency'], gpu_memory_utilization=.3,
                     enforce_eager=options['eager'])
        pipe.send(('ready', None))
        ids = {}
        peak = dict(running=0, remote_waiting=0, waiting=0, pending=0)
        drain = False
        while True:
            while pipe.poll():
                command, data = pipe.recv()
                if command == 'stop':
                    engine.exit()
                    pipe.send(('stopped', None))
                    return
                if command == 'add':
                    rid, prompt, params, handoff = data
                    engine.add_request(prompt, params, request_id=rid, kv_transfer_params=handoff)
                    if role == 'decode':
                        ids[engine.scheduler.requests[rid].seq_id] = rid
                if command == 'drain':
                    drain = True
            s = engine.scheduler
            for key, collection in [('running', s.running), ('remote_waiting', s.remote_waiting),
                                    ('waiting', s.waiting), ('pending', s.connector.pending)]:
                peak[key] = max(peak[key], len(collection))
            if not engine.is_finished():
                finished, _, _ = engine.step()
                for handoff in engine.get_kv_handoffs():
                    pipe.send(('handoff', handoff))
                for sid, tokens in finished:
                    pipe.send(('done', (ids.pop(sid), tokens)))
                assert not engine.get_transfer_failures(), 'transfer failure'
            elif drain:
                b = s.block_manager
                assert not s.requests and not s.remote_waiting and not s.cancelled
                assert not b.reserved_block_ids and not s.connector.has_pending_work()
                assert not engine.model_runner.connector.has_pending_work()
                assert len(b.free_block_ids) == len(b.blocks)
                assert all(block.ref_count == 0 for block in b.blocks)
                assert not ids
                pipe.send(('drained', dict(peak=peak, free_blocks=len(b.free_block_ids))))
                peak = dict.fromkeys(peak, 0)
                drain = False
            time.sleep(.001)
    except BaseException:
        pipe.send(('error', traceback.format_exc()))
    finally:
        if engine is not None:
            engine.exit()
        pipe.close()


def run(args, prompts, params):
    ctx = mp.get_context('spawn')
    workers = []
    records = []
    previous = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        for i, role in enumerate(('prefill', 'decode')):
            parent, child = ctx.Pipe()
            os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
            process = ctx.Process(target=worker, args=(child, args.model, role, vars(args)))
            process.start()
            child.close()
            workers.append((parent, process))
        for pipe, _ in workers:
            assert pipe.poll(300), 'startup timeout'
            event, data = pipe.recv()
            assert event == 'ready', data
        for wave in range(args.waves):
            start = time.monotonic()
            outputs, handoffs, drained = {}, set(), {}
            sent_drain = False
            for i, prompt in enumerate(prompts):
                workers[0][0].send(('add', (f'{wave}:{i}', prompt, params, None)))
            while len(drained) < 2:
                assert time.monotonic() - start < 600, 'wave timeout'
                for index, (pipe, process) in enumerate(workers):
                    assert process.is_alive(), f'worker exited: {process.exitcode}'
                    while pipe.poll():
                        event, data = pipe.recv()
                        if event == 'error':
                            raise RuntimeError(data)
                        if event == 'handoff':
                            rid, handoff = data
                            assert rid not in handoffs, 'duplicate handoff'
                            handoffs.add(rid)
                            workers[1][0].send(('add', (rid, prompts[int(rid.split(':')[1])], params, handoff)))
                        elif event == 'done':
                            rid, tokens = data
                            assert rid in handoffs and rid not in outputs, 'missing handoff / duplicate output'
                            assert len(tokens) == params.max_tokens, 'truncated output'
                            outputs[rid] = tokens
                        elif event == 'drained':
                            drained[index] = data
                if len(outputs) == len(prompts) and not sent_drain:
                    for pipe, _ in workers:
                        pipe.send(('drain', None))
                    sent_drain = True
                time.sleep(.001)
            record = dict(wave=wave, seconds=time.monotonic()-start, resources=drained,
                          outputs=[outputs[f'{wave}:{i}'] for i in range(len(prompts))])
            records.append(record)
            print(json.dumps({k:v for k,v in record.items() if k != 'outputs'}), flush=True)
        for pipe, process in reversed(workers):
            pipe.send(('stop', None))
            assert pipe.poll(150), 'shutdown timeout'
            event, data = pipe.recv()
            assert event == 'stopped', data
            process.join(20)
            assert process.exitcode == 0, f'shutdown exit {process.exitcode}'
        return records
    finally:
        if previous is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = previous
        for pipe, process in reversed(workers):
            if process.is_alive():
                process.terminate()
                process.join(10)
            pipe.close()


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--concurrency', type=int, default=32)
    parser.add_argument('--requests', type=int, default=128)
    parser.add_argument('--waves', type=int, default=3)
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--port', type=int, default=5700)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--report', default='/tmp/pd-stress.json')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = tokenizer.encode('The capital of France is Paris. Water freezes at zero degrees Celsius. ')
    lengths = [1, 31, 127, 128, 129, 255, 256, 257, 383, 511, 512, 513]
    unique = [(base * (n // len(base) + 1))[:n] for n in lengths]
    prompts = [unique[i % len(unique)] for i in range(args.requests)]
    params = SamplingParams(temperature=1e-6, max_tokens=args.max_tokens, ignore_eos=True)
    records = run(args, prompts, params)
    baseline = LLM(args.model, enforce_eager=True, max_model_len=1024,
                   max_num_batched_tokens=128, max_num_seqs=args.concurrency, gpu_memory_utilization=.3)
    try:
        expected = [baseline.generate([p], params, use_tqdm=False)[0]['token_ids'] for p in unique]
    finally:
        baseline.exit()
    mismatches = []
    for record in records:
        for i, actual in enumerate(record['outputs']):
            reference = expected[i % len(unique)]
            if actual != reference:
                first = next(j for j,(a,b) in enumerate(zip(actual, reference)) if a != b)
                mismatches.append(dict(wave=record['wave'], request=i, prompt_length=len(prompts[i]), first_difference=first,
                                       actual=actual, expected=reference))
    report = dict(options=vars(args), records=records, baseline=expected, mismatches=mismatches,
                  total_requests=args.requests*args.waves)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"Completed {report['total_requests']} requests; {len(mismatches)} baseline mismatches; report: {args.report}", flush=True)


if __name__ == '__main__':
    main()
