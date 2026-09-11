"""Offline ShareGPT burst benchmark: TP=2 and 1P+1D (with optional single/replica controls).

PYTHONPATH=. python -m examples.bench_pd /path/to/model --dataset /path/to/sharegpt.json
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
from nanovllm import LLM, SamplingParams
from nanovllm.distributed.kv_transfer import PDConfig


def worker(pipe, role, index, options):
    engine = None
    try:
        engine = LLM(options['model'], max_model_len=options['max_model_len'],
                     tensor_parallel_size=2 if role == 'tp' else 1,
                     worker_shm_name=f"nano_bench_{options['port']}",
                     max_num_seqs=options['concurrency'],
                     max_num_batched_tokens=options['token_budget'],
                     gpu_memory_utilization=options['memory_utilization'], enforce_eager=options['eager'],
                     distributed_init_method=f"tcp://127.0.0.1:{options['port']+100+index}",
                     pd_config=PDConfig(role, port=options['port']+index, transfer_timeout=180)
                     if role in ('prefill', 'decode') else None)
        pipe.send(('ready', engine.config.num_kvcache_blocks))
        active = {}
        drain = False
        while True:
            while pipe.poll():
                cmd, data = pipe.recv()
                if cmd == 'stop':
                    engine.exit()
                    pipe.send(('stopped', None))
                    return
                if cmd == 'drain':
                    drain = True
                if cmd == 'add':
                    for rid, tokens, remote in data:
                        engine.add_request(tokens, SamplingParams(temperature=1e-6,
                                           max_tokens=options['output_tokens'], ignore_eos=True),
                                           request_id=rid, kv_transfer_params=remote)
                        seq = engine.scheduler.waiting[-1]
                        if role != 'prefill':
                            active[rid] = (seq, [])
            if not engine.is_finished():
                engine.step()
                now = time.perf_counter()
                events = []
                for rid, (seq, stamps) in list(active.items()):
                    if seq.num_completion_tokens > len(stamps):
                        assert seq.num_completion_tokens == len(stamps)+1
                        stamps.append(now)
                        if len(stamps) == 1:
                            events.append(('first', rid))
                        if seq.is_finished:
                            assert len(stamps) == options['output_tokens']
                            events.append(('done', (rid, stamps, seq.completion_token_ids)))
                            del active[rid]
                for rid, handoff in engine.get_kv_handoffs():
                    events.append(('handoff', (rid, handoff)))
                failures = engine.get_transfer_failures()
                if failures:
                    raise RuntimeError(failures)
                if events:
                    pipe.send(('events', events))
                delay = options['pd_step_sleep'] if role in ('prefill', 'decode') else options['step_sleep']
                if delay:
                    time.sleep(delay)
            elif drain:
                assert not active
                s = engine.scheduler
                assert len(s.block_manager.free_block_ids) == len(s.block_manager.blocks)
                if s.connector:
                    assert not s.requests and not s.connector.has_pending_work()
                    assert not engine.model_runner.connector.has_pending_work()
                pipe.send(('drained', None))
                drain = False
            else:
                time.sleep(.001)
    except BaseException:
        pipe.send(('error', traceback.format_exc()))
    finally:
        if engine is not None:
            engine.exit()
        pipe.close()


def run_mode(mode, workloads, options, save):
    workers = []
    previous = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        roles = (['prefill', 'decode'] if mode == 'pd' else ['tp'] if mode == 'tp'
                 else ['ordinary']*(2 if mode == 'replicas' else 1))
        for index, role in enumerate(roles):
            pipe, child = mp.get_context('spawn').Pipe()
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' if mode == 'tp' else str(index)
            proc = mp.get_context('spawn').Process(target=worker, args=(child, role, index, options))
            proc.start()
            child.close()
            workers.append((pipe, proc))
        capacities = []
        for pipe, _ in workers:
            assert pipe.poll(300), 'startup timeout'
            event, data = pipe.recv()
            assert event == 'ready', data
            capacities.append(data)
        for case, batches in workloads.items():
            for wave, prompts in enumerate(batches):
                start = time.perf_counter()
                first, finished, traces, outputs, handoffs = {}, {}, {}, {}, set()
                for index, (pipe, _) in enumerate(workers):
                    if mode == 'pd' and index == 1:
                        continue
                    batch = [(str(i), p, None) for i,p in enumerate(prompts)
                             if mode != 'replicas' or i % 2 == index]
                    pipe.send(('add', batch))
                while len(finished) != len(prompts):
                    assert time.perf_counter()-start < 600, 'generation timeout'
                    for pipe, proc in workers:
                        assert proc.is_alive(), f'worker exit {proc.exitcode}'
                        while pipe.poll():
                            event, data = pipe.recv()
                            if event == 'error':
                                raise RuntimeError(data)
                            assert event == 'events', (event, data)
                            for event, data in data:
                                now = time.perf_counter()
                                if event == 'handoff':
                                    rid, handoff = data
                                    assert rid not in handoffs
                                    handoffs.add(rid)
                                    workers[1][0].send(('add', [(rid, prompts[int(rid)], handoff)]))
                                elif event == 'first':
                                    assert data not in first
                                    first[data] = now-start
                                elif event == 'done':
                                    rid, stamps, tokens = data
                                    assert rid not in finished and rid in first
                                    finished[rid] = now-start
                                    traces[rid] = stamps
                                    outputs[rid] = tokens
                    time.sleep(.0001)
                duration = max(finished.values())
                for pipe, _ in workers:
                    pipe.send(('drain', None))
                for pipe, _ in workers:
                    assert pipe.poll(180), 'drain timeout'
                    event, data = pipe.recv()
                    assert event == 'drained', (event, data)
                itl = [dt for stamps in traces.values() for dt in np.diff(stamps)]
                tpot = [(x[-1]-x[0])/(len(x)-1) for x in traces.values()]
                record = dict(mode=mode, case=case, wave=wave, warmup=wave==0,
                              gpu_count=2 if mode == 'tp' else len(workers), cache_blocks=capacities,
                              seconds=duration, requests=len(prompts),
                              input_tokens=sum(map(len,prompts)), output_tokens=sum(map(len,outputs.values())),
                              output_tok_s=sum(map(len,outputs.values()))/duration,
                              request_s=len(prompts)/duration,
                              ttft_p50_ms=float(np.percentile(list(first.values()),50)*1000),
                              ttft_p95_ms=float(np.percentile(list(first.values()),95)*1000),
                              e2e_p95_ms=float(np.percentile(list(finished.values()),95)*1000),
                              itl_p50_ms=float(np.percentile(itl,50)*1000),
                              itl_p95_ms=float(np.percentile(itl,95)*1000),
                              tpot_p50_ms=float(np.percentile(tpot,50)*1000),
                              first=first, finished=finished, timestamps=traces, outputs=outputs)
                save(record)
                print(json.dumps({k:v for k,v in record.items()
                                  if k not in ('first','finished','timestamps','outputs')}), flush=True)
        for pipe, proc in reversed(workers):
            pipe.send(('stop',None))
            assert pipe.poll(200), 'shutdown timeout'
            event, data = pipe.recv()
            assert event == 'stopped', data
            proc.join(20)
            assert proc.exitcode == 0, proc.exitcode
    finally:
        if previous is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = previous
        for pipe, proc in reversed(workers):
            if proc.is_alive():
                proc.terminate()
                proc.join(10)
            pipe.close()


def main():
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--workloads-from', help='Reuse exact tokenized batches from a previous report')
    parser.add_argument('--requests',type=int,default=64)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--output-tokens',type=int,default=64)
    parser.add_argument('--concurrency',type=int,default=512)
    parser.add_argument('--token-budget',type=int,default=16384)
    parser.add_argument('--max-model-len',type=int,default=4096)
    parser.add_argument('--memory-utilization',type=float,default=.9)
    parser.add_argument('--port',type=int,default=6100)
    parser.add_argument('--step-sleep',type=float,default=0, help='Ordinary engine per-step delay; default has no delay')
    parser.add_argument('--pd-step-sleep',type=float,default=.001, help='PD per-step yield, matching qwen3_pd example')
    parser.add_argument('--eager',action='store_true')
    parser.add_argument('--modes',nargs='+',default=['tp','pd'],choices=['single','replicas','tp','pd'])
    parser.add_argument('--report',default='/tmp/pd-benchmark.json')
    args=parser.parse_args()
    assert args.requests % 2 == 0 and args.output_tokens > 1
    if args.workloads_from:
        workloads=json.loads(Path(args.workloads_from).read_text())['workloads']
        assert all(len(batches)==args.rounds+1 and all(len(b)==args.requests for b in batches) for batches in workloads.values())
    else:
        tokenizer=AutoTokenizer.from_pretrained(args.model)
        records=json.loads(Path(args.dataset).read_text())
        random.Random(20260911).shuffle(records)
        need=args.requests*(args.rounds+1)*2
        short,long=[],[]
        seen=set()
        for row in records:
            messages=row.get('conversations',[])
            if not messages or messages[0].get('from') not in ('human','user'):
                continue
            tokens=tokenizer.encode(tokenizer.apply_chat_template(
                [{'role':'user','content':messages[0]['value']}], tokenize=False,
                add_generation_prompt=True,enable_thinking=False), truncation=True, max_length=2049)
            identity=tuple(tokens)
            if identity in seen:
                continue
            seen.add(identity)
            if 64 <= len(tokens) <= 256 and len(short)<need:
                short.append(tokens)
            if 1024 <= len(tokens) <= 2048 and len(long)<need:
                long.append(tokens)
            if len(short)==need and len(long)==need:
                break
        assert min(len(short),len(long)) >= need, (len(short),len(long),need)
        n=args.requests
        batches=args.rounds+1
        workloads={
            'short':[short[i*n:(i+1)*n] for i in range(batches)],
            'long':[long[i*n:(i+1)*n] for i in range(batches)],
            'mixed':[short[batches*n+i*n//2:batches*n+(i+1)*n//2]+long[batches*n+i*n//2:batches*n+(i+1)*n//2]
                     for i in range(batches)]}
        for batches in workloads.values():
            for batch in batches:
                random.Random(71).shuffle(batch)
    report=dict(options=vars(args),workloads=workloads,records=[],
                workload_sha256=hashlib.sha256(json.dumps(workloads).encode()).hexdigest())
    def save(record):
        report['records'].append(record)
        Path(args.report).write_text(json.dumps(report))
    for mode in args.modes:
        run_mode(mode,workloads,vars(args),save)


if __name__=='__main__':
    main()
