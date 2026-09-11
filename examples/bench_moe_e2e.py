"""Full-model EP=2 benchmark; run identical script against each revision via PYTHONPATH.

Includes engine scheduling, model forward, collectives, sampling and detokenization.
Excludes model loading, tokenization, warmup and prefix-cache reset. CUDA Graph,
continuous batching and chunked prefill are disabled for a common baseline.
"""
import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter


MODEL = '/home/huggingface/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--output', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output-tokens', type=int, default=64)
    args = parser.parse_args()
    import torch
    import triton
    import flash_attn
    import nanovllm
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.block_manager import BlockManager
    from transformers import AutoTokenizer

    source = Path(nanovllm.__file__).resolve().parents[1]
    print(f'Source: {source}; label: {args.label}', flush=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cases = [(1, 128), (8, 128), (1, 1024)]
    config = dict(tensor_parallel_size=2, enable_expert_parallel=True,
                  enforce_eager=True, max_model_len=1152, max_num_batched_tokens=1024,
                  max_num_seqs=8, gpu_memory_utilization=0.85,
                  enable_continuous_batching=False, enable_chunked_prefill=False)
    result = dict(label=args.label, source=str(source), model=args.model, config=config,
                  torch=torch.__version__, triton=triton.__version__,
                  flash_attn=flash_attn.__version__, python=platform.python_version(),
                  gpu=[torch.cuda.get_device_name(i) for i in range(2)],
                  output_tokens=args.output_tokens, repeats=args.repeats, cases=[])
    start = perf_counter()
    llm = LLM(args.model, **config)
    result['initialization_s'] = perf_counter() - start
    print(f'Initialized in {result["initialization_s"]:.2f}s', flush=True)
    try:
        for batch, length in cases:
            prompts = []
            for i in range(batch):
                text = (f'任务编号{i}。请阅读以下材料并解释混合专家模型推理中的计算与通信。\n' +
                        '混合专家模型为每个输入选择部分专家。张量并行拆分矩阵运算，专家并行分配不同专家。' * 300)
                prompts.append(tokenizer.encode(text, add_special_tokens=False)[:length])
            assert all(len(p) == length for p in prompts)
            case = dict(batch=batch, input_tokens_per_request=length,
                        prompt_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(), runs=[])
            params = SamplingParams(temperature=1e-6, max_tokens=args.output_tokens, ignore_eos=True)
            # First full generation warms both prefill and all decode shapes for this case.
            for repeat in range(-1, args.repeats):
                assert llm.is_finished()
                bm = llm.scheduler.block_manager
                llm.scheduler.block_manager = BlockManager(len(bm.blocks), bm.block_size)
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                steps = []
                original_step = llm.step
                begin = 0.0
                def measured_step():
                    t = perf_counter()
                    finished, prefill, decode = original_step()
                    now = perf_counter()
                    steps.append(dict(elapsed_s=now - t, prefill_tokens=prefill,
                                      decode_tokens=decode, since_start_s=now - begin))
                    return finished, prefill, decode
                llm.step = measured_step
                try:
                    torch.cuda.synchronize()
                    begin = perf_counter()
                    outputs = llm.generate(prompts, params, use_tqdm=False)
                    torch.cuda.synchronize()
                    elapsed = perf_counter() - begin
                finally:
                    llm.step = original_step
                assert sum(s['prefill_tokens'] for s in steps) == batch * length
                assert len(steps) == args.output_tokens
                assert all(len(o['token_ids']) == args.output_tokens for o in outputs)
                decode_steps = [s for s in steps if s['decode_tokens']]
                run = dict(repeat=repeat, total_s=elapsed, ttft_s=steps[0]['since_start_s'],
                           prefill_step_s=steps[0]['elapsed_s'],
                           decode_step_mean_s=statistics.mean(s['elapsed_s'] for s in decode_steps),
                           decode_tokens_per_s=sum(s['decode_tokens'] for s in decode_steps) /
                                               sum(s['elapsed_s'] for s in decode_steps),
                           output_tokens_per_s=batch * args.output_tokens / elapsed,
                           outputs=outputs, steps=steps)
                print(f'{args.label} B={batch} input={length} rep={repeat}: '
                      f'total={elapsed:.3f}s TTFT={run["ttft_s"]:.3f}s '
                      f'decode_step={run["decode_step_mean_s"]*1000:.2f}ms '
                      f'output={run["output_tokens_per_s"]:.2f}tok/s', flush=True)
                if repeat >= 0:
                    case['runs'].append(run)
            case['median'] = {key: statistics.median(r[key] for r in case['runs']) for key in (
                'total_s', 'ttft_s', 'prefill_step_s', 'decode_step_mean_s',
                'decode_tokens_per_s', 'output_tokens_per_s')}
            result['cases'].append(case)
            Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    finally:
        llm.exit()


if __name__ == '__main__':
    main()
