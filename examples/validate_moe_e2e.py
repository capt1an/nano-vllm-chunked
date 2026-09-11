"""Teacher-forced B=8 full-model numerical validation against the legacy EP path.

Run the same file with PYTHONPATH pointing at each revision. Store large logits
under /tmp; compare identical 64-step prefixes twice with fresh KV cache metadata.
"""
import argparse
import json
from pathlib import Path
from types import MethodType

MODEL = '/home/huggingface/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--greedy', action='store_true', help='Free-run exact argmax instead of teacher forcing')
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    import nanovllm
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.block_manager import BlockManager
    from transformers import AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # Use exactly the benchmark's prompts and baseline repeat-0 continuation.
    baseline = json.loads(Path(args.reference).read_text())
    case = baseline['cases'][1]
    assert (case['batch'], case['input_tokens_per_request']) == (8, 128)
    continuation = torch.tensor([o['token_ids'] for o in case['runs'][0]['outputs']], dtype=torch.long)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    prompts = []
    for i in range(8):
        text = (f'任务编号{i}。请阅读以下材料并解释混合专家模型推理中的计算与通信。\n' +
                '混合专家模型为每个输入选择部分专家。张量并行拆分矩阵运算，专家并行分配不同专家。' * 300)
        prompts.append(tokenizer.encode(text, add_special_tokens=False)[:128])
    config = dict(tensor_parallel_size=2, enable_expert_parallel=True, enforce_eager=True,
                  max_model_len=1152, max_num_batched_tokens=1024, max_num_seqs=8,
                  gpu_memory_utilization=0.85, enable_continuous_batching=False,
                  enable_chunked_prefill=False)
    print('Source:', Path(nanovllm.__file__).resolve(), flush=True)
    llm = LLM(MODEL, **config)
    print('Model loaded', flush=True)
    state = dict(step=0, repeat=0)
    captures, diagnostics, all_logits, all_sampled, all_outputs = {}, [], [], [], []
    checkpoints = (0, 29, 55)

    def selected_rows(tensor):
        return tensor[torch.arange(127, 1024, 128, device=tensor.device)] if tensor.shape[0] == 1024 else tensor

    def capture_hook(layer_id, kind):
        def hook(module, inputs, out):
            if state['repeat'] == 0 and state['step'] in checkpoints:
                value = out[0] if isinstance(out, tuple) else out
                captures[f"{state['step']}/{layer_id}/{kind}"] = selected_rows(value).detach().cpu()
        return hook

    def relative_error(value, reference):
        a, b = value.float(), reference.float()
        difference = a - b
        return dict(relative_l2=(difference.norm() / b.norm().clamp_min(1e-12)).item(),
                    max_abs=difference.abs().max().item(),
                    rms=difference.square().mean().sqrt().item(),
                    cosine=F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())

    def diagnostic_wrapper(layer_id, original):
        def forward(block, hidden, weights, selected):
            if state['repeat'] == 0 and state['step'] in checkpoints:
                captures[f"{state['step']}/{layer_id}/selected_experts"] = selected_rows(selected).detach().cpu()
            result = original(hidden, weights, selected)
            if not args.diagnostics or state['repeat'] != 0 or state['step'] not in (29, 55):
                return result
            from nanovllm.layers.moe_grouped import grouped_moe
            from nanovllm.layers.moe_permute import counting_sort_moe_routes
            plan = counting_sort_moe_routes(selected, block.first_expert_id, block.last_expert_id)
            actual = grouped_moe(hidden, weights, plan, block.grouped_gate_up, block.grouped_down)
            ids = plan.sorted_route_ids.long()
            token_ids = ids // selected.shape[1]
            offsets = plan.expert_offsets.tolist()
            permuted = hidden[token_ids]
            legacy_expert = torch.empty_like(permuted)
            fp32_expert = torch.empty_like(permuted, dtype=torch.float32)
            # Match the exact old per-expert module, activation and BF16 multiply/index_add.
            for e, (start, end) in enumerate(zip(offsets, offsets[1:])):
                if start == end:
                    continue
                expert = block.experts[str(e + block.first_expert_id)]
                legacy_expert[start:end] = expert(permuted[start:end])
                gate, up = F.linear(permuted[start:end].float(), expert.gate_up_proj.weight.float()).chunk(2, -1)
                fp32_expert[start:end] = F.linear(F.silu(gate) * up, expert.down_proj.weight.float())
            route_weights = weights.flatten()[ids, None]
            legacy = torch.zeros_like(hidden).index_add_(0, token_ids, legacy_expert * route_weights)
            reference = torch.zeros_like(hidden, dtype=torch.float32).index_add_(
                0, token_ids, fp32_expert * route_weights.float())
            row = dict(step=state['step'], layer=layer_id,
                       grouped_vs_fp32=relative_error(actual, reference),
                       legacy_vs_fp32=relative_error(legacy, reference),
                       grouped_vs_legacy=relative_error(actual, legacy))
            diagnostics.append(row)
            if layer_id in (0, 47):
                print('Same-input local expert check:', row, flush=True)
            return result
        return forward

    handles = []
    for i, layer in enumerate(llm.model_runner.model.model.layers):
        handles.append(layer.register_forward_hook(capture_hook(i, 'hidden')))
        handles.append(layer.mlp.gate.register_forward_hook(capture_hook(i, 'router_logits')))
        old = layer.mlp._forward_expert_parallel
        layer.mlp._forward_expert_parallel = MethodType(diagnostic_wrapper(i, old), layer.mlp)

    class ForcedSampler(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original
            self.logits = []
            self.sampled = []

        def forward(self, logits, temperatures):
            self.logits.append(logits.detach().cpu())
            if args.greedy:
                output = logits.argmax(dim=-1)
                self.sampled.append(output.detach().cpu())
            else:
                self.sampled.append(self.original(logits, temperatures).detach().cpu())
                output = continuation[:, state['step']].to(logits.device)
            state['step'] += 1
            return output

    sampler = ForcedSampler(llm.model_runner.sampler)
    llm.model_runner.sampler = sampler
    try:
        for repeat in range(2):
            state.update(step=0, repeat=repeat)
            bm = llm.scheduler.block_manager
            llm.scheduler.block_manager = BlockManager(len(bm.blocks), bm.block_size)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            sampler.logits.clear()
            sampler.sampled.clear()
            outputs = llm.generate(prompts, SamplingParams(temperature=1e-6, max_tokens=64, ignore_eos=True), use_tqdm=False)
            assert state['step'] == 64
            if not args.greedy:
                assert [o['token_ids'] for o in outputs] == continuation.tolist()
            all_outputs.append(outputs)
            all_logits.append(torch.stack(sampler.logits))
            all_sampled.append(torch.stack(sampler.sampled))
            print('Greedy' if args.greedy else 'Teacher-forced', 'pass complete:', repeat, flush=True)
        torch.save(dict(source=str(Path(nanovllm.__file__).resolve()), config=config,
                        continuation=continuation, logits=all_logits, sampled=all_sampled,
                        captures=captures, diagnostics=diagnostics, outputs=all_outputs, greedy=args.greedy), args.output)
        print('Saved:', args.output, flush=True)
    finally:
        for h in handles:
            h.remove()
        llm.exit()


if __name__ == '__main__':
    main()
