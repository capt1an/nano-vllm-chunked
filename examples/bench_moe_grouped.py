"""Local EP compute benchmark (excludes all-reduce and model loading).

Run: .venv/bin/python -m examples.bench_moe_grouped --tokens 32 128 512
"""
import argparse

import torch
import torch.nn.functional as F
import triton

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.moe_grouped import grouped_moe
from nanovllm.layers.moe_permute import counting_sort_moe_routes


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', nargs='+', type=int, default=[32, 128, 512])
    parser.add_argument('--experts', type=int, default=128)
    parser.add_argument('--ep-size', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=2048)
    parser.add_argument('--intermediate', type=int, default=768)
    args = parser.parse_args()
    assert args.experts % args.ep_size == 0
    torch.manual_seed(42)
    local = args.experts // args.ep_size
    w1 = torch.randn(local, args.intermediate * 2, args.hidden, device='cuda', dtype=torch.bfloat16) * 0.01
    w2 = torch.randn(local, args.hidden, args.intermediate, device='cuda', dtype=torch.bfloat16) * 0.01
    act = SiluAndMul()
    print('tokens  loop_ms  grouped_ms  speedup')
    for tokens in args.tokens:
        x = torch.randn(tokens, args.hidden, device='cuda', dtype=torch.bfloat16)
        selected = torch.rand(tokens, args.experts, device='cuda').topk(8, dim=-1).indices
        weights = torch.softmax(torch.randn(tokens, 8, device='cuda'), -1).to(x.dtype)

        def old():
            plan = counting_sort_moe_routes(selected, 0, local)
            routes = plan.sorted_route_ids
            token_ids = routes // 8
            route_weights = weights.flatten()[routes]
            permuted = x[token_ids]
            expert_out = torch.empty_like(permuted)
            offsets = plan.expert_offsets.tolist()
            for e in range(local):
                start, end = offsets[e:e + 2]
                if start != end:
                    expert_out[start:end] = F.linear(act(F.linear(permuted[start:end], w1[e])), w2[e])
            return torch.zeros_like(x).index_add_(0, token_ids, expert_out * route_weights[:, None])

        def new():
            plan = counting_sort_moe_routes(selected, 0, local, capacity_buffer=True)
            return grouped_moe(x, weights, plan, w1, w2)

        expected, actual = old(), new()
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.05)
        baseline = triton.testing.do_bench(old, warmup=100, rep=300)
        grouped = triton.testing.do_bench(new, warmup=100, rep=300)
        print(f'{tokens:6d} {baseline:8.3f} {grouped:11.3f} {baseline / grouped:8.2f}x', flush=True)


if __name__ == '__main__':
    main()
