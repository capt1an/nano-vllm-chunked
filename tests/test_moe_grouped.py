import tempfile
from datetime import timedelta
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import Qwen3MoeConfig

from nanovllm.layers.moe_grouped import grouped_moe
from nanovllm.layers.moe_permute import counting_sort_moe_routes
from nanovllm.models.qwen3_moe import Qwen3MoeSparseMoeBlock


def reference(x, weights, selected, first, w1, w2):
    result = torch.zeros_like(x, dtype=torch.float32)
    for e in range(w1.shape[0]):
        token, slot = torch.where(selected == e + first)
        gate, up = F.linear(x[token], w1[e]).chunk(2, dim=-1)
        value = F.linear((F.silu(gate.float()) * up.float()).to(x.dtype), w2[e])
        result.index_add_(0, token, value.float() * weights[token, slot, None].float())
    return result.to(x.dtype)


def ep_worker(rank, rendezvous):
    from nanovllm.distributed import initialize_model_parallel, destroy_model_parallel
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    initialize_model_parallel(2, True)
    try:
        torch.manual_seed(42)
        cfg = Qwen3MoeConfig(hidden_size=64, moe_intermediate_size=32,
                            num_experts=6, num_experts_per_tok=2, hidden_act='silu')
        block = Qwen3MoeSparseMoeBlock(cfg).to(device=rank, dtype=torch.float16)
        w1 = torch.randn(6, 64, 64, device=rank, dtype=torch.float16) * 0.1
        w2 = torch.randn(6, 64, 32, device=rank, dtype=torch.float16) * 0.1
        for local, expert in enumerate(block.experts.values()):
            e = block.first_expert_id + local
            expert.gate_up_proj.weight.weight_loader(expert.gate_up_proj.weight, w1[e, :32], 0)
            expert.gate_up_proj.weight.weight_loader(expert.gate_up_proj.weight, w1[e, 32:], 1)
            expert.down_proj.weight.weight_loader(expert.down_proj.weight, w2[e])
        x = torch.randn(19, 64, device=rank, dtype=torch.float16) * 0.2
        weights = torch.full((19, 2), 0.5, device=rank, dtype=torch.float16)
        for mode in ('mixed', 'empty_rank'):
            selected = torch.randint(0, 6, (19, 2), device=rank)
            if mode == 'empty_rank':
                selected.zero_()
            actual = block._forward_expert_parallel(x, weights, selected)
            torch.testing.assert_close(actual, reference(x, weights, selected, 0, w1, w2),
                                       atol=1e-3, rtol=0.03)
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.device_count() >= 2, 'Two CUDA devices required')
class DistributedGroupedMoETest(unittest.TestCase):
    def test_two_rank_ep(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ep_worker, args=('file://' + directory + '/rendezvous',), nprocs=2)


class PackedWeightsTest(unittest.TestCase):
    def test_checkpoint_load_and_dtype_conversion(self):
        with patch('nanovllm.models.qwen3_moe.is_expert_parallel_enabled', return_value=True), \
             patch('nanovllm.models.qwen3_moe.get_expert_parallel_rank', return_value=0), \
             patch('nanovllm.models.qwen3_moe.get_expert_parallel_world_size', return_value=1), \
             patch('nanovllm.models.qwen3_moe.get_expert_parallel_group', return_value=None), \
             patch('nanovllm.layers.linear.get_tensor_parallel_rank', return_value=0), \
             patch('nanovllm.layers.linear.get_tensor_parallel_world_size', return_value=1), \
             patch('nanovllm.layers.linear.get_tensor_parallel_group', return_value=None):
            block = Qwen3MoeSparseMoeBlock(Qwen3MoeConfig(
                hidden_size=32, moe_intermediate_size=16, num_experts=3,
                num_experts_per_tok=2, hidden_act='silu',
            ))
        for e, expert in enumerate(block.experts.values()):
            param = expert.gate_up_proj.weight
            param.weight_loader(param, torch.full((16, 32), float(e + 1)), 0)
            param.weight_loader(param, torch.full((16, 32), float(e + 4)), 1)
            expert.down_proj.weight.weight_loader(expert.down_proj.weight, torch.full((32, 16), float(e)))
        state = {k: v.clone() for k, v in block.state_dict().items()}
        block.to(dtype=torch.float64)
        self.assertNotIn('grouped_gate_up', block.state_dict())
        for e, expert in enumerate(block.experts.values()):
            self.assertEqual(expert.gate_up_proj.weight.data_ptr(), block.grouped_gate_up[e].data_ptr())
            self.assertEqual(expert.down_proj.weight.data_ptr(), block.grouped_down[e].data_ptr())
            torch.testing.assert_close(block.grouped_gate_up[e], state[f'experts.{e}.gate_up_proj.weight'].double())
        block.load_state_dict(state)
        torch.testing.assert_close(block.grouped_down[2], state['experts.2.down_proj.weight'].double())


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class GroupedMoETest(unittest.TestCase):
    def run_case(self, dtype, tokens, hidden, intermediate, mode):
        torch.manual_seed(13)
        experts, top_k, first = 5, 3, 2
        x = torch.randn(tokens, hidden, device='cuda', dtype=dtype) * 0.2
        w1 = torch.randn(experts, 2 * intermediate, hidden, device='cuda', dtype=dtype) * 0.1
        w2 = torch.randn(experts, hidden, intermediate, device='cuda', dtype=dtype) * 0.1
        selected = torch.randint(0, 9, (tokens, top_k), device='cuda')
        if mode == 'remote':
            selected.fill_(0)
        elif mode == 'skew':
            selected.fill_(first + 2)
        weights = torch.softmax(torch.randn(tokens, top_k, device='cuda'), -1).to(dtype)
        plan = counting_sort_moe_routes(selected, first, first + experts, capacity_buffer=True)
        self.assertEqual(plan.sorted_route_ids.numel(), tokens * top_k)
        actual = grouped_moe(x, weights, plan, w1, w2)
        expected = reference(x, weights, selected, first, w1, w2)
        torch.testing.assert_close(actual, expected, atol={torch.float16: 1e-3, torch.bfloat16: 1e-2, torch.float32: 1e-5}[dtype],
                                   rtol=0.03 if dtype != torch.float32 else 1e-4)
        return x, weights, selected, w1, w2

    def test_ragged_empty_and_skewed(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for tokens, hidden, intermediate, mode in (
                (37, 70, 45, 'random'), (1, 64, 32, 'random'),
                (23, 64, 32, 'remote'), (73, 64, 32, 'skew'),
                (0, 64, 32, 'random'), (128, 2048, 768, 'random'),
            ):
                with self.subTest(dtype=dtype, tokens=tokens, mode=mode):
                    self.run_case(dtype, tokens, hidden, intermediate, mode)

    def test_graph_replay_with_changed_routes(self):
        x, weights, selected, w1, w2 = self.run_case(torch.float16, 37, 70, 45, 'random')
        def run():
            plan = counting_sort_moe_routes(selected, 2, 7, capacity_buffer=True)
            return grouped_moe(x, weights, plan, w1, w2)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        for expert in (4, 0, 6):
            selected.fill_(expert)
            graph.replay()
            torch.testing.assert_close(out, reference(x, weights, selected, 2, w1, w2), atol=0.0003, rtol=0.03)


if __name__ == '__main__':
    unittest.main()
