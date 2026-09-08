import unittest

import torch

from nanovllm.layers.moe_permute import counting_sort_moe_routes


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton kernels")
class MoEPermuteTest(unittest.TestCase):

    def test_counting_sort_filters_and_buckets_local_routes(self):
        selected_experts = torch.tensor(
            [[0, 2], [3, 5], [4, 2], [1, 3]],
            dtype=torch.int64,
            device="cuda",
        )

        plan = counting_sort_moe_routes(selected_experts, 2, 5)

        self.assertEqual(plan.expert_offsets.cpu().tolist(), [0, 2, 4, 5])
        sorted_routes = plan.sorted_route_ids.long()
        flat_experts = selected_experts.flatten()
        self.assertEqual(
            flat_experts[sorted_routes].cpu().tolist(),
            [2, 2, 3, 3, 4],
        )
        inverse = plan.inverse_route_ids.flatten()
        torch.testing.assert_close(
            inverse[sorted_routes],
            torch.arange(5, dtype=torch.int32, device="cuda"),
        )
        remote_route_ids = torch.tensor([0, 3, 6], device="cuda")
        torch.testing.assert_close(
            inverse[remote_route_ids],
            torch.full((3,), -1, dtype=torch.int32, device="cuda"),
        )

    def test_counting_sort_handles_no_local_routes(self):
        selected_experts = torch.tensor(
            [[0, 1], [0, 1]],
            dtype=torch.int64,
            device="cuda",
        )

        plan = counting_sort_moe_routes(selected_experts, 2, 4)

        self.assertEqual(plan.sorted_route_ids.numel(), 0)
        self.assertEqual(plan.expert_offsets.cpu().tolist(), [0, 0, 0])
        self.assertTrue(torch.all(plan.inverse_route_ids == -1))

    def test_counting_sort_matches_random_histogram(self):
        generator = torch.Generator(device="cuda").manual_seed(7)
        selected_experts = torch.randint(
            0,
            128,
            (1024, 8),
            generator=generator,
            device="cuda",
        )

        plan = counting_sort_moe_routes(selected_experts, 64, 128)

        flat_experts = selected_experts.flatten()
        local_mask = (flat_experts >= 64) & (flat_experts < 128)
        expected_counts = torch.bincount(
            flat_experts[local_mask] - 64,
            minlength=64,
        ).to(torch.int32)
        actual_counts = plan.expert_offsets[1:] - plan.expert_offsets[:-1]
        torch.testing.assert_close(actual_counts, expected_counts)

        sorted_routes = plan.sorted_route_ids.long()
        expected_routes = torch.nonzero(local_mask, as_tuple=False).flatten()
        torch.testing.assert_close(
            torch.sort(sorted_routes).values,
            expected_routes,
        )
        torch.testing.assert_close(
            plan.inverse_route_ids.flatten()[sorted_routes],
            torch.arange(sorted_routes.numel(), dtype=torch.int32, device="cuda"),
        )


if __name__ == "__main__":
    unittest.main()
