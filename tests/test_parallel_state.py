import unittest
from unittest.mock import patch

from nanovllm.config import Config
from nanovllm.distributed.parallel_state import _generate_parallel_groups
from nanovllm.models.qwen3_moe import get_local_expert_range


class ParallelStateTest(unittest.TestCase):

    def test_tp_only_groups(self):
        tp_groups, ep_groups = _generate_parallel_groups(2, 2, False)

        self.assertEqual(tp_groups, [[0, 1]])
        self.assertEqual(ep_groups, [[0], [1]])

    def test_tp_and_ep_groups_can_overlap(self):
        tp_groups, ep_groups = _generate_parallel_groups(2, 2, True)

        self.assertEqual(tp_groups, [[0, 1]])
        self.assertEqual(ep_groups, [[0, 1]])

    def test_each_tp_replica_gets_an_ep_group(self):
        tp_groups, ep_groups = _generate_parallel_groups(4, 2, True)

        self.assertEqual(tp_groups, [[0, 1], [2, 3]])
        self.assertEqual(ep_groups, [[0, 1], [2, 3]])

    def test_rejects_inconsistent_world_size(self):
        with self.assertRaisesRegex(ValueError, "world_size"):
            _generate_parallel_groups(3, 2, False)

    def test_local_experts_are_contiguously_sharded(self):
        self.assertEqual(get_local_expert_range(128, 0, 2), (0, 64))
        self.assertEqual(get_local_expert_range(128, 1, 2), (64, 128))

    def test_rejects_uneven_expert_sharding(self):
        with self.assertRaisesRegex(ValueError, "num_experts"):
            get_local_expert_range(3, 0, 2)

    @patch("nanovllm.config.AutoConfig.from_pretrained")
    def test_enabling_ep_does_not_increase_world_size(self, from_pretrained):
        from_pretrained.return_value.max_position_embeddings = 4096

        config = Config(
            "unused",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
        )

        self.assertEqual(config.world_size, 2)


if __name__ == "__main__":
    unittest.main()
