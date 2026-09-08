import unittest
from types import SimpleNamespace

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
from nanovllm.models.registry import get_model_class


class ModelRegistryTest(unittest.TestCase):

    def test_resolves_qwen3(self):
        config = SimpleNamespace(
            architectures=["Qwen3ForCausalLM"],
        )

        self.assertIs(
            get_model_class(config),
            Qwen3ForCausalLM,
        )

    def test_resolves_qwen3_moe(self):
        # Registry support is independent of whether the model's TODO forward
        # path has been implemented yet.
        config = SimpleNamespace(
            architectures=["Qwen3MoeForCausalLM"],
        )

        self.assertIs(
            get_model_class(config),
            Qwen3MoeForCausalLM,
        )

    def test_rejects_unsupported_architecture(self):
        config = SimpleNamespace(
            architectures=["UnknownForCausalLM"],
        )

        with self.assertRaisesRegex(
            ValueError,
            "UnknownForCausalLM",
        ):
            get_model_class(config)

    def test_rejects_missing_architectures(self):
        config = SimpleNamespace()

        with self.assertRaisesRegex(
            ValueError,
            "missing",
        ):
            get_model_class(config)


if __name__ == "__main__":
    unittest.main()
