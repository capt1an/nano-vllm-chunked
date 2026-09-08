from torch import nn
from transformers import PretrainedConfig

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM


_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
}


def get_model_class(config: PretrainedConfig) -> type[nn.Module]:
    architectures = getattr(config, "architectures", None) or []
    for architecture in architectures:
        model_class = _MODEL_REGISTRY.get(architecture)
        if model_class is not None:
            return model_class

    supported = ", ".join(sorted(_MODEL_REGISTRY))
    requested = ", ".join(architectures) or "<missing>"

    raise ValueError(
        f"Unsupported model architecture: {requested}. "
        f"Supported architectures: {supported}"
    )
