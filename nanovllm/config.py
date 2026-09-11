import os
from dataclasses import dataclass
from transformers import AutoConfig
from nanovllm.distributed.kv_transfer.config import PDConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enable_expert_parallel: bool = False
    enforce_eager: bool = False
    enable_continuous_batching: bool = True
    enable_chunked_prefill: bool = True
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # None preserves ordinary inference; PD uses independent scheduled engines.
    pd_config: PDConfig | None = None
    distributed_init_method: str = "tcp://localhost:8767"
    worker_shm_name: str = "nanovllm"

    def __post_init__(self):
        if self.pd_config is not None:
            if not isinstance(self.pd_config, PDConfig):
                raise TypeError("pd_config must be a PDConfig")
            if self.tensor_parallel_size != 1 or self.enable_expert_parallel:
                raise ValueError("PD initially requires TP=1 and no EP")
            if self.max_num_seqs <= 0 or self.max_num_batched_tokens <= 0:
                raise ValueError("PD sequence and token budgets must be positive")
        # assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        quant = getattr(self.hf_config, "quantization_config", None)
        if quant is not None:
            from nanovllm.layers.awq import validate_awq
            import torch
            validate_awq(quant)
            if self.hf_config.model_type != "qwen3" or self.tensor_parallel_size != 1 or self.enable_expert_parallel:
                raise ValueError("AWQ currently supports dense Qwen3 with TP=1 and no EP")
            if self.hf_config.dtype != torch.float16:
                raise ValueError("AWQ currently requires float16 model dtype")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

    @property
    def world_size(self) -> int:
        # With no DP/PP yet, EP reuses the TP ranks rather than adding ranks.
        return self.tensor_parallel_size
