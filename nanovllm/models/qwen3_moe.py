"""Correctness-first Qwen3-MoE implementation for tensor/expert parallelism.

The target checkpoint is Qwen/Qwen3-30B-A3B-Instruct-2507.  Its relevant
configuration is:

* hidden_size=2048
* num_hidden_layers=48
* num_experts=128
* num_experts_per_tok=8
* moe_intermediate_size=768
* norm_topk_prob=True

With EP disabled, each expert uses tensor-parallel linear layers. With EP
enabled, experts are sharded between EP ranks and each local expert is complete.
"""

import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3MoeConfig

from nanovllm.distributed import (
    get_expert_parallel_group,
    get_expert_parallel_rank,
    get_expert_parallel_world_size,
    is_expert_parallel_enabled,
)
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import MergedReplicatedLinear, ReplicatedLinear
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.moe_permute import counting_sort_moe_routes
from nanovllm.layers.moe_grouped import grouped_moe
from nanovllm.models.qwen3 import Qwen3Attention, Qwen3MLP


def get_local_expert_range(
    num_experts: int,
    ep_rank: int,
    ep_size: int,
) -> tuple[int, int]:
    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by EP size ({ep_size})"
        )
    experts_per_rank = num_experts // ep_size
    start = ep_rank * experts_per_rank
    return start, start + experts_per_rank


class Qwen3MoeExpertMLP(nn.Module):
    """A complete local expert with no tensor-parallel collective."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_up_proj = MergedReplicatedLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = ReplicatedLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        return self.down_proj(self.act_fn(gate_up))


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Readable correctness-first MoE block.

    The router is replicated. EP ranks own disjoint complete experts and sum
    their partial token outputs once at the end of the block.
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.use_ep = is_expert_parallel_enabled()
        self.ep_rank = get_expert_parallel_rank()
        self.ep_size = get_expert_parallel_world_size()
        self.ep_group = get_expert_parallel_group()
        self.first_expert_id, self.last_expert_id = get_local_expert_range(
            self.num_experts,
            self.ep_rank,
            self.ep_size,
        )

        # The checkpoint key is ``model.layers.N.mlp.gate.weight`` with shape
        # [num_experts, hidden_size].  ReplicatedLinear preserves that name and
        # keeps routing decisions identical across TP ranks.
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
        )

        expert_cls = Qwen3MoeExpertMLP if self.use_ep else Qwen3MLP
        # Global ids are kept as ModuleDict keys so local parameter names still
        # match checkpoint paths such as ``mlp.experts.64.gate_proj.weight``.
        self.experts = nn.ModuleDict({
            str(expert_id): expert_cls(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act,
            )
            for expert_id in range(self.first_expert_id, self.last_expert_id)
        })
        if self.use_ep:
            # 仅初始化时打包；forward 不复制权重。buffer 不增加 state_dict 键。
            self.register_buffer("grouped_gate_up", torch.stack([
                expert.gate_up_proj.weight.detach() for expert in self.experts.values()
            ]), persistent=False)
            self.register_buffer("grouped_down", torch.stack([
                expert.down_proj.weight.detach() for expert in self.experts.values()
            ]), persistent=False)
            self._bind_grouped_weights()

    def _bind_grouped_weights(self):
        # 保留原 checkpoint 参数名和 weight_loader，但参数直接引用连续专家缓冲区。
        for local_id, expert in enumerate(self.experts.values()):
            expert.gate_up_proj.weight.data = self.grouped_gate_up[local_id]
            expert.down_proj.weight.data = self.grouped_down[local_id]

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if self.use_ep:
            # .to() / dtype 转换后重新建立参数与缓冲区的共享存储。
            self._bind_grouped_weights()
        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Route every token to top-k experts and combine their outputs.

        Args:
            hidden_states: nano-vLLM flattens all scheduled tokens, so the
                expected shape is [num_tokens, hidden_size], not the 3-D shape
                used by the Hugging Face training implementation.

        Returns:
            Tensor with the same shape and dtype as ``hidden_states``.
        """
        router_logits = self.gate(hidden_states)

        router_logits = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(router_logits, k=self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        if self.use_ep:
            return self._forward_expert_parallel(
                hidden_states,
                routing_weights,
                selected_experts,
            )
        return self._forward_tensor_parallel(
            hidden_states,
            routing_weights,
            selected_experts,
        )

    def _forward_tensor_parallel(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Preserve the original TP=2, EP=1 reference implementation."""
        T, K = selected_experts.shape
        tokenids = torch.arange(T, device=hidden_states.device).repeat_interleave(K)
        expertids = selected_experts.reshape(-1)
        permuteidx = torch.argsort(expertids)
        sorted_expert_ids = expertids[permuteidx]
        sorted_token_ids = tokenids[permuteidx]
        permuted_hidden_states = hidden_states[sorted_token_ids]
        token_per_expert = torch.bincount(sorted_expert_ids, minlength=self.num_experts)

        output = torch.zeros_like(permuted_hidden_states)
        startidx = 0
        for expert_id in range(self.num_experts):
            count = token_per_expert[expert_id].item()
            endidx = startidx + count
            if count != 0:
                output[startidx:endidx] = self.experts[str(expert_id)](
                    permuted_hidden_states[startidx:endidx]
                )
            startidx = endidx

        unpermuted_output = torch.empty_like(output)
        unpermuted_output[permuteidx] = output
        unpermuted_output = unpermuted_output.view(T, K, hidden_states.shape[-1])

        return (unpermuted_output * routing_weights.unsqueeze(-1)).sum(dim=1)

    def _forward_expert_parallel(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute local expert contributions and combine them over EP ranks."""
        # 容量模式无需 .item()；有效路由数和专家桶边界始终保留在 GPU。
        routing_plan = counting_sort_moe_routes(
            selected_experts,
            self.first_expert_id,
            self.last_expert_id,
            capacity_buffer=True,
        )
        # 第一层 GEMM 融合输入 permutation；最后按 inverse 索引加权归并。
        local_output = grouped_moe(
            hidden_states, routing_weights, routing_plan,
            self.grouped_gate_up, self.grouped_down,
        )

        # 各 rank 持有同一批 token，分别计算各自专家的贡献，最后跨 rank 求和。
        # 即使本 rank 没有本地路由，也必须参与 all_reduce。
        dist.all_reduce(local_output, group=self.ep_group)
        return local_output


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 10_000_000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        is_sparse = (
            layer_idx not in config.mlp_only_layers
            and config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        )
        if is_sparse:
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            # The selected 2507 checkpoint never takes this branch, but keeping
            # it makes the layer definition agree with Qwen3MoeConfig semantics.
            self.mlp = Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The residual/RMSNorm order is identical to dense Qwen3; only the MLP
        # implementation differs.  Keeping this path aligned makes later dense
        # versus MoE debugging much easier.
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    # These replacements are substring-based in the current loader and also
    # match expert paths such as ``mlp.experts.17.gate_proj.weight``.
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        ep_rank = get_expert_parallel_rank()
        ep_size = get_expert_parallel_world_size()
        self.first_expert_id, self.last_expert_id = get_local_expert_range(
            config.num_experts,
            ep_rank,
            ep_size,
        )
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def is_weight_local(self, weight_name: str) -> bool:
        """Return whether an expert checkpoint tensor belongs to this EP rank."""
        marker = ".experts."
        if marker not in weight_name:
            return True
        expert_id = int(weight_name.split(marker, 1)[1].split(".", 1)[0])
        return self.first_expert_id <= expert_id < self.last_expert_id
