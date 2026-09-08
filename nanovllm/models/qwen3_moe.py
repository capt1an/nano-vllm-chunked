"""Qwen3-MoE model scaffold.

This first implementation intentionally uses a readable ModuleList of experts.
It is meant to establish numerical correctness with tensor parallelism before
adding a fused MoE kernel or expert parallelism.

The target checkpoint is Qwen/Qwen3-30B-A3B-Instruct-2507.  Its relevant
configuration is:

* hidden_size=2048
* num_hidden_layers=48
* num_experts=128
* num_experts_per_tok=8
* moe_intermediate_size=768
* norm_topk_prob=True

Do not add EP-specific sharding to this file yet.  With tensor_parallel_size=2,
the existing parallel linear layers shard every expert's MLP weights, which is
enough for the BF16 checkpoint to fit on the two RTX A6000 GPUs.
"""

import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen3MoeConfig

from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.models.qwen3 import Qwen3Attention, Qwen3MLP


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Readable correctness-first MoE block.

    Each expert uses nano-vLLM's existing tensor-parallel Qwen3MLP.  The router
    is replicated on every TP rank, so all ranks select the same experts and
    execute RowParallelLinear collectives in the same expert order.

    This is deliberately not a fast implementation: Python dispatch and 128
    separate expert modules are useful for learning and correctness checking.
    A fused/grouped implementation belongs in a later optimization commit.
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # The checkpoint key is ``model.layers.N.mlp.gate.weight`` with shape
        # [num_experts, hidden_size].  ReplicatedLinear preserves that name and
        # keeps routing decisions identical across TP ranks.
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
        )

        # Qwen3MLP names its packed first projection ``gate_up_proj``.  The
        # packed_modules_mapping on Qwen3MoeForCausalLM below maps each expert's
        # separate gate_proj/up_proj checkpoint tensors into that parameter.
        self.experts = nn.ModuleList([
            Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act,
            )
            for _ in range(config.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Route every token to top-k experts and combine their outputs.

        Args:
            hidden_states: nano-vLLM flattens all scheduled tokens, so the
                expected shape is [num_tokens, hidden_size], not the 3-D shape
                used by the Hugging Face training implementation.

        Returns:
            Tensor with the same shape and dtype as ``hidden_states``.
        """
        # TODO(you): implement the correctness-first routing path in five steps.
        #
        # 1. Validate/understand the input shape, then compute router logits:
        #       [T, H] -> self.gate -> [T, E]
        router_logits = self.gate(hidden_states)

        # 2. Compute softmax in float32 along the expert dimension.  Select
        #    self.top_k values and expert ids for every token.  Expected shapes:
        #       routing_weights: [T, K]
        #       selected_experts: [T, K]
        #    Convert the selected weights back to hidden_states.dtype only after
        #    softmax/top-k.  Compare this detail with Transformers' reference.
        router_logits = torch.softmax(router_logits, dim=-1, dtype=torch.float32) # fp32
        routing_weights, selected_experts = torch.topk(router_logits, k=self.top_k, dim=-1)

        # 3. Because this checkpoint has norm_topk_prob=True, divide each row of
        #    the selected weights by its row sum.  Keep the code conditional so
        #    the block remains correct for other Qwen3-MoE checkpoints.
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # 4. Allocate an output with torch.zeros_like(hidden_states).  Iterate
        #    expert ids in ascending order.  For each expert, locate every
        #    (token_index, topk_slot) pair assigned to it, gather those tokens,
        #    call self.experts[expert_id], multiply by the corresponding routing
        #    weights, and accumulate with index_add_.  A token appears K times,
        #    so ordinary indexed assignment would be incorrect.
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
        for expert_id, expert in enumerate(self.experts):
            count = token_per_expert[expert_id].item()
            endidx = startidx + count
            if count != 0:
                output[startidx:endidx] = expert(permuted_hidden_states[startidx:endidx])
            startidx = endidx

        unpermuted_output = torch.empty_like(output)
        unpermuted_output[permuteidx] = output
        unpermuted_output = unpermuted_output.view(T, K, hidden_states.shape[-1])

        # 5. Return the accumulated output and assert its shape matches the
        #    input while debugging.
        #
        # TP warning: do not skip experts differently on different TP ranks.
        # RowParallelLinear performs an all_reduce, so every rank must call the
        # same non-empty experts in the same order or the process can deadlock.
        output = (unpermuted_output * routing_weights.unsqueeze(-1)).sum(dim=1)

        return output


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
