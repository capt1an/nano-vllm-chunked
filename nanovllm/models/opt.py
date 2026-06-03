import torch
from torch import nn
import torch.distributed as dist
from transformers import OPTConfig

from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead


class OPTAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        # max_position: int = 4096 * 32,
        head_dim: int | None = None,
        # rms_norm_eps: float = 1e-06,
        # qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # self.q_size = self.num_heads * self.head_dim
        # self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        # self.qkv_bias = qkv_bias


        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)  # 注意是 out_proj

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        N = hidden_states.shape[0]
        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)
        o = self.attn(q, k, v)
        o = o.view(N, self.num_heads * self.head_dim)
        return self.out_proj(o)


class OPTDecoderLayer(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
    ) -> None:
        super().__init__()
        self.self_attn = OPTAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_attention_heads, # MHA
            head_dim=getattr(config, 'head_dim', None),
        )
        self.activation_fn = nn.ReLU()
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-05, elementwise_affine=True)
        self.fc1 = nn.Linear(in_features=config.hidden_size, out_features=config.ffn_dim, bias=True)
        self.fc2 = nn.Linear(in_features=config.ffn_dim, out_features=config.hidden_size, bias=True)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-05, elementwise_affine=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        # residual: torch.Tensor | None,
    ) -> torch.Tensor:
        # add & layerNorm
        # if residual is None:
        #     hidden_states, residual = self.self_attn_layer_norm(hidden_states), hidden_states
        # else:
        #     hidden_states, residual = self.self_attn_layer_norm(hidden_states, residual)

        # residual
        residual = hidden_states
        # layerNorm
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # attention
        hidden_states = self.self_attn(hidden_states)
        # add Norm
        hidden_states = residual + hidden_states

        # FNN

        # residual
        residual = hidden_states
        # layerNorm
        hidden_states = self.final_layer_norm(hidden_states)
        # fc1
        hidden_states = self.fc1(hidden_states)
        # activation
        hidden_states = self.activation_fn(hidden_states)
        # fc2
        hidden_states = self.fc2(hidden_states)
        # add norm
        hidden_states = residual + hidden_states

        return hidden_states


class OPTLearnedPositionalEmbedding(nn.Embedding):

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
    ):
        self.offset = 2

        super().__init__(
            num_embeddings + self.offset,
            embedding_dim,
        )

    def forward(
        self,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        return nn.functional.embedding(
            position_ids + self.offset,
            self.weight,
        )

class OPTDecoder(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
    ) -> None:
        super().__init__()
        # self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size) # nanovllm 的embedding实现
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=1) 
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-05, elementwise_affine=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        tokens_embeds = self.embed_tokens(input_ids)
        pos_embeds = self.embed_positions(positions)
        hidden_states = tokens_embeds + pos_embeds
        for layer in self.layers:
            hidden_states= layer(hidden_states)
        hidden_states= self.final_layer_norm(hidden_states)
        return hidden_states

class OPTModel(nn.Module):

    def __init__(
        self,
        config: OPTConfig
    ) -> None:
        super().__init__()
        self.decoder = OPTDecoder(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        return self.decoder(
            input_ids,
            positions,
        )

class OPTForCausalLM(nn.Module):

    def __init__(
        self,
        config: OPTConfig
    ) -> None:
        super().__init__()
        if not hasattr(config, "num_key_value_heads"):
            config.num_key_value_heads = config.num_attention_heads
        self.model = OPTModel(config)
        
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.lm_head.weight.data = self.model.decoder.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)