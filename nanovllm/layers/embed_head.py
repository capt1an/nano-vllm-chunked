import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.distributed import (
    get_tensor_parallel_group,
    get_tensor_parallel_rank,
    get_tensor_parallel_src_rank,
    get_tensor_parallel_world_size,
)
from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = get_tensor_parallel_rank()
        self.tp_size = get_tensor_parallel_world_size()
        self.tp_group = get_tensor_parallel_group()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y, group=self.tp_group)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()

        if context.has_prefill:
            prefill_last = context.cu_seqlens_q[1:] - 1

            if context.has_decode:
                decode_indices = torch.arange(
                    context.num_prefill_tokens,
                    context.num_prefill_tokens + context.num_decode_seqs,
                    device=x.device,
                    dtype=prefill_last.dtype,
                )

                sample_indices = torch.cat(
                    [prefill_last, decode_indices]
                )
            else:
                sample_indices = prefill_last

            x = x[sample_indices].contiguous()
            
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(
                logits,
                all_logits,
                dst=get_tensor_parallel_src_rank(),
                group=self.tp_group,
            )
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
