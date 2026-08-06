from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    # 替代原来的 is_prefill: bool
    num_prefill_tokens: int = 0
    num_decode_seqs:    int = 0

    # prefill 专用 (flash_attn_varlen_func)
    cu_seqlens_q:          torch.Tensor | None = None
    cu_seqlens_k:          torch.Tensor | None = None
    max_seqlen_q:          int = 0
    max_seqlen_k:          int = 0
    prefill_block_tables:  torch.Tensor | None = None   # 仅 prefix-cache 时非 None

    # decode 专用 (flash_attn_with_kvcache)
    context_lens:          torch.Tensor | None = None
    decode_block_tables:   torch.Tensor | None = None

    # 共享: 覆盖所有 token，prefill slots 在前，decode slots 在后
    slot_mapping:          torch.Tensor | None = None

    # ── 派生属性 ──────────────────────────────────────────────────────────

    @property
    def has_prefill(self) -> bool:
        return self.num_prefill_tokens > 0

    @property
    def has_decode(self) -> bool:
        return self.num_decode_seqs > 0

    @property
    def is_pure_prefill(self) -> bool:
        return self.has_prefill and not self.has_decode

    @property
    def is_pure_decode(self) -> bool:
        return self.has_decode and not self.has_prefill

    @property
    def is_mixed(self) -> bool:
        return self.has_prefill and self.has_decode

    @property
    def split(self) -> int:
        """Attention 层用它切分 q/k/v：q[:split] 给 prefill，q[split:] 给 decode。"""
        return self.num_prefill_tokens

    @property
    def prefill_slot_mapping(self) -> torch.Tensor | None:
        if self.slot_mapping is None or not self.has_prefill:
            return None
        return self.slot_mapping[:self.num_prefill_tokens]

    @property
    def decode_slot_mapping(self) -> torch.Tensor | None:
        if self.slot_mapping is None or not self.has_decode:
            return None
        return self.slot_mapping[self.num_prefill_tokens:]


_CONTEXT: Context = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(ctx: Context) -> None:
    """接受完整的 Context 对象，替代原来的关键字参数版本。"""
    global _CONTEXT
    _CONTEXT = ctx


def reset_context() -> None:
    global _CONTEXT
    _CONTEXT = Context()