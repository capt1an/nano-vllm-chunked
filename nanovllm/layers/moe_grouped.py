"""Inference-only grouped expert GEMMs with GPU-resident routing metadata."""

import torch
import triton
import triton.language as tl

from nanovllm.layers.moe_permute import MoERoutingPlan


@triton.jit
def _grouped_linear_kernel(
    X, W, Y, ROUTES, OFFSETS,
    E: tl.constexpr, N: tl.constexpr, K: tl.constexpr, TOP_K: tl.constexpr,
    GATHER: tl.constexpr, EB: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # 将所有专家的 M tiles 连成一个任务空间；桶边界始终留在 GPU。
    experts = tl.arange(0, EB)
    starts = tl.load(OFFSETS + experts, experts < E, 0)
    ends = tl.load(OFFSETS + experts + 1, experts < E, 0)
    tiles = tl.cdiv(ends - starts, BM)
    tile_ends = tl.cumsum(tiles)
    tile_id = tl.program_id(0)
    if tile_id < tl.sum(tiles, 0):
        expert = tl.sum((tile_id >= tile_ends).to(tl.int32), 0)
        start = tl.load(OFFSETS + expert)
        end = tl.load(OFFSETS + expert + 1)
        preceding = tl.sum(tl.where(experts < expert, tiles, 0), 0)
        rows = start + (tile_id - preceding) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        if GATHER:
            # 第一层融合 permutation：排序位置 -> route_id -> 原始 token 行。
            source_rows = tl.load(ROUTES + rows, rows < end, 0) // TOP_K
        else:
            source_rows = rows
        acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(K, BK)):
            ks = block * BK + tl.arange(0, BK)
            a = tl.load(X + source_rows[:, None] * K + ks[None, :],
                        (rows[:, None] < end) & (ks[None, :] < K), 0)
            b = tl.load(W + expert.to(tl.int64) * N * K + cols[None, :] * K + ks[:, None],
                        (cols[None, :] < N) & (ks[:, None] < K), 0)
            acc += tl.dot(a, b, input_precision="ieee")
        tl.store(Y + rows[:, None] * N + cols[None, :], acc,
                 (rows[:, None] < end) & (cols[None, :] < N))


@triton.jit
def _silu_mul_kernel(X, Y, OFFSETS, E: tl.constexpr, I: tl.constexpr, B: tl.constexpr):
    ids = tl.program_id(0) * B + tl.arange(0, B)
    count = tl.load(OFFSETS + E)
    mask = ids < count * I
    row, col = ids // I, ids % I
    gate = tl.load(X + row * (2 * I) + col, mask, 0).to(tl.float32)
    up = tl.load(X + row * (2 * I) + I + col, mask, 0).to(tl.float32)
    tl.store(Y + ids, gate * tl.sigmoid(gate) * up, mask)


@triton.jit
def _combine_kernel(X, INVERSE, WEIGHTS, Y, H: tl.constexpr, TOP_K: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0)
    cols = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.zeros((B,), tl.float32)
    # 每个输出元素由唯一 program 写入，无需清零或 index_add_ 原子累加。
    for slot in range(TOP_K):
        pos = tl.load(INVERSE + token * TOP_K + slot)
        weight = tl.load(WEIGHTS + token * TOP_K + slot).to(tl.float32)
        value = tl.load(X + tl.maximum(pos, 0) * H + cols,
                        (pos >= 0) & (cols < H), 0).to(tl.float32)
        acc += value * weight
    tl.store(Y + token * H + cols, acc, cols < H)


def grouped_moe(hidden_states, routing_weights, plan: MoERoutingPlan, gate_up_weight, down_weight):
    """Compute local contributions; weights have layouts [E, 2I, H], [E, H, I].

    Buffers use route capacity, but kernels only read/write the valid prefix
    described by expert_offsets. No token dropping or host count reads.
    """
    if hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("grouped MoE supports float16, bfloat16 and float32")
    tensors = (hidden_states, routing_weights, gate_up_weight, down_weight)
    if any(t.device != hidden_states.device or t.dtype != hidden_states.dtype for t in tensors):
        raise ValueError("grouped MoE tensors must share device and dtype")
    if not hidden_states.is_cuda or any(not t.is_contiguous() for t in tensors):
        raise ValueError("grouped MoE requires contiguous CUDA tensors")
    tokens, hidden = hidden_states.shape
    experts, twice_intermediate, weight_hidden = gate_up_weight.shape
    intermediate = twice_intermediate // 2
    if twice_intermediate % 2 or weight_hidden != hidden or down_weight.shape != (experts, hidden, intermediate):
        raise ValueError("incompatible expert weight shapes")
    if routing_weights.shape != plan.inverse_route_ids.shape or routing_weights.shape[0] != tokens:
        raise ValueError("incompatible routing shapes")
    if plan.expert_offsets.numel() != experts + 1:
        raise ValueError("incompatible expert offsets")
    top_k = routing_weights.shape[1]
    capacity = plan.sorted_route_ids.numel()
    output = torch.empty_like(hidden_states)
    if tokens == 0:
        return output
    if top_k == 0 or capacity == 0:
        return output.zero_()
    gate_up = hidden_states.new_empty((capacity, 2 * intermediate))
    activated = hidden_states.new_empty((capacity, intermediate))
    expert_output = hidden_states.new_empty((capacity, hidden))
    # sum(ceil(count_e/BM)) <= ceil(capacity/BM) + E - 1。
    # 多余 CTA 在 GPU 上退出，不需要将实际任务数读回主机。
    bm, bn, bk = 16, 64, 32
    m_tiles = triton.cdiv(capacity, bm) + experts - 1
    for x, w, y, n, k, gather in (
        (hidden_states, gate_up_weight, gate_up, 2 * intermediate, hidden, True),
        (activated, down_weight, expert_output, hidden, intermediate, False),
    ):
        _grouped_linear_kernel[(m_tiles, triton.cdiv(n, bn))](
            x, w, y, plan.sorted_route_ids, plan.expert_offsets,
            experts, n, k, top_k, gather, triton.next_power_of_2(experts),
            bm, bn, bk,
        )
        if gather:
            _silu_mul_kernel[(triton.cdiv(capacity * intermediate, 256),)](
                gate_up, activated, plan.expert_offsets, experts, intermediate, 256,
            )
    _combine_kernel[(tokens, triton.cdiv(hidden, 256))](
        expert_output, plan.inverse_route_ids, routing_weights, output, hidden, top_k, 256,
    )
    return output
