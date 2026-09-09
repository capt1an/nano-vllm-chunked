from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class MoERoutingPlan:
    """Expert-major ordering metadata for the local routes on one EP rank."""

    # [R_local] 或容量 [T*K]（仅 offsets[-1] 之前有效）：排序后的位置 -> 原始 route_id（只包含本 rank 的路由）。
    sorted_route_ids: torch.Tensor
    # [E_local + 1]：专家 e 的路由位于 [offsets[e], offsets[e + 1])。
    expert_offsets: torch.Tensor
    # [T, K]：原始路由 -> 排序后的位置；非本地路由为 -1。
    inverse_route_ids: torch.Tensor


@triton.jit
def _count_expert_routes_kernel(
    expert_ids_ptr,
    expert_counts_ptr,
    num_routes,
    first_expert_id: tl.constexpr,
    num_local_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 一个 Triton program 处理 BLOCK_SIZE 条展平路由；不是一个 program 对应一个专家。
    route_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 最后一个 program 可能不足一块，用 mask 屏蔽越界 lane。
    route_mask = route_ids < num_routes
    expert_ids = tl.load(expert_ids_ptr + route_ids, mask=route_mask, other=-1)
    # 全局专家编号转成本地桶编号，仅保留 [first_expert_id, last_expert_id)。
    local_expert_ids = expert_ids - first_expert_id
    local_mask = route_mask & (local_expert_ids >= 0) & (
        local_expert_ids < num_local_experts
    )
    # 无效 lane 使用安全下标 0，但 local_mask 会禁止它实际读写计数器。
    safe_local_expert_ids = tl.where(local_mask, local_expert_ids, 0)
    # 同一专家可能被多个 lane/program 命中，必须原子加，避免计数丢失。
    tl.atomic_add(
        expert_counts_ptr + safe_local_expert_ids,
        1,
        mask=local_mask,
    )


@triton.jit
def _exclusive_scan_expert_counts_kernel(
    expert_counts_ptr,
    expert_offsets_ptr,
    num_local_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 仅启动一个 program；每个 lane 对应一个专家，补齐的 lane 计数为 0。
    expert_ids = tl.arange(0, BLOCK_SIZE)
    expert_mask = expert_ids < num_local_experts
    counts = tl.load(expert_counts_ptr + expert_ids, mask=expert_mask, other=0)
    # 例：counts=[2,2,1] -> inclusive=[2,4,5] -> exclusive=[0,2,4]。
    # exclusive 就是每个专家桶在排序结果中的起点。
    inclusive_offsets = tl.cumsum(counts, axis=0)
    exclusive_offsets = inclusive_offsets - counts
    tl.store(
        expert_offsets_ptr + expert_ids,
        exclusive_offsets,
        mask=expert_mask,
    )
    # 最后多存一个总数，得到 offsets=[0,2,4,5]，便于统一切片。
    tl.store(expert_offsets_ptr + num_local_experts, tl.sum(counts, axis=0))


@triton.jit
def _scatter_routes_kernel(
    expert_ids_ptr,
    write_offsets_ptr,
    sorted_route_ids_ptr,
    inverse_route_ids_ptr,
    num_routes,
    first_expert_id: tl.constexpr,
    num_local_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 一个 Triton program 处理 BLOCK_SIZE 条展平路由；不是一个 program 对应一个专家。
    route_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 最后一个 program 可能不足一块，用 mask 屏蔽越界 lane。
    route_mask = route_ids < num_routes
    expert_ids = tl.load(expert_ids_ptr + route_ids, mask=route_mask, other=-1)
    # 全局专家编号转成本地桶编号，仅保留 [first_expert_id, last_expert_id)。
    local_expert_ids = expert_ids - first_expert_id
    local_mask = route_mask & (local_expert_ids >= 0) & (
        local_expert_ids < num_local_experts
    )
    # 无效 lane 使用安全下标 0，但 local_mask 会禁止它实际读写计数器。
    safe_local_expert_ids = tl.where(local_mask, local_expert_ids, 0)
    # atomic_add 返回加一前的旧值：为该路由领取桶内唯一的写入位置。
    # 并发领取的顺序不固定，所以同一专家桶内不是稳定排序。
    positions = tl.atomic_add(
        write_offsets_ptr + safe_local_expert_ids,
        1,
        mask=local_mask,
    )
    # 同时建立正向和反向索引；这里移动的是路由编号，还没有搬运 hidden_states。
    tl.store(sorted_route_ids_ptr + positions, route_ids, mask=local_mask)
    tl.store(inverse_route_ids_ptr + route_ids, positions, mask=local_mask)


def counting_sort_moe_routes(
    selected_experts: torch.Tensor,
    first_expert_id: int,
    last_expert_id: int,
    *,
    capacity_buffer: bool = False,
) -> MoERoutingPlan:
    """Bucket top-k routes by their local expert without a comparison sort.

    Route ids index the flattened ``[num_tokens, top_k]`` routing tensors. The
    returned route order is expert-major but intentionally unstable within an
    expert; expert GEMMs do not depend on ordering within a bucket.
    With capacity_buffer=True, allocate T*K entries without a host count read;
    only the prefix ending at expert_offsets[-1] is valid.
    """
    if not selected_experts.is_cuda:
        raise ValueError("selected_experts must be a CUDA tensor")
    if selected_experts.ndim != 2:
        raise ValueError("selected_experts must have shape [num_tokens, top_k]")
    if last_expert_id <= first_expert_id:
        raise ValueError("the local expert range must be non-empty")

    # selected_experts 为 [T, K]；route_id = token_id * K + topk_slot。
    # route_id // K 找回 token，route_id % K 找回其 top-k 槽位。
    selected_experts = selected_experts.contiguous()
    num_routes = selected_experts.numel()
    num_local_experts = last_expert_id - first_expert_id
    device = selected_experts.device

    expert_counts = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
    expert_offsets = torch.empty(
        num_local_experts + 1,
        dtype=torch.int32,
        device=device,
    )
    # scatter 只写本地路由，其他位置保留 -1。
    inverse_route_ids = torch.full(
        (num_routes,),
        -1,
        dtype=torch.int32,
        device=device,
    )

    if num_routes == 0:
        expert_offsets.zero_()
        return MoERoutingPlan(
            sorted_route_ids=torch.empty(0, dtype=torch.int32, device=device),
            expert_offsets=expert_offsets,
            inverse_route_ids=inverse_route_ids.view_as(selected_experts),
        )

    # 第 1 步：遍历 T*K 条路由，统计各本地专家的路由数。
    route_block_size = 256
    route_grid = (triton.cdiv(num_routes, route_block_size),)
    _count_expert_routes_kernel[route_grid](
        selected_experts,
        expert_counts,
        num_routes,
        first_expert_id=first_expert_id,
        num_local_experts=num_local_experts,
        BLOCK_SIZE=route_block_size,
    )

    # 第 2 步：对计数做排他前缀和，得到各专家桶的边界。
    scan_block_size = triton.next_power_of_2(num_local_experts)
    if scan_block_size > 1024:
        raise ValueError("counting-sort scan supports at most 1024 local experts")
    _exclusive_scan_expert_counts_kernel[(1,)](
        expert_counts,
        expert_offsets,
        num_local_experts=num_local_experts,
        BLOCK_SIZE=scan_block_size,
    )

    # .item() 将 GPU 上的本地路由总数读到 CPU，会发生主机同步，
    # 随后按准确大小分配索引数组；若后续支持容量缓冲区，可考虑避免此同步。
    # EP grouped GEMM 使用容量模式，后续 kernel 根据 GPU offsets 屏蔽无效尾部。
    num_local_routes = num_routes if capacity_buffer else int(expert_offsets[-1].item())
    sorted_route_ids = torch.empty(
        num_local_routes,
        dtype=torch.int32,
        device=device,
    )
    # 第 3 步：从各桶起点开始原子分配位置，写入路由索引。
    # 游标会被 scatter 改写，必须 clone，保留原始桶边界供专家计算使用。
    write_offsets = expert_offsets[:-1].clone()
    _scatter_routes_kernel[route_grid](
        selected_experts,
        write_offsets,
        sorted_route_ids,
        inverse_route_ids,
        num_routes,
        first_expert_id=first_expert_id,
        num_local_experts=num_local_experts,
        BLOCK_SIZE=route_block_size,
    )

    return MoERoutingPlan(
        sorted_route_ids=sorted_route_ids,
        expert_offsets=expert_offsets,
        inverse_route_ids=inverse_route_ids.view_as(selected_experts),
    )
