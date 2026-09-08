from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class MoERoutingPlan:
    """Expert-major ordering metadata for the local routes on one EP rank."""

    sorted_route_ids: torch.Tensor
    expert_offsets: torch.Tensor
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
    route_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    route_mask = route_ids < num_routes
    expert_ids = tl.load(expert_ids_ptr + route_ids, mask=route_mask, other=-1)
    local_expert_ids = expert_ids - first_expert_id
    local_mask = route_mask & (local_expert_ids >= 0) & (
        local_expert_ids < num_local_experts
    )
    safe_local_expert_ids = tl.where(local_mask, local_expert_ids, 0)
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
    expert_ids = tl.arange(0, BLOCK_SIZE)
    expert_mask = expert_ids < num_local_experts
    counts = tl.load(expert_counts_ptr + expert_ids, mask=expert_mask, other=0)
    inclusive_offsets = tl.cumsum(counts, axis=0)
    exclusive_offsets = inclusive_offsets - counts
    tl.store(
        expert_offsets_ptr + expert_ids,
        exclusive_offsets,
        mask=expert_mask,
    )
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
    route_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    route_mask = route_ids < num_routes
    expert_ids = tl.load(expert_ids_ptr + route_ids, mask=route_mask, other=-1)
    local_expert_ids = expert_ids - first_expert_id
    local_mask = route_mask & (local_expert_ids >= 0) & (
        local_expert_ids < num_local_experts
    )
    safe_local_expert_ids = tl.where(local_mask, local_expert_ids, 0)
    positions = tl.atomic_add(
        write_offsets_ptr + safe_local_expert_ids,
        1,
        mask=local_mask,
    )
    tl.store(sorted_route_ids_ptr + positions, route_ids, mask=local_mask)
    tl.store(inverse_route_ids_ptr + route_ids, positions, mask=local_mask)


def counting_sort_moe_routes(
    selected_experts: torch.Tensor,
    first_expert_id: int,
    last_expert_id: int,
) -> MoERoutingPlan:
    """Bucket top-k routes by their local expert without a comparison sort.

    Route ids index the flattened ``[num_tokens, top_k]`` routing tensors. The
    returned route order is expert-major but intentionally unstable within an
    expert; expert GEMMs do not depend on ordering within a bucket.
    """
    if not selected_experts.is_cuda:
        raise ValueError("selected_experts must be a CUDA tensor")
    if selected_experts.ndim != 2:
        raise ValueError("selected_experts must have shape [num_tokens, top_k]")
    if last_expert_id <= first_expert_id:
        raise ValueError("the local expert range must be non-empty")

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

    scan_block_size = triton.next_power_of_2(num_local_experts)
    if scan_block_size > 1024:
        raise ValueError("counting-sort scan supports at most 1024 local experts")
    _exclusive_scan_expert_counts_kernel[(1,)](
        expert_counts,
        expert_offsets,
        num_local_experts=num_local_experts,
        BLOCK_SIZE=scan_block_size,
    )

    # A single host synchronization gives the exact local buffer size. Once a
    # grouped GEMM accepts capacity-sized buffers, this can remain on device.
    num_local_routes = int(expert_offsets[-1].item())
    sorted_route_ids = torch.empty(
        num_local_routes,
        dtype=torch.int32,
        device=device,
    )
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
