from dataclasses import dataclass

import torch.distributed as dist


@dataclass(frozen=True)
class ParallelGroup:
    process_group: dist.ProcessGroup
    ranks: tuple[int, ...]
    rank: int

    @property
    def world_size(self) -> int:
        return len(self.ranks)


_TP_GROUP: ParallelGroup | None = None
_EP_GROUP: ParallelGroup | None = None
_ENABLE_EP = False


def _generate_parallel_groups(
    world_size: int,
    tensor_parallel_size: int,
    enable_expert_parallel: bool,
) -> tuple[list[list[int]], list[list[int]]]:
    """Create TP groups and optionally reuse those ranks for MoE EP."""
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be positive")
    if world_size % tensor_parallel_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by "
            f"tensor_parallel_size ({tensor_parallel_size})"
        )

    tp_groups = [
        list(range(start, start + tensor_parallel_size))
        for start in range(0, world_size, tensor_parallel_size)
    ]
    ep_groups = tp_groups if enable_expert_parallel else [[rank] for rank in range(world_size)]
    return tp_groups, ep_groups


def initialize_model_parallel(
    tensor_parallel_size: int,
    enable_expert_parallel: bool = False,
) -> None:
    global _TP_GROUP, _EP_GROUP, _ENABLE_EP

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first")
    if _TP_GROUP is not None or _EP_GROUP is not None:
        raise RuntimeError("model parallel groups are already initialized")

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    tp_groups, ep_groups = _generate_parallel_groups(
        world_size,
        tensor_parallel_size,
        enable_expert_parallel,
    )
    _ENABLE_EP = enable_expert_parallel

    for ranks in tp_groups:
        group = dist.group.WORLD if len(ranks) == world_size else dist.new_group(ranks)
        if global_rank in ranks:
            _TP_GROUP = ParallelGroup(group, tuple(ranks), ranks.index(global_rank))

    if enable_expert_parallel:
        # In the current no-DP topology, MoE EP reinterprets the same ranks and
        # communicator used by TP instead of creating another parallel axis.
        _EP_GROUP = _TP_GROUP
    else:
        for ranks in ep_groups:
            group = dist.group.WORLD if len(ranks) == world_size else dist.new_group(ranks)
            if global_rank in ranks:
                _EP_GROUP = ParallelGroup(group, tuple(ranks), ranks.index(global_rank))

    if _TP_GROUP is None or _EP_GROUP is None:
        raise RuntimeError(f"rank {global_rank} was not assigned to a model parallel group")


def destroy_model_parallel() -> None:
    global _TP_GROUP, _EP_GROUP, _ENABLE_EP
    _TP_GROUP = None
    _EP_GROUP = None
    _ENABLE_EP = False


def _require_group(group: ParallelGroup | None, name: str) -> ParallelGroup:
    if group is None:
        raise RuntimeError(f"{name} parallel group is not initialized")
    return group


def get_tensor_parallel_group() -> dist.ProcessGroup:
    return _require_group(_TP_GROUP, "tensor").process_group


def get_tensor_parallel_rank() -> int:
    return _require_group(_TP_GROUP, "tensor").rank


def get_tensor_parallel_world_size() -> int:
    return _require_group(_TP_GROUP, "tensor").world_size


def get_tensor_parallel_src_rank() -> int:
    return _require_group(_TP_GROUP, "tensor").ranks[0]


def get_expert_parallel_group() -> dist.ProcessGroup:
    return _require_group(_EP_GROUP, "expert").process_group


def get_expert_parallel_rank() -> int:
    return _require_group(_EP_GROUP, "expert").rank


def get_expert_parallel_world_size() -> int:
    return _require_group(_EP_GROUP, "expert").world_size


def is_expert_parallel_enabled() -> bool:
    return _ENABLE_EP
