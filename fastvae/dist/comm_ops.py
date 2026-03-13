import torch
import torch.distributed as dist
import torch.nn.functional as F

from fastvae.dist.env import DistributedEnv as dist_env


def _pad_on_dim(x: torch.Tensor, dim: int, pad_size: int) -> torch.Tensor:
    if pad_size <= 0:
        return x
    # F.pad expects (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
    pad = [0] * (2 * x.dim())
    pad[(x.dim() - 1 - dim) * 2 + 1] = pad_size  # right side of target dim
    return F.pad(x, pad)


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(f"dim {dim} out of range for tensor with {ndim} dims")
    return dim


def _get_shard_range(length: int, world_size: int, rank: int) -> tuple[int, int]:
    base = length // world_size
    remainder = length % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def _slice_on_dim(x: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    slices = [slice(None)] * x.dim()
    slices[dim] = slice(start, end)
    return x[tuple(slices)].contiguous()


def gather_tensor(x: torch.Tensor, dim: int = 3) -> torch.Tensor:
    group = dist_env.get_vae_group()
    world_size = dist_env.get_vae_group_size()

    if world_size <= 1:
        return x

    dim = _normalize_dim(dim, x.dim())

    device = x.device
    size_value = torch.tensor([x.shape[dim]], device=device, dtype=torch.int64)
    size_list = [torch.zeros_like(size_value) for _ in range(world_size)]
    dist.all_gather(size_list, size_value, group=group)
    sizes = [int(s.item()) for s in size_list]
    max_size = max(sizes)

    if x.shape[dim] < max_size:
        x = _pad_on_dim(x, dim, max_size - x.shape[dim])

    shards = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(shards, x.contiguous(), group=group)
    trimmed = [_slice_on_dim(t, dim, 0, sizes[i]) for i, t in enumerate(shards)]
    return torch.cat(trimmed, dim=dim)


def split_tensor(x: torch.Tensor, dim: int = 3) -> torch.Tensor:
    rank = dist_env.get_vae_rank()
    world_size = dist_env.get_vae_group_size()
    if world_size <= 1:
        return x

    dim = _normalize_dim(dim, x.dim())
    start, end = _get_shard_range(x.shape[dim], world_size, rank)
    return _slice_on_dim(x, dim, start, end)


import torch
import torch.distributed as dist
import torch.nn.functional as F

from fastvae.dist.env import DistributedEnv as dist_env


def _pad_on_dim(x: torch.Tensor, dim: int, pad_size: int) -> torch.Tensor:
    if pad_size <= 0:
        return x
    # F.pad expects (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
    pad = [0] * (2 * x.dim())
    pad[(x.dim() - 1 - dim) * 2 + 1] = pad_size  # right side of target dim
    return F.pad(x, pad)


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(f"dim {dim} out of range for tensor with {ndim} dims")
    return dim


def _get_shard_range(length: int, world_size: int, rank: int) -> tuple[int, int]:
    base = length // world_size
    remainder = length % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def _slice_on_dim(x: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    slices = [slice(None)] * x.dim()
    slices[dim] = slice(start, end)
    return x[tuple(slices)].contiguous()


def gather_tensor(x: torch.Tensor, dim: int = 3) -> torch.Tensor:
    group = dist_env.get_vae_group()
    world_size = dist_env.get_vae_group_size()

    if world_size <= 1:
        return x

    dim = _normalize_dim(dim, x.dim())

    device = x.device
    height = torch.tensor([x.shape[dim]], device=device, dtype=torch.int64)
    size_list = [torch.zeros_like(height) for _ in range(world_size)]
    dist.all_gather(size_list, height, group=group)
    sizes = [int(s.item()) for s in size_list]
    max_h = max(sizes)
    if x.shape[dim] < max_h:
        pad_h = max_h - x.shape[dim]
        x = _pad_on_dim(x, dim, pad_h)

    shards = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(shards, x.contiguous(), group=group)
    trimmed = [_slice_on_dim(t, dim, 0, sizes[i]) for i, t in enumerate(shards)]
    return torch.cat(trimmed, dim=dim)


def split_tensor(x: torch.Tensor, dim: int = 3) -> torch.Tensor:
    rank = dist_env.get_vae_rank()
    world_size = dist_env.get_vae_group_size()
    if world_size <= 1:
        return x

    dim = _normalize_dim(dim, x.dim())
    start, end = _get_shard_range(x.shape[dim], world_size, rank)
    return _slice_on_dim(x, dim, start, end)
