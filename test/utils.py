import os
import socket

import torch
import torch.distributed as dist

from fastvae.dist.env import DistributedEnv as dist_env

USE_CUDA = torch.cuda.is_available() and torch.cuda.device_count() >= 8
TEST_DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


def get_test_device(rank: int | None = None) -> torch.device:
    if not USE_CUDA:
        return torch.device("cpu")
    if rank is None:
        return torch.device("cuda")
    return torch.device("cuda", rank)


def split_height(total_height: int, world_size: int, rank: int) -> tuple[int, int]:
    base = total_height // world_size
    remainder = total_height % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def init_dist(
    rank: int | None = None,
    world_size: int | None = None,
    init_method: str | None = None,
) -> tuple[int, int, int]:
    if not dist.is_available():
        return 0, 0, 1

    if rank is None or world_size is None or init_method is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        init_method = (
            f"tcp://127.0.0.1:{find_free_port()}" if world_size <= 1 else "env://"
        )
    else:
        local_rank = rank

    backend = "nccl" if USE_CUDA else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
    if USE_CUDA:
        torch.cuda.set_device(local_rank)
    dist_env.initialize(dist.group.WORLD)
    return rank, local_rank, world_size


def destroy_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
