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


def init_dist(rank: int, world_size: int, init_method: str) -> None:
    backend = "nccl" if USE_CUDA else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    dist_env.initialize(dist.group.WORLD)


def destroy_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
