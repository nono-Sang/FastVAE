import socket

import torch
import torch.distributed as dist

from fastvae.dist.env import DistributedEnv as dist_env


def get_test_device(rank: int | None = None) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if rank is None:
        return torch.device("cuda")
    return torch.device("cuda", rank)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def init_dist(rank, world_size, init_method) -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    dist_env.initialize(dist.group.WORLD)


def destroy_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
