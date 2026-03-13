import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


class DistributedEnv:
    _vae_group = None
    _vae_rank = None
    _vae_group_size = None

    @classmethod
    def initialize(cls, vae_group: ProcessGroup):
        assert vae_group is not None, "vae_group must be provided"
        cls._vae_group = vae_group
        cls._vae_rank = vae_group.rank()
        cls._vae_group_size = vae_group.size()

    @classmethod
    def get_vae_group(cls) -> ProcessGroup:
        return cls._vae_group

    @classmethod
    def get_vae_rank(cls) -> int:
        return cls._vae_rank

    @classmethod
    def get_vae_group_size(cls) -> int:
        return cls._vae_group_size
