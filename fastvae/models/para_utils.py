# Adapted from https://github.com/RiseAI-Sys/ParaVAE/blob/main/paravae/models/WAN2_1/patch_vae.py

from typing import List, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_2_t, _size_3_t, _size_4_t

from fastvae.dist.env import DistributedEnv as dist_env


def _calc_patch_height_index(patch_height_list: List[Tensor]) -> Tensor:
    patch_heights = torch.stack(patch_height_list).view(-1)
    return torch.cat([patch_heights.new_zeros(1), torch.cumsum(patch_heights, dim=0)])


def _calc_top_halo_size(
    local_rank, world_size, patch_height_index, kernel_size, padding, stride
):
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    if local_rank == 0:
        return 0
    nstep_before_top = (
        patch_height_index[local_rank] + padding - (kernel_size - 1) // 2 + stride - 1
    ) // stride
    top_halo_size = patch_height_index[local_rank] - (
        nstep_before_top * stride - padding
    )
    return top_halo_size


def _calc_bottom_halo_size(
    local_rank, world_size, patch_height_index, kernel_size, padding, stride
):
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    if local_rank == world_size - 1:
        return 0
    nstep_before_bottom = (
        patch_height_index[local_rank + 1]
        + padding
        - (kernel_size - 1) // 2
        + stride
        - 1
    ) // stride
    bottom_halo_size = (
        (nstep_before_bottom - 1) * stride
        + kernel_size
        - padding
        - patch_height_index[local_rank + 1]
    )
    return bottom_halo_size


def halo_exchange(
    x: Tensor,
    *,
    rank: int,
    group,
    prev_bottom_halo_size: int,
    next_top_halo_size: int,
    curr_top_halo_size: int,
    curr_bottom_halo_size: int,
) -> Tensor:
    b, c, t, _, w = x.shape
    device = x.device

    comm_ops = []
    top_halo_recv = None
    bottom_halo_recv = None

    # send halo region to prev rank
    if prev_bottom_halo_size > 0:
        top_halo_send = x[:, :, :, :prev_bottom_halo_size, :].contiguous()
        comm_ops.append(dist.P2POp(dist.isend, top_halo_send, rank - 1, group=group))

    # send halo region to next rank
    if next_top_halo_size > 0:
        bottom_halo_send = x[:, :, :, -next_top_halo_size:, :].contiguous()
        comm_ops.append(dist.P2POp(dist.isend, bottom_halo_send, rank + 1, group=group))

    # recv halo region from prev rank
    if curr_top_halo_size > 0:
        top_halo_recv = torch.empty(
            [b, c, t, curr_top_halo_size, w], dtype=x.dtype, device=device
        )
        comm_ops.append(dist.P2POp(dist.irecv, top_halo_recv, rank - 1, group=group))
    elif curr_top_halo_size < 0:
        x = x[:, :, :, -curr_top_halo_size:, :]

    # recv halo region from next rank
    if curr_bottom_halo_size > 0:
        bottom_halo_recv = torch.empty(
            [b, c, t, curr_bottom_halo_size, w], dtype=x.dtype, device=device
        )
        comm_ops.append(dist.P2POp(dist.irecv, bottom_halo_recv, rank + 1, group=group))
    elif curr_bottom_halo_size < 0:
        x = x[:, :, :, :curr_bottom_halo_size, :]

    if comm_ops:
        reqs = dist.batch_isend_irecv(comm_ops)
        for req in reqs:
            req.wait()

    if top_halo_recv is not None:
        x = torch.cat([top_halo_recv, x], dim=-2)
    if bottom_halo_recv is not None:
        x = torch.cat([x, bottom_halo_recv], dim=-2)

    return x


class DistZeroPad2d(nn.ZeroPad2d):
    def __init__(
        self,
        padding: _size_4_t,
    ):
        super().__init__(padding)

        self.rank = dist_env.get_vae_rank()
        self.world_size = dist_env.get_vae_group_size()
        self.group = dist_env.get_vae_group()

    def forward(self, x):
        if self.world_size == 1:
            return super().forward(x)

        left, right, top, bottom = self.padding
        top = top if self.rank == 0 else 0
        bottom = bottom if self.rank == self.world_size - 1 else 0
        return F.pad(x, (left, right, top, bottom))


class DistConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # (left_w, right_w, top_h, bottom_h)
        self._padding = (
            self.padding[1],
            self.padding[1],
            self.padding[0],
            self.padding[0],
        )

        self.padding = (0, 0)  # disable padding in nn.Conv2d

        self.rank = dist_env.get_vae_rank()
        self.world_size = dist_env.get_vae_group_size()
        self.group = dist_env.get_vae_group()

    def forward(self, x):
        padding = list(self._padding)

        if self.world_size == 1:
            x = F.pad(x, padding)
            return super().forward(x)

        height = x.shape[-2]
        device = x.device
        patch_height_list = [
            torch.zeros(1, dtype=torch.int64, device=device)
            for _ in range(self.world_size)
        ]
        dist.all_gather(
            patch_height_list,
            torch.tensor([height], dtype=torch.int64, device=device),
            group=self.group,
        )
        patch_height_index = _calc_patch_height_index(patch_height_list)
        self.patch_height_index = patch_height_index.cpu().tolist()

        self.curr_top_halo_size = _calc_top_halo_size(
            self.rank,
            self.world_size,
            self.patch_height_index,
            self.kernel_size[0],
            self.padding[0],
            self.stride[0],
        )

        self.curr_bottom_halo_size = _calc_bottom_halo_size(
            self.rank,
            self.world_size,
            self.patch_height_index,
            self.kernel_size[0],
            self.padding[0],
            self.stride[0],
        )

        self.prev_bottom_halo_size = 0
        if self.rank != 0:
            self.prev_bottom_halo_size = _calc_bottom_halo_size(
                self.rank - 1,
                self.world_size,
                self.patch_height_index,
                self.kernel_size[0],
                self.padding[0],
                self.stride[0],
            )

        self.next_top_halo_size = 0
        if self.rank != self.world_size - 1:
            self.next_top_halo_size = _calc_top_halo_size(
                self.rank + 1,
                self.world_size,
                self.patch_height_index,
                self.kernel_size[0],
                self.padding[0],
                self.stride[0],
            )

        # Reuse 3D halo exchange by inserting a dummy temporal dimension.
        # (b*t, c, h, w) -> (b*t, c, 1, h, w)
        x = halo_exchange(
            x.unsqueeze(2),
            rank=self.rank,
            group=self.group,
            prev_bottom_halo_size=self.prev_bottom_halo_size,
            next_top_halo_size=self.next_top_halo_size,
            curr_top_halo_size=self.curr_top_halo_size,
            curr_bottom_halo_size=self.curr_bottom_halo_size,
        ).squeeze(2)

        # adjust padding
        if self.rank == 0:
            padding[3] = 0  # bottom no padding
        elif self.rank == self.world_size - 1:
            padding[2] = 0  # top no padding
        else:
            padding[2] = 0  # top no padding
            padding[3] = 0  # bottom no padding

        x = F.pad(x, padding)
        return super().forward(x)


class DistWanCausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # (left_w, right_w, top_h, bottom_h, front_t, back_t)
        self.causal_padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )

        self.padding = (0, 0, 0)  # disable padding in nn.Conv3d

        self.rank = dist_env.get_vae_rank()
        self.world_size = dist_env.get_vae_group_size()
        self.group = dist_env.get_vae_group()

    def forward(self, x, cache_x=None):
        padding = list(self.causal_padding)
        if cache_x is not None and self.causal_padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]

        if self.world_size == 1:
            x = F.pad(x, padding)
            return super().forward(x)

        height = x.shape[-2]
        device = x.device
        patch_height_list = [
            torch.zeros(1, dtype=torch.int64, device=device)
            for _ in range(self.world_size)
        ]
        dist.all_gather(
            patch_height_list,
            torch.tensor([height], dtype=torch.int64, device=device),
            group=self.group,
        )
        patch_height_index = _calc_patch_height_index(patch_height_list)
        self.patch_height_index = patch_height_index.cpu().tolist()

        self.curr_top_halo_size = _calc_top_halo_size(
            self.rank,
            self.world_size,
            self.patch_height_index,
            self.kernel_size[1],
            self.padding[1],
            self.stride[1],
        )

        self.curr_bottom_halo_size = _calc_bottom_halo_size(
            self.rank,
            self.world_size,
            self.patch_height_index,
            self.kernel_size[1],
            self.padding[1],
            self.stride[1],
        )

        self.prev_bottom_halo_size = 0
        if self.rank != 0:
            self.prev_bottom_halo_size = _calc_bottom_halo_size(
                self.rank - 1,
                self.world_size,
                self.patch_height_index,
                self.kernel_size[1],
                self.padding[1],
                self.stride[1],
            )

        self.next_top_halo_size = 0
        if self.rank != self.world_size - 1:
            self.next_top_halo_size = _calc_top_halo_size(
                self.rank + 1,
                self.world_size,
                self.patch_height_index,
                self.kernel_size[1],
                self.padding[1],
                self.stride[1],
            )

        x = halo_exchange(
            x,
            rank=self.rank,
            group=self.group,
            prev_bottom_halo_size=self.prev_bottom_halo_size,
            next_top_halo_size=self.next_top_halo_size,
            curr_top_halo_size=self.curr_top_halo_size,
            curr_bottom_halo_size=self.curr_bottom_halo_size,
        )

        # adjust padding
        if self.rank == 0:
            padding[3] = 0  # bottom no padding
        elif self.rank == self.world_size - 1:
            padding[2] = 0  # top no padding
        else:
            padding[2] = 0  # top no padding
            padding[3] = 0  # bottom no padding

        x = F.pad(x, padding)
        return super().forward(x)


class DistLTX2VideoCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        )
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        self.groups = groups
        self.spatial_padding_mode = spatial_padding_mode

        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        # (left_w, right_w, top_h, bottom_h, front_t, back_t)
        self._spatial_padding = (width_pad, width_pad, height_pad, height_pad, 0, 0)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=groups,
            padding=0,
            padding_mode="zeros",
        )

        self.rank = dist_env.get_vae_rank()
        self.world_size = dist_env.get_vae_group_size()
        self.group = dist_env.get_vae_group()

    def _pad_spatial(self, x: torch.Tensor, padding: list[int]) -> torch.Tensor:
        if self.spatial_padding_mode == "zeros":
            return F.pad(x, padding)
        return F.pad(x, padding, mode=self.spatial_padding_mode)

    def forward(self, hidden_states: torch.Tensor, causal: bool = True) -> torch.Tensor:
        time_kernel_size = self.kernel_size[0]

        if causal:
            pad_left = hidden_states[:, :, :1, :, :].repeat(
                (1, 1, time_kernel_size - 1, 1, 1)
            )
            hidden_states = torch.concatenate([pad_left, hidden_states], dim=2)
        else:
            half = (time_kernel_size - 1) // 2
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, half, 1, 1))
            pad_right = hidden_states[:, :, -1:, :, :].repeat((1, 1, half, 1, 1))
            hidden_states = torch.concatenate(
                [pad_left, hidden_states, pad_right], dim=2
            )

        padding = list(self._spatial_padding)
        if self.world_size == 1:
            hidden_states = self._pad_spatial(hidden_states, padding)
            return self.conv(hidden_states)

        height = hidden_states.shape[-2]
        device = hidden_states.device
        patch_height_list = [
            torch.zeros(1, dtype=torch.int64, device=device)
            for _ in range(self.world_size)
        ]
        dist.all_gather(
            patch_height_list,
            torch.tensor([height], dtype=torch.int64, device=device),
            group=self.group,
        )
        patch_height_index = _calc_patch_height_index(patch_height_list)
        self.patch_height_index = patch_height_index.cpu().tolist()

        height_padding = self._spatial_padding[2]
        self.curr_top_halo_size = _calc_top_halo_size(
            self.rank,
            self.world_size,
            self.patch_height_index,
            self.kernel_size[1],
            height_padding,
            self.stride[1],
        )

        self.curr_bottom_halo_size = _calc_bottom_halo_size(
            self.rank,
            self.world_size,
            self.patch_height_index,
            self.kernel_size[1],
            height_padding,
            self.stride[1],
        )

        self.prev_bottom_halo_size = 0
        if self.rank != 0:
            self.prev_bottom_halo_size = _calc_bottom_halo_size(
                self.rank - 1,
                self.world_size,
                self.patch_height_index,
                self.kernel_size[1],
                height_padding,
                self.stride[1],
            )

        self.next_top_halo_size = 0
        if self.rank != self.world_size - 1:
            self.next_top_halo_size = _calc_top_halo_size(
                self.rank + 1,
                self.world_size,
                self.patch_height_index,
                self.kernel_size[1],
                height_padding,
                self.stride[1],
            )

        hidden_states = halo_exchange(
            hidden_states,
            rank=self.rank,
            group=self.group,
            prev_bottom_halo_size=self.prev_bottom_halo_size,
            next_top_halo_size=self.next_top_halo_size,
            curr_top_halo_size=self.curr_top_halo_size,
            curr_bottom_halo_size=self.curr_bottom_halo_size,
        )

        # adjust padding
        if self.rank == 0:
            padding[3] = 0  # bottom no padding
        elif self.rank == self.world_size - 1:
            padding[2] = 0  # top no padding
        else:
            padding[2] = 0  # top no padding
            padding[3] = 0  # bottom no padding

        hidden_states = self._pad_spatial(hidden_states, padding)
        return self.conv(hidden_states)
