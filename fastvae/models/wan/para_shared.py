import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fastvae.dist.env import DistributedEnv as dist_env
from fastvae.models.para_utils import DistCausalConv3d, DistConv2d, DistZeroPad2d
from fastvae.models.wan.vae2_1 import RMS_norm, Upsample

CACHE_T = 2


class DistResample(nn.Module):
    def __init__(self, dim, mode, variant: str = "vae21"):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        if variant not in ("vae21", "vae22"):
            raise ValueError(f"Unsupported variant: {variant}")
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.variant = variant

        up_out_channels = dim // 2 if variant == "vae21" else dim

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                DistConv2d(dim, up_out_channels, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                DistConv2d(dim, up_out_channels, 3, padding=1),
            )
            self.time_conv = DistCausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0)
            )
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                DistZeroPad2d((0, 1, 0, 1)),
                DistConv2d(dim, dim, 3, stride=(2, 2)),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                DistZeroPad2d((0, 1, 0, 1)),
                DistConv2d(dim, dim, 3, stride=(2, 2)),
            )
            self.time_conv = DistCausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] != "Rep"
                    ):
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :]
                                .unsqueeze(2)
                                .to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] == "Rep"
                    ):
                        cache_x = torch.cat(
                            [torch.zeros_like(cache_x).to(cache_x.device), cache_x],
                            dim=2,
                        )
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)

        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)
                    )
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class DistResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            DistCausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            DistCausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = (
            DistCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, DistCausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class DistAttentionBlock(nn.Module):
    """
    Causal self-attention with a single head for height-split input.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        nn.init.zeros_(self.proj.weight)

    def _gather_full_height(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[int] | None]:
        world_size = dist_env.get_vae_group_size()
        if world_size <= 1:
            return x, None

        group = dist_env.get_vae_group()
        height = torch.tensor([x.shape[-2]], device=x.device, dtype=torch.int64)
        size_list = [torch.zeros_like(height) for _ in range(world_size)]
        dist.all_gather(size_list, height, group=group)
        sizes = [int(s.item()) for s in size_list]

        max_h = max(sizes)
        if x.shape[-2] < max_h:
            x = F.pad(x, (0, 0, 0, max_h - x.shape[-2], 0, 0))

        patch_list = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(patch_list, x, group=group)
        patch_list = [
            t[..., : sizes[i], :].contiguous() for i, t in enumerate(patch_list)
        ]
        return torch.cat(patch_list, dim=-2), sizes

    def _split_height(self, x: torch.Tensor, sizes: list[int] | None) -> torch.Tensor:
        if sizes is None:
            return x
        rank = dist_env.get_vae_rank()
        start = sum(sizes[:rank])
        end = start + sizes[rank]
        return x[..., start:end, :].contiguous()

    def forward(self, x):
        x_full, sizes = self._gather_full_height(x)
        identity = x_full
        b, c, t, h, w = x_full.size()
        x_full = rearrange(x_full, "b c t h w -> (b t) c h w")
        x_full = self.norm(x_full)
        q, k, v = (
            self.to_qkv(x_full)
            .reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        x_full = F.scaled_dot_product_attention(q, k, v)
        x_full = x_full.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x_full = self.proj(x_full)
        x_full = rearrange(x_full, "(b t) c h w-> b c t h w", t=t)
        x_full = x_full + identity
        return self._split_height(x_full, sizes)
