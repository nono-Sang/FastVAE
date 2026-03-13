import logging

import torch
import torch.amp as amp
import torch.nn as nn

from fastvae.dist.comm_ops import gather_tensor, split_tensor
from fastvae.models.para_utils import DistCausalConv3d
from fastvae.models.wan.para_shared import (
    CACHE_T,
    DistAttentionBlock,
    DistResample,
    DistResidualBlock,
)
from fastvae.models.wan.vae2_2 import AvgDown3D, DupUp3D, RMS_norm, patchify, unpatchify


class DistDownResidualBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, dropout, mult, temperal_downsample=False, down_flag=False
    ):
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        downsamples = []
        for _ in range(mult):
            downsamples.append(DistResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(DistResample(out_dim, mode=mode, variant="vae22"))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)
        return x + self.avg_shortcut(x_copy)


class DistUpResidualBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, dropout, mult, temperal_upsample=False, up_flag=False
    ):
        super().__init__()
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        upsamples = []
        for _ in range(mult):
            upsamples.append(DistResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(DistResample(out_dim, mode=mode, variant="vae22"))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        return x_main


class DistEncoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv1 = DistCausalConv3d(12, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = (
                temperal_downsample[i] if i < len(temperal_downsample) else False
            )
            downsamples.append(
                DistDownResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            DistResidualBlock(out_dim, out_dim, dropout),
            DistAttentionBlock(out_dim),
            DistResidualBlock(out_dim, out_dim, dropout),
        )

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            DistCausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = split_tensor(x, 3)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.middle:
            if isinstance(layer, DistResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head:
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

        x = gather_tensor(x, 3)
        return x


class DistDecoder3D(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv1 = DistCausalConv3d(z_dim, dims[0], 3, padding=1)

        self.middle = nn.Sequential(
            DistResidualBlock(dims[0], dims[0], dropout),
            DistAttentionBlock(dims[0]),
            DistResidualBlock(dims[0], dims[0], dropout),
        )

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(
                DistUpResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temperal_upsample=t_up_flag,
                    up_flag=i != len(dim_mult) - 1,
                )
            )
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            DistCausalConv3d(out_dim, 12, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        x = split_tensor(x, 3)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, DistResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)

        for layer in self.head:
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
        x = gather_tensor(x, 3)
        return x


def count_dist_conv3d(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, DistCausalConv3d):
            count += 1
    return count


class DistWanVAE_(nn.Module):
    def __init__(
        self,
        dim=160,
        dec_dim=256,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        self.encoder = DistEncoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
        )
        self.conv1 = DistCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = DistCausalConv3d(z_dim, z_dim, 1)
        self.decoder = DistDecoder3D(
            dec_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
        )

    def forward(self, x, scale=[0, 1]):
        mu = self.encode(x, scale)
        x_recon = self.decode(mu, scale)
        return x_recon, mu

    def encode(self, x, scale):
        self.clear_cache()
        x = patchify(x, patch_size=2)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=True,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_], 2)
        out = unpatchify(out, patch_size=2)
        self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_dist_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = count_dist_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _dist_video_vae(pretrained_path=None, z_dim=16, dim=160, device="cpu", **kwargs):
    cfg = dict(
        dim=dim,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, True],
        dropout=0.0,
    )
    cfg.update(**kwargs)

    with torch.device("meta"):
        model = DistWanVAE_(**cfg)

    logging.info(f"loading {pretrained_path}")
    model.load_state_dict(torch.load(pretrained_path, map_location=device), assign=True)

    return model


class DistWan2_2_VAE:
    def __init__(
        self,
        z_dim=48,
        c_dim=160,
        vae_pth=None,
        dim_mult=[1, 2, 4, 4],
        temperal_downsample=[False, True, True],
        dtype=torch.float,
        device="cuda",
    ):
        self.dtype = dtype
        self.device = device

        mean = torch.tensor(
            [
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.1570,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.1230,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.0520,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667,
            ],
            dtype=dtype,
            device=device,
        )
        std = torch.tensor(
            [
                0.4765,
                1.0364,
                0.4514,
                1.1677,
                0.5313,
                0.4990,
                0.4818,
                0.5013,
                0.8158,
                1.0344,
                0.5894,
                1.0901,
                0.6885,
                0.6165,
                0.8454,
                0.4978,
                0.5759,
                0.3523,
                0.7135,
                0.6804,
                0.5833,
                1.4146,
                0.8986,
                0.5659,
                0.7069,
                0.5338,
                0.4889,
                0.4917,
                0.4069,
                0.4999,
                0.6866,
                0.4093,
                0.5709,
                0.6065,
                0.6415,
                0.4944,
                0.5726,
                1.2042,
                0.5458,
                1.6887,
                0.3971,
                1.0600,
                0.3943,
                0.5537,
                0.5444,
                0.4089,
                0.7468,
                0.7744,
            ],
            dtype=dtype,
            device=device,
        )
        self.scale = [mean, 1.0 / std]

        self.model = (
            _dist_video_vae(
                pretrained_path=vae_pth,
                z_dim=z_dim,
                dim=c_dim,
                dim_mult=dim_mult,
                temperal_downsample=temperal_downsample,
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        )

    def encode(self, videos):
        try:
            if not isinstance(videos, list):
                raise TypeError("videos should be a list")
            with amp.autocast(device_type="cuda", dtype=self.dtype):
                return [
                    self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                    for u in videos
                ]
        except TypeError as e:
            logging.info(e)
            return None

    def decode(self, zs):
        try:
            if not isinstance(zs, list):
                raise TypeError("zs should be a list")
            with amp.autocast(device_type="cuda", dtype=self.dtype):
                return [
                    self.model.decode(u.unsqueeze(0), self.scale)
                    .float()
                    .clamp_(-1, 1)
                    .squeeze(0)
                    for u in zs
                ]
        except TypeError as e:
            logging.info(e)
            return None
