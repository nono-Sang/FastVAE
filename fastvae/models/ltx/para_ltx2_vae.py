import logging

import torch

from fastvae.dist.comm_ops import gather_tensor, split_tensor
from fastvae.dist.env import DistributedEnv as dist_env
from fastvae.models.para_utils import DistLTX2VideoCausalConv3d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def _iter_down_spatial_strides(blocks):
#     if not blocks:
#         return
#     for block in blocks:
#         downsamplers = getattr(block, "downsamplers", None)
#         if downsamplers is None:
#             continue
#         for sampler in downsamplers:
#             stride = getattr(sampler, "stride", None)
#             if stride is None and hasattr(sampler, "conv"):
#                 stride = getattr(sampler.conv, "stride", None)
#             if stride is None:
#                 continue
#             yield stride[1]


# def _iter_up_spatial_strides(blocks):
#     if not blocks:
#         return
#     for block in blocks:
#         upsamplers = getattr(block, "upsamplers", None)
#         if upsamplers is None:
#             continue
#         for sampler in upsamplers:
#             stride = getattr(sampler, "stride", None)
#             if stride is None and hasattr(sampler, "conv"):
#                 stride = getattr(sampler.conv, "stride", None)
#             if stride is None:
#                 continue
#             yield stride[1]


# def _calc_down_align_factor(blocks) -> int:
#     factor = 1
#     for s in _iter_down_spatial_strides(blocks) or []:
#         if s == 2:
#             factor *= 2
#     return factor


# def _calc_up_total_factor(blocks) -> int:
#     factor = 1
#     for s in _iter_up_spatial_strides(blocks) or []:
#         if s == 2:
#             factor *= 2
#     return factor


# def _calc_aligned_split_sizes(total: int, align: int) -> list[int] | None:
#     world_size = dist_env.get_vae_group_size()
#     if world_size <= 1 or align <= 1:
#         return None
#     if total % align != 0:
#         return None
#     units = total // align
#     base = units // world_size
#     remainder = units % world_size
#     sizes = [base + (1 if i < remainder else 0) for i in range(world_size)]
#     return [s * align for s in sizes]


# def _downsampled_sizes(sizes: list[int] | None, factor: int) -> list[int] | None:
#     if sizes is None or factor == 1:
#         return sizes
#     out = []
#     start = 0
#     for size in sizes:
#         end = start + size
#         out.append(end // factor - start // factor)
#         start = end
#     return out


# def _upsampled_sizes(sizes: list[int] | None, factor: int) -> list[int] | None:
#     if sizes is None or factor == 1:
#         return sizes
#     return [s * factor for s in sizes]


def _patch_downsampler3d_forward(forward_func):
    def dist_downsampler3d_forward(
        self, hidden_states: torch.Tensor, causal: bool | None = None
    ):
        hidden_states = gather_tensor(hidden_states, dim=3)
        hidden_states = forward_func(self, hidden_states, causal=causal)
        hidden_states = split_tensor(hidden_states, dim=3)
        return hidden_states

    return dist_downsampler3d_forward


def _patch_encoder3d_forward():
    def dist_encoder3d_forward(
        self, hidden_states: torch.Tensor, causal: bool | None = None
    ):
        p = self.patch_size
        p_t = self.patch_size_t

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        causal = causal or self.is_causal

        hidden_states = hidden_states.reshape(
            batch_size,
            num_channels,
            post_patch_num_frames,
            p_t,
            post_patch_height,
            p,
            post_patch_width,
            p,
        )
        # Thanks for driving me insane with the weird patching order :(
        hidden_states = hidden_states.permute(0, 1, 3, 7, 5, 2, 4, 6).flatten(1, 4)

        # # 1. auto split height with aligned sizes for spatial downsampling
        # align = _calc_down_align_factor(self.down_blocks)
        # pad_h = 0
        # if dist_env.get_vae_group_size() > 1 and align > 1:
        #     total_h = hidden_states.shape[3]
        #     required = align * dist_env.get_vae_group_size()
        #     pad_h = (required - total_h % required) % required
        #     if pad_h:
        #         hidden_states = F.pad(hidden_states, (0, 0, 0, pad_h, 0, 0))
        # sizes = _calc_aligned_split_sizes(hidden_states.shape[3], align)
        # hidden_states = split_tensor(hidden_states, dim=3, sizes=sizes)

        hidden_states = split_tensor(hidden_states, dim=3)
        hidden_states = self.conv_in(hidden_states, causal=causal)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for down_block in self.down_blocks:
                hidden_states = self._gradient_checkpointing_func(
                    down_block, hidden_states, None, None, causal
                )

            hidden_states = self._gradient_checkpointing_func(
                self.mid_block, hidden_states, None, None, causal
            )
        else:
            for down_block in self.down_blocks:
                hidden_states = down_block(hidden_states, causal=causal)

            hidden_states = self.mid_block(hidden_states, causal=causal)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states, causal=causal)

        last_channel = hidden_states[:, -1:]
        last_channel = last_channel.repeat(1, hidden_states.size(1) - 2, 1, 1, 1)
        hidden_states = torch.cat([hidden_states, last_channel], dim=1)

        # 2. auto gather height
        # sizes_out = _downsampled_sizes(sizes, align)
        # hidden_states = gather_tensor(hidden_states, dim=3, sizes=sizes_out)
        # if pad_h:
        #     pad_out = pad_h // align
        #     if pad_out:
        #         hidden_states = hidden_states[:, :, :, :-pad_out, :]

        hidden_states = gather_tensor(hidden_states, dim=3)
        return hidden_states

    return dist_encoder3d_forward


def _patch_decoder3d_forward():
    def dist_decoder3d_forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
        causal: bool | None = None,
    ):
        causal = causal or self.is_causal

        # 1. auto split height
        # pad_h = 0
        # sizes = None
        # if dist_env.get_vae_group_size() > 1:
        #     total_h = hidden_states.shape[3]
        #     if total_h < dist_env.get_vae_group_size():
        #         pad_h = dist_env.get_vae_group_size() - total_h
        #         hidden_states = F.pad(hidden_states, (0, 0, 0, pad_h, 0, 0))
        #         sizes = [1] * dist_env.get_vae_group_size()
        # hidden_states = split_tensor(hidden_states, dim=3, sizes=sizes)

        hidden_states = split_tensor(hidden_states, dim=3)
        hidden_states = self.conv_in(hidden_states, causal=causal)

        if self.timestep_scale_multiplier is not None:
            temb = temb * self.timestep_scale_multiplier

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                self.mid_block, hidden_states, temb, None, causal
            )

            for up_block in self.up_blocks:
                hidden_states = self._gradient_checkpointing_func(
                    up_block, hidden_states, temb, None, causal
                )
        else:
            hidden_states = self.mid_block(hidden_states, temb, causal=causal)

            for up_block in self.up_blocks:
                hidden_states = up_block(hidden_states, temb, causal=causal)

        hidden_states = self.norm_out(hidden_states)

        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1).unflatten(1, (2, -1))
            temb = temb + self.scale_shift_table[None, ..., None, None, None]
            shift, scale = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states, causal=causal)

        p = self.patch_size
        p_t = self.patch_size_t

        # 2. auto gather height
        # up_factor = _calc_up_total_factor(self.up_blocks)
        # sizes_out = _upsampled_sizes(sizes, up_factor)
        # hidden_states = gather_tensor(hidden_states, dim=3, sizes=sizes_out)
        # if pad_h:
        #     pad_out = pad_h * up_factor
        #     if pad_out:
        #         hidden_states = hidden_states[:, :, :, :-pad_out, :]

        hidden_states = gather_tensor(hidden_states, dim=3)
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size, -1, p_t, p, p, num_frames, height, width
        )
        hidden_states = (
            hidden_states.permute(0, 1, 5, 2, 6, 4, 7, 3)
            .flatten(6, 7)
            .flatten(4, 5)
            .flatten(2, 3)
        )
        return hidden_states

    return dist_decoder3d_forward


def apply_ltx2_dist_patch():
    from diffusers.models.autoencoders import autoencoder_kl_ltx2 as ltx2_vae

    if not dist_env.get_vae_group():
        logger.warning("LTX2VAE requires VAE group to be provided, skipping.")
        return

    if getattr(ltx2_vae, "_FASTVAE_DIST_PATCHED", False):
        logger.warning("LTX2VAE has already been patched, skipping.")
        return

    # keep originals for optional restore
    ltx2_vae._FASTVAE_DIST_ORIGS = {
        "ltx2_causal_conv3d": ltx2_vae.LTX2VideoCausalConv3d,
        "ltx2_downsampler3d_forward": ltx2_vae.LTXVideoDownsampler3d.forward,
        "ltx2_encoder3d_forward": ltx2_vae.LTX2VideoEncoder3d.forward,
        "ltx2_decoder3d_forward": ltx2_vae.LTX2VideoDecoder3d.forward,
    }

    ltx2_vae.LTX2VideoCausalConv3d = DistLTX2VideoCausalConv3d
    ltx2_vae.LTXVideoDownsampler3d.forward = _patch_downsampler3d_forward(
        ltx2_vae._FASTVAE_DIST_ORIGS["ltx2_downsampler3d_forward"]
    )
    ltx2_vae.LTX2VideoEncoder3d.forward = _patch_encoder3d_forward()
    ltx2_vae.LTX2VideoDecoder3d.forward = _patch_decoder3d_forward()

    ltx2_vae._FASTVAE_DIST_PATCHED = True
    logger.info("LTX2VAE has been patched successfully.")


def remove_ltx2_dist_patch():
    from diffusers.models.autoencoders import autoencoder_kl_ltx2 as ltx2_vae

    if not getattr(ltx2_vae, "_FASTVAE_DIST_PATCHED", False):
        logger.warning("LTX2VAE has not been patched, skipping.")
        return

    origs = getattr(ltx2_vae, "_FASTVAE_DIST_ORIGS", None)
    if origs is None:
        logger.warning("LTX2VAE has not been patched, skipping.")
        return

    ltx2_vae.LTX2VideoCausalConv3d = origs["ltx2_causal_conv3d"]
    ltx2_vae.LTXVideoDownsampler3d.forward = origs["ltx2_downsampler3d_forward"]
    ltx2_vae.LTX2VideoEncoder3d.forward = origs["ltx2_encoder3d_forward"]
    ltx2_vae.LTX2VideoDecoder3d.forward = origs["ltx2_decoder3d_forward"]

    delattr(ltx2_vae, "_FASTVAE_DIST_ORIGS")
    ltx2_vae._FASTVAE_DIST_PATCHED = False
    logger.info("LTX2VAE has been unpatched successfully.")
