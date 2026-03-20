import logging

import torch

from fastvae.dist.comm_ops import gather_tensor, split_tensor
from fastvae.dist.env import DistributedEnv as dist_env
from fastvae.models.para_utils import DistLTX2VideoCausalConv3d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
