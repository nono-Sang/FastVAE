import logging

from fastvae.dist.comm_ops import gather_tensor, split_tensor
from fastvae.dist.env import DistributedEnv as dist_env
from fastvae.models.para_utils import DistConv2d, DistWanCausalConv3d, DistZeroPad2d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _patch_attention_block_forward(forward_func):
    def dist_attention_block_forward(self, x):
        x, sizes = gather_tensor(x, dim=-2, return_sizes=True)
        x = forward_func(self, x)
        x = split_tensor(x, dim=-2, sizes=sizes)
        return x

    return dist_attention_block_forward


def _downsampled_sizes(sizes: list[int], factor: int) -> list[int]:
    if factor == 1 or dist_env.get_vae_group_size() <= 1:
        return sizes
    out_sizes = []
    start = 0
    for size in sizes:
        end = start + size
        out_sizes.append(end // factor - start // factor)
        start = end
    return out_sizes


def _patch_avgdown3d_forward(forward_func):
    def dist_avgdown3d_forward(self, x):
        x, sizes = gather_tensor(x, dim=3, return_sizes=True)
        x = forward_func(self, x)
        sizes_out = _downsampled_sizes(sizes, self.factor_s)
        x = split_tensor(x, dim=3, sizes=sizes_out)
        return x

    return dist_avgdown3d_forward


def _patch_encoder3d_forward(forward_func):
    def dist_encoder3d_forward(self, x, feat_cache=None, feat_idx=[0]):
        x = split_tensor(x, dim=3)
        x = forward_func(self, x, feat_cache, feat_idx)
        x = gather_tensor(x, dim=3)
        return x

    return dist_encoder3d_forward


def _patch_decoder3d_forward(forward_func):
    def dist_decoder3d_forward(
        self, x, feat_cache=None, feat_idx=[0], first_chunk=False
    ):
        x = split_tensor(x, dim=3)
        x = forward_func(self, x, feat_cache, feat_idx, first_chunk)
        x = gather_tensor(x, dim=3)
        return x

    return dist_decoder3d_forward


def apply_wan_dist_patch():
    from diffusers.models.autoencoders import autoencoder_kl_wan as wan_vae

    if not dist_env.get_vae_group():
        logger.warning("WanVAE requires VAE group to be provided, skipping.")
        return

    if getattr(wan_vae, "_FASTVAE_DIST_PATCHED", False):
        logger.warning("WanVAE has already been patched, skipping.")
        return

    # keep originals for optional restore
    wan_vae._FASTVAE_DIST_ORIGS = {
        "conv2d": wan_vae.nn.Conv2d,
        "zero_pad2d": wan_vae.nn.ZeroPad2d,
        "wan_causal_conv3d": wan_vae.WanCausalConv3d,
        "wan_avgdown3d_forward": wan_vae.AvgDown3D.forward,
        "wan_attention_forward": wan_vae.WanAttentionBlock.forward,
        "wan_encoder3d_forward": wan_vae.WanEncoder3d.forward,
        "wan_decoder3d_forward": wan_vae.WanDecoder3d.forward,
    }

    wan_vae.nn.Conv2d = DistConv2d
    wan_vae.nn.ZeroPad2d = DistZeroPad2d
    wan_vae.WanCausalConv3d = DistWanCausalConv3d

    wan_vae.AvgDown3D.forward = _patch_avgdown3d_forward(
        wan_vae._FASTVAE_DIST_ORIGS["wan_avgdown3d_forward"]
    )
    wan_vae.WanAttentionBlock.forward = _patch_attention_block_forward(
        wan_vae._FASTVAE_DIST_ORIGS["wan_attention_forward"]
    )
    wan_vae.WanEncoder3d.forward = _patch_encoder3d_forward(
        wan_vae._FASTVAE_DIST_ORIGS["wan_encoder3d_forward"]
    )
    wan_vae.WanDecoder3d.forward = _patch_decoder3d_forward(
        wan_vae._FASTVAE_DIST_ORIGS["wan_decoder3d_forward"]
    )

    wan_vae._FASTVAE_DIST_PATCHED = True
    logger.info("WanVAE has been patched successfully.")


def remove_wan_dist_patch():
    from diffusers.models.autoencoders import autoencoder_kl_wan as wan_vae

    if not getattr(wan_vae, "_FASTVAE_DIST_PATCHED", False):
        logger.warning("WanVAE has not been patched, skipping.")
        return

    origs = getattr(wan_vae, "_FASTVAE_DIST_ORIGS", None)
    if origs is None:
        logger.warning("WanVAE has not been patched, skipping.")
        return

    wan_vae.nn.Conv2d = origs["conv2d"]
    wan_vae.nn.ZeroPad2d = origs["zero_pad2d"]
    wan_vae.WanCausalConv3d = origs["wan_causal_conv3d"]

    wan_vae.AvgDown3D.forward = origs["wan_avgdown3d_forward"]
    wan_vae.WanAttentionBlock.forward = origs["wan_attention_forward"]
    wan_vae.WanEncoder3d.forward = origs["wan_encoder3d_forward"]
    wan_vae.WanDecoder3d.forward = origs["wan_decoder3d_forward"]

    delattr(wan_vae, "_FASTVAE_DIST_ORIGS")
    wan_vae._FASTVAE_DIST_PATCHED = False
    logger.info("WanVAE has been unpatched successfully.")
