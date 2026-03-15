from fastvae.dist.comm_ops import gather_tensor, split_tensor
from fastvae.models.para_utils import DistCausalConv3d, DistConv2d, DistZeroPad2d


def patch_attention_block_forward(forward_func):
    def dist_attention_block_forward(self, x):
        x, sizes = gather_tensor(x, dim=-2, return_sizes=True)
        x = forward_func(self, x)
        x = split_tensor(x, dim=-2, sizes=sizes)
        return x

    return dist_attention_block_forward


def patch_encoder3d_forward(forward_func):
    def dist_encoder3d_forward(self, x, feat_cache=None, feat_idx=[0]):
        x = split_tensor(x, dim=3)
        x = forward_func(self, x, feat_cache, feat_idx)
        x = gather_tensor(x, dim=3)
        return x

    return dist_encoder3d_forward


def patch_decoder3d_forward(forward_func):
    def dist_decoder3d_forward(
        self, x, feat_cache=None, feat_idx=[0], first_chunk=False
    ):
        x = split_tensor(x, dim=3)
        x = forward_func(self, x, feat_cache, feat_idx, first_chunk)
        x = gather_tensor(x, dim=3)
        return x

    return dist_decoder3d_forward


def apply_wan_dist_patch():
    """
    Monkey patch diffusers WanVAE to use distributed ops.
    """
    from diffusers.models.autoencoders import autoencoder_kl_wan as wan_vae

    if getattr(wan_vae, "_FASTVAE_DIST_PATCHED", False):
        return

    # keep originals for optional restore
    wan_vae._FASTVAE_DIST_ORIGS = {
        "conv2d": wan_vae.nn.Conv2d,
        "zero_pad2d": wan_vae.nn.ZeroPad2d,
        "wan_causal_conv3d": wan_vae.WanCausalConv3d,
        "wan_attention_forward": wan_vae.WanAttentionBlock.forward,
        "wan_encoder3d_forward": wan_vae.WanEncoder3d.forward,
        "wan_decoder3d_forward": wan_vae.WanDecoder3d.forward,
    }

    wan_vae.nn.Conv2d = DistConv2d
    wan_vae.nn.ZeroPad2d = DistZeroPad2d
    wan_vae.WanCausalConv3d = DistCausalConv3d

    wan_vae.WanAttentionBlock.forward = patch_attention_block_forward(
        wan_vae._FASTVAE_DIST_ORIGS["wan_attention_forward"]
    )
    wan_vae.WanEncoder3d.forward = patch_encoder3d_forward(
        wan_vae._FASTVAE_DIST_ORIGS["wan_encoder3d_forward"]
    )
    wan_vae.WanDecoder3d.forward = patch_decoder3d_forward(
        wan_vae._FASTVAE_DIST_ORIGS["wan_decoder3d_forward"]
    )

    wan_vae._FASTVAE_DIST_PATCHED = True


def remove_wan_dist_patch():
    """
    Restore patched methods if needed (optional).
    """
    from diffusers.models.autoencoders import autoencoder_kl_wan as wan_vae

    if not getattr(wan_vae, "_FASTVAE_DIST_PATCHED", False):
        return

    origs = getattr(wan_vae, "_FASTVAE_DIST_ORIGS", None)
    if origs is None:
        return

    wan_vae.nn.Conv2d = origs["conv2d"]
    wan_vae.nn.ZeroPad2d = origs["zero_pad2d"]
    wan_vae.WanCausalConv3d = origs["wan_causal_conv3d"]

    wan_vae.WanAttentionBlock.forward = origs["wan_attention_forward"]
    wan_vae.WanEncoder3d.forward = origs["wan_encoder3d_forward"]
    wan_vae.WanDecoder3d.forward = origs["wan_decoder3d_forward"]

    delattr(wan_vae, "_FASTVAE_DIST_ORIGS")
    wan_vae._FASTVAE_DIST_PATCHED = False
