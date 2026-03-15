import os
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanCausalConv3d,
    WanDecoder3d,
    WanEncoder3d,
)
from torch.testing import assert_close

from fastvae.models.wan.configs import WAN22_DIFFUSERS_CONFIG
from fastvae.models.wan.para_wan_vae import apply_wan_dist_patch, remove_wan_dist_patch

from ..utils import destroy_dist, find_free_port, get_test_device, init_dist

DEFAULT_CONFIG = {
    "base_dim": 96,
    "decoder_base_dim": None,
    "z_dim": 16,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
    "non_linearity": "silu",
    "is_residual": False,
    "in_channels": 3,
    "out_channels": 3,
}

TEST_CASES = [
    {
        "name": "wan21_test",
        "model_kwargs": {},
        "encode_shapes": [(1, 3, 8, 128, 64), (1, 3, 8, 64, 32)],
        "decode_shapes": [(1, 16, 4, 64, 32), (1, 16, 2, 48, 24)],
        "warmup": False,
    },
    {
        "name": "wan22_test",
        "model_kwargs": WAN22_DIFFUSERS_CONFIG,
        "encode_shapes": [(1, 12, 8, 128, 64), (1, 12, 8, 64, 32)],
        "decode_shapes": [(1, 48, 4, 64, 32), (1, 48, 2, 48, 24)],
        "warmup": True,
    },
]


def _init_attention_nonzero(model: nn.Module) -> None:
    for module in model.modules():
        if module.__class__.__name__.endswith("AttentionBlock"):
            if hasattr(module, "proj"):
                nn.init.normal_(module.proj.weight, mean=0.0, std=0.02)
                if getattr(module.proj, "bias", None) is not None:
                    nn.init.zeros_(module.proj.bias)


def _build_model_kwargs(case: dict) -> tuple[dict, dict]:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(case["model_kwargs"])
    decoder_base_dim = cfg["decoder_base_dim"] or cfg["base_dim"]
    encoder_kwargs = {
        "in_channels": cfg["in_channels"],
        "dim": cfg["base_dim"],
        "z_dim": cfg["z_dim"] * 2,
        "dim_mult": cfg["dim_mult"],
        "num_res_blocks": cfg["num_res_blocks"],
        "attn_scales": cfg["attn_scales"],
        "temperal_downsample": cfg["temperal_downsample"],
        "dropout": cfg["dropout"],
        "non_linearity": cfg["non_linearity"],
        "is_residual": cfg["is_residual"],
    }
    decoder_kwargs = {
        "dim": decoder_base_dim,
        "z_dim": cfg["z_dim"],
        "dim_mult": cfg["dim_mult"],
        "num_res_blocks": cfg["num_res_blocks"],
        "attn_scales": cfg["attn_scales"],
        "temperal_upsample": cfg["temperal_downsample"][::-1],
        "dropout": cfg["dropout"],
        "non_linearity": cfg["non_linearity"],
        "out_channels": cfg["out_channels"],
        "is_residual": cfg["is_residual"],
    }
    return encoder_kwargs, decoder_kwargs


def _dist_worker_encode(
    rank: int,
    world_size: int,
    init_method: str,
    payload_file: str,
    output_dir: str,
) -> None:
    init_dist(rank, world_size, init_method)
    apply_wan_dist_patch()
    payload = torch.load(payload_file)
    device = get_test_device(rank)
    x = payload["x"].to(device)

    model = WanEncoder3d(**payload["encoder_kwargs"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    with torch.no_grad():
        if payload.get("warmup", False):
            feat_cache = [None] * payload["feat_cache_size"]
            warmup_x = x[..., :1, :, :]
            _ = model(warmup_x, feat_cache=feat_cache, feat_idx=[0])
            out = model(x, feat_cache=feat_cache, feat_idx=[0]).cpu()
        else:
            out = model(x).cpu()

    torch.save(out, os.path.join(output_dir, f"rank{rank}.pt"))
    remove_wan_dist_patch()
    destroy_dist()


def _dist_worker_decode(
    rank: int,
    world_size: int,
    init_method: str,
    payload_file: str,
    output_dir: str,
) -> None:
    init_dist(rank, world_size, init_method)
    apply_wan_dist_patch()
    payload = torch.load(payload_file)
    device = get_test_device(rank)
    z = payload["z"].to(device)

    model = WanDecoder3d(**payload["decoder_kwargs"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    with torch.no_grad():
        if payload.get("warmup", False):
            feat_cache = [None] * payload["feat_cache_size"]
            warmup_z = z[..., :1, :, :]
            _ = model(warmup_z, feat_cache=feat_cache, feat_idx=[0], first_chunk=True)
            out = model(z, feat_cache=feat_cache, feat_idx=[0]).cpu()
        else:
            out = model(z).cpu()

    torch.save(out, os.path.join(output_dir, f"rank{rank}.pt"))
    remove_wan_dist_patch()
    destroy_dist()


def _run_multi_rank_encode(payload: dict, world_size: int) -> None:
    destroy_dist()
    output_dir = tempfile.mkdtemp()
    payload_file = tempfile.NamedTemporaryFile(delete=False)
    payload_file.close()
    torch.save(payload, payload_file.name)
    init_method = f"tcp://127.0.0.1:{find_free_port()}"

    try:
        mp.spawn(
            _dist_worker_encode,
            args=(world_size, init_method, payload_file.name, output_dir),
            nprocs=world_size,
            join=True,
        )

        outputs = []
        for rank in range(world_size):
            outputs.append(torch.load(os.path.join(output_dir, f"rank{rank}.pt")))
        actual = outputs[0]
        for other in outputs[1:]:
            assert torch.equal(other, actual)

        device = get_test_device()
        ref_model = WanEncoder3d(**payload["encoder_kwargs"])
        ref_model.load_state_dict(payload["state_dict"])
        ref_model.to(device)
        ref_model.eval()
        with torch.no_grad():
            if payload.get("warmup", False):
                feat_cache = [None] * payload["feat_cache_size"]
                warmup_x = payload["x"].to(device)[..., :1, :, :]
                _ = ref_model(warmup_x, feat_cache=feat_cache, feat_idx=[0])
                expected = ref_model(
                    payload["x"].to(device), feat_cache=feat_cache, feat_idx=[0]
                ).cpu()
            else:
                expected = ref_model(payload["x"].to(device)).cpu()

        assert_close(actual, expected, atol=5e-3, rtol=5e-3)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        os.remove(payload_file.name)


def _run_multi_rank_decode(payload: dict, world_size: int) -> None:
    destroy_dist()
    output_dir = tempfile.mkdtemp()
    payload_file = tempfile.NamedTemporaryFile(delete=False)
    payload_file.close()
    torch.save(payload, payload_file.name)
    init_method = f"tcp://127.0.0.1:{find_free_port()}"

    try:
        mp.spawn(
            _dist_worker_decode,
            args=(world_size, init_method, payload_file.name, output_dir),
            nprocs=world_size,
            join=True,
        )

        outputs = []
        for rank in range(world_size):
            outputs.append(torch.load(os.path.join(output_dir, f"rank{rank}.pt")))
        actual = outputs[0]
        for other in outputs[1:]:
            assert torch.equal(other, actual)

        device = get_test_device()
        ref_model = WanDecoder3d(**payload["decoder_kwargs"])
        ref_model.load_state_dict(payload["state_dict"])
        ref_model.to(device)
        ref_model.eval()
        with torch.no_grad():
            if payload.get("warmup", False):
                feat_cache = [None] * payload["feat_cache_size"]
                warmup_z = payload["z"].to(device)[..., :1, :, :]
                _ = ref_model(
                    warmup_z, feat_cache=feat_cache, feat_idx=[0], first_chunk=True
                )
                expected = ref_model(
                    payload["z"].to(device), feat_cache=feat_cache, feat_idx=[0]
                ).cpu()
            else:
                expected = ref_model(payload["z"].to(device)).cpu()

        assert_close(actual, expected, atol=5e-3, rtol=5e-3)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        os.remove(payload_file.name)


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
@pytest.mark.parametrize("shape_idx", [0, 1])
@pytest.mark.parametrize("world_size", [1, 8])
def test_dist_autoencoder_kl_wan_encode(case, shape_idx, world_size):
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    torch.manual_seed(0)
    encoder_kwargs, _ = _build_model_kwargs(case)
    shape = case["encode_shapes"][shape_idx]
    x = torch.randn(*shape)
    ref_model = WanEncoder3d(**encoder_kwargs)
    _init_attention_nonzero(ref_model)
    feat_cache_size = sum(isinstance(m, WanCausalConv3d) for m in ref_model.modules())
    payload = {
        "x": x,
        "encoder_kwargs": encoder_kwargs,
        "state_dict": ref_model.state_dict(),
        "warmup": case["warmup"],
        "feat_cache_size": feat_cache_size,
    }
    _run_multi_rank_encode(payload, world_size=world_size)


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
@pytest.mark.parametrize("shape_idx", [0, 1])
@pytest.mark.parametrize("world_size", [1, 8])
def test_dist_autoencoder_kl_wan_decode(case, shape_idx, world_size):
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    torch.manual_seed(0)
    _, decoder_kwargs = _build_model_kwargs(case)
    shape = case["decode_shapes"][shape_idx]
    z = torch.randn(*shape)
    ref_model = WanDecoder3d(**decoder_kwargs)
    _init_attention_nonzero(ref_model)
    feat_cache_size = sum(isinstance(m, WanCausalConv3d) for m in ref_model.modules())
    payload = {
        "z": z,
        "decoder_kwargs": decoder_kwargs,
        "state_dict": ref_model.state_dict(),
        "warmup": case["warmup"],
        "feat_cache_size": feat_cache_size,
    }
    _run_multi_rank_decode(payload, world_size=world_size)
