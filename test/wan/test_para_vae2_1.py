import os
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.testing import assert_close

from fastvae.models.wan.para_vae2_1 import DistDecoder3D, DistEncoder3d
from fastvae.models.wan.vae2_1 import Decoder3d, Encoder3d

from ..utils import destroy_dist, find_free_port, get_test_device, init_dist


def _init_attention_nonzero(model: nn.Module) -> None:
    for module in model.modules():
        if module.__class__.__name__.endswith("AttentionBlock"):
            if hasattr(module, "proj"):
                nn.init.normal_(module.proj.weight, mean=0.0, std=0.02)
                if getattr(module.proj, "bias", None) is not None:
                    nn.init.zeros_(module.proj.bias)


def _dist_worker_encoder(
    rank: int,
    world_size: int,
    init_method: str,
    payload_file: str,
    output_dir: str,
) -> None:
    init_dist(rank, world_size, init_method)
    payload = torch.load(payload_file)
    device = get_test_device(rank)
    x = payload["x"].to(device)

    model = DistEncoder3d(**payload["model_kwargs"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)

    torch.save(out.cpu(), os.path.join(output_dir, f"rank{rank}.pt"))
    destroy_dist()


def _dist_decoder_worker(
    rank: int,
    world_size: int,
    init_method: str,
    payload_file: str,
    output_dir: str,
) -> None:
    init_dist(rank, world_size, init_method)
    payload = torch.load(payload_file)
    device = get_test_device(rank)
    z = payload["z"].to(device)

    model = DistDecoder3D(**payload["model_kwargs"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(z)

    torch.save(out.cpu(), os.path.join(output_dir, f"rank{rank}.pt"))
    destroy_dist()


def _run_multi_rank_encoder(payload: dict, world_size: int) -> None:
    destroy_dist()
    output_dir = tempfile.mkdtemp()
    payload_file = tempfile.NamedTemporaryFile(delete=False)
    payload_file.close()
    torch.save(payload, payload_file.name)
    init_method = f"tcp://127.0.0.1:{find_free_port()}"

    try:
        mp.spawn(
            _dist_worker_encoder,
            args=(world_size, init_method, payload_file.name, output_dir),
            nprocs=world_size,
            join=True,
        )

        outputs = []
        for rank in range(world_size):
            outputs.append(torch.load(os.path.join(output_dir, f"rank{rank}.pt")))
        actual = outputs[0]
        for other in outputs[1:]:
            assert_close(other, actual, atol=5e-3, rtol=5e-3)

        device = get_test_device()
        ref_model = Encoder3d(**payload["model_kwargs"])
        ref_model.load_state_dict(payload["state_dict"])
        ref_model.to(device)
        ref_model.eval()
        with torch.no_grad():
            expected = ref_model(payload["x"].to(device)).cpu()

        assert_close(actual, expected, atol=5e-3, rtol=5e-3)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        os.remove(payload_file.name)


def _run_multi_rank_decoder(payload: dict, world_size: int) -> None:
    destroy_dist()
    output_dir = tempfile.mkdtemp()
    payload_file = tempfile.NamedTemporaryFile(delete=False)
    payload_file.close()
    torch.save(payload, payload_file.name)
    init_method = f"tcp://127.0.0.1:{find_free_port()}"

    try:
        mp.spawn(
            _dist_decoder_worker,
            args=(world_size, init_method, payload_file.name, output_dir),
            nprocs=world_size,
            join=True,
        )

        outputs = []
        for rank in range(world_size):
            outputs.append(torch.load(os.path.join(output_dir, f"rank{rank}.pt")))
        actual = outputs[0]
        for other in outputs[1:]:
            assert_close(other, actual, atol=5e-3, rtol=5e-3)

        device = get_test_device()
        ref_model = Decoder3d(**payload["model_kwargs"])
        ref_model.load_state_dict(payload["state_dict"])
        ref_model.to(device)
        ref_model.eval()
        with torch.no_grad():
            expected = ref_model(payload["z"].to(device)).cpu()

        assert_close(actual, expected, atol=5e-3, rtol=5e-3)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        os.remove(payload_file.name)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 3, 8, 128, 64),
        (1, 3, 5, 64, 32),
    ],
)
@pytest.mark.parametrize("world_size", [1, 8])
def test_dist_encoder3d_vae21(shape, world_size):
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    torch.manual_seed(0)
    model_kwargs = {}
    x = torch.randn(*shape)

    ref_model = Encoder3d(**model_kwargs)
    _init_attention_nonzero(ref_model)
    payload = {
        "x": x,
        "model_kwargs": model_kwargs,
        "state_dict": ref_model.state_dict(),
    }
    _run_multi_rank_encoder(payload, world_size=world_size)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 4, 64, 32),
        (1, 4, 2, 48, 24),
    ],
)
@pytest.mark.parametrize("world_size", [1, 8])
def test_dist_decoder3d_vae21(shape, world_size):
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    torch.manual_seed(0)
    model_kwargs = {"attn_scales": [0.25]}
    z = torch.randn(*shape)

    ref_model = Decoder3d(**model_kwargs)
    _init_attention_nonzero(ref_model)
    payload = {
        "z": z,
        "model_kwargs": model_kwargs,
        "state_dict": ref_model.state_dict(),
    }
    _run_multi_rank_decoder(payload, world_size=world_size)
