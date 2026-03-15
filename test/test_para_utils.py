import os
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
from torch.testing import assert_close

from fastvae.models.para_utils import DistCausalConv3d, DistConv2d, DistZeroPad2d

from .utils import (
    destroy_dist,
    find_free_port,
    get_test_device,
    init_dist,
    split_height,
)


def _reference_zero_pad2d(
    x: torch.Tensor, padding: tuple[int, int, int, int]
) -> torch.Tensor:
    return nn.ZeroPad2d(padding)(x)


def _reference_conv2d(
    x: torch.Tensor,
    state_dict: dict,
    in_ch: int,
    out_ch: int,
    kernel: int,
    stride: int,
    padding: int,
) -> torch.Tensor:
    layer = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
    layer.load_state_dict(state_dict)
    return layer.to(x.device)(x)


def _reference_causal_conv3d(
    x: torch.Tensor,
    state_dict: dict,
    in_ch: int,
    out_ch: int,
    kernel: int,
    stride: int,
    padding: int,
) -> torch.Tensor:
    layer = WanCausalConv3d(in_ch, out_ch, kernel, stride=stride, padding=padding)
    layer.load_state_dict(state_dict)
    return layer.to(x.device)(x)


def _dist_worker(
    rank: int,
    world_size: int,
    init_method: str,
    payload_file: str,
    output_dir: str,
    op: str,
) -> None:
    init_dist(rank, world_size, init_method)
    payload = torch.load(payload_file)
    device = get_test_device(rank)
    x = payload["x"].to(device)
    start, end = split_height(x.shape[-2], world_size, rank)

    if op == "zero_pad2d":
        layer = DistZeroPad2d(payload["padding"])
        out = layer(x[..., start:end, :])
    elif op == "conv2d":
        layer = DistConv2d(
            payload["in_ch"],
            payload["out_ch"],
            kernel_size=payload["kernel"],
            stride=payload["stride"],
            padding=payload["padding"],
        )
        layer.load_state_dict(payload["state_dict"])
        layer.to(device)
        out = layer(x[..., start:end, :])
    elif op == "causal_conv3d":
        layer = DistCausalConv3d(
            payload["in_ch"],
            payload["out_ch"],
            kernel_size=payload["kernel"],
            stride=payload["stride"],
            padding=payload["padding"],
        )
        layer.load_state_dict(payload["state_dict"])
        layer.to(device)
        out = layer(x[..., start:end, :])
    else:
        raise ValueError(f"Unknown op: {op}")

    torch.save(out.cpu(), os.path.join(output_dir, f"rank{rank}.pt"))
    destroy_dist()


def _run_multi_rank(op: str, payload: dict, world_size: int) -> None:
    destroy_dist()
    output_dir = tempfile.mkdtemp()
    payload_file = tempfile.NamedTemporaryFile(delete=False)
    payload_file.close()
    torch.save(payload, payload_file.name)
    init_method = f"tcp://127.0.0.1:{find_free_port()}"

    try:
        mp.spawn(
            _dist_worker,
            args=(world_size, init_method, payload_file.name, output_dir, op),
            nprocs=world_size,
            join=True,
        )

        outputs = []
        for rank in range(world_size):
            outputs.append(torch.load(os.path.join(output_dir, f"rank{rank}.pt")))
        actual = torch.cat(outputs, dim=-2)

        device = get_test_device()
        x = payload["x"].to(device)
        if op == "zero_pad2d":
            expected = _reference_zero_pad2d(x, payload["padding"])
        elif op == "conv2d":
            expected = _reference_conv2d(
                x,
                payload["state_dict"],
                payload["in_ch"],
                payload["out_ch"],
                payload["kernel"],
                payload["stride"],
                payload["padding"],
            )
        elif op == "causal_conv3d":
            expected = _reference_causal_conv3d(
                x,
                payload["state_dict"],
                payload["in_ch"],
                payload["out_ch"],
                payload["kernel"],
                payload["stride"],
                payload["padding"],
            )
        else:
            raise ValueError(f"Unknown op: {op}")

        if op == "zero_pad2d":
            assert torch.equal(actual, expected.cpu())
        elif op == "conv2d":
            assert_close(actual, expected.cpu(), atol=1e-5, rtol=1e-5)
        elif op == "causal_conv3d":
            assert_close(actual, expected.cpu(), atol=1e-3, rtol=1e-3)

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        os.remove(payload_file.name)


@pytest.mark.parametrize("padding", [(1, 2, 3, 4), (0, 1, 2, 0)])
@pytest.mark.parametrize("shape", [(1, 8, 90, 40), (2, 4, 16, 12)])
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_dist_zero_pad2d(padding, shape, world_size):
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    x = torch.randn(*shape)
    payload = {"padding": padding, "x": x}
    _run_multi_rank("zero_pad2d", payload, world_size=world_size)


@pytest.mark.parametrize(
    "case",
    [
        {"shape": (1, 8, 90, 40), "kernel": 3, "padding": 1, "stride": 1},
        {"shape": (2, 4, 16, 12), "kernel": 3, "padding": 1, "stride": 1},
    ],
)
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_dist_conv2d(case, world_size):
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    batch, in_ch, height, width = case["shape"]
    out_ch = in_ch * 3
    conv = nn.Conv2d(
        in_ch, out_ch, case["kernel"], stride=case["stride"], padding=case["padding"]
    )
    state_dict = conv.state_dict()
    x = torch.randn(batch, in_ch, height, width)
    payload = {
        "x": x,
        "in_ch": in_ch,
        "out_ch": out_ch,
        "kernel": case["kernel"],
        "padding": case["padding"],
        "stride": case["stride"],
        "state_dict": state_dict,
    }
    _run_multi_rank("conv2d", payload, world_size=world_size)


@pytest.mark.parametrize(
    "case",
    [
        {"shape": (1, 6, 8, 90, 40), "kernel": 3, "padding": 1, "stride": 1},
        {"shape": (2, 4, 5, 16, 12), "kernel": 3, "padding": 1, "stride": 1},
    ],
)
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_dist_causal_conv3d(case, world_size):
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    batch, in_ch, frames, height, width = case["shape"]
    out_ch = in_ch * 3
    conv = WanCausalConv3d(
        in_ch, out_ch, case["kernel"], stride=case["stride"], padding=case["padding"]
    )
    state_dict = conv.state_dict()
    x = torch.randn(batch, in_ch, frames, height, width)
    payload = {
        "x": x,
        "in_ch": in_ch,
        "out_ch": out_ch,
        "kernel": case["kernel"],
        "padding": case["padding"],
        "stride": case["stride"],
        "state_dict": state_dict,
    }
    _run_multi_rank("causal_conv3d", payload, world_size=world_size)
