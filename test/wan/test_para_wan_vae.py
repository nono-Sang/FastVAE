import os
import shutil
import tempfile

import pytest
import torch
import torch.multiprocessing as mp
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from torch.testing import assert_close

from fastvae.models.wan.configs import WAN22_DIFFUSERS_CONFIG
from fastvae.models.wan.para_wan_vae import apply_wan_dist_patch, remove_wan_dist_patch

from ..utils import destroy_dist, find_free_port, get_test_device, init_dist


def _para_vae_func(
    rank: int, world_size: int, init_method: str, payload_file: str, output_dir: str
):
    init_dist(rank, world_size, init_method)
    apply_wan_dist_patch()

    payload = torch.load(payload_file)
    device = get_test_device(rank)
    x = payload["x"].to(device)
    z = payload["z"].to(device)
    model_kwargs = payload["model_kwargs"]
    state_dict = payload["state_dict"]

    model = AutoencoderKLWan(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    with torch.no_grad():
        encoder_out = model.encode(x).latent_dist.mode().cpu()
        decoder_out = model.decode(z).sample.cpu()

    if rank == 0:
        torch.save(encoder_out, os.path.join(output_dir, "encoder_out.pt"))
        torch.save(decoder_out, os.path.join(output_dir, "decoder_out.pt"))

    remove_wan_dist_patch()
    destroy_dist()


@pytest.mark.parametrize("model_kwargs", [{}, WAN22_DIFFUSERS_CONFIG])
@pytest.mark.parametrize("x_shape", [(1, 3, 8, 128, 64), (1, 3, 4, 160, 160)])
@pytest.mark.parametrize("world_size", [1, 8])
def test_para_wan_vae(x_shape, model_kwargs, world_size):
    torch.manual_seed(0)

    x = torch.randn(*x_shape)

    ## ref model
    device = get_test_device()
    x_ = x.to(device)
    ref_model = AutoencoderKLWan(**model_kwargs)
    ref_model.to(device).eval()

    with torch.no_grad():
        ref_encoder_out = ref_model.encode(x_).latent_dist.mode().cpu()
        z = torch.randn(*ref_encoder_out.shape)
        z_ = z.to(device)
        ref_decoder_out = ref_model.decode(z_).sample.cpu()

    ## para model
    payload = {
        "x": x,
        "z": z,
        "model_kwargs": model_kwargs,
        "state_dict": ref_model.state_dict(),
    }

    output_dir = tempfile.mkdtemp()
    payload_file = tempfile.NamedTemporaryFile(delete=False)
    payload_file.close()
    torch.save(payload, payload_file.name)
    init_method = f"tcp://127.0.0.1:{find_free_port()}"  # spawn

    try:
        mp.spawn(
            _para_vae_func,
            args=(world_size, init_method, payload_file.name, output_dir),
            nprocs=world_size,
            join=True,
        )
        encoder_out = torch.load(os.path.join(output_dir, "encoder_out.pt"))
        decoder_out = torch.load(os.path.join(output_dir, "decoder_out.pt"))
        assert_close(encoder_out, ref_encoder_out, atol=5e-3, rtol=5e-3)
        assert_close(decoder_out, ref_decoder_out, atol=5e-3, rtol=5e-3)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        os.remove(payload_file.name)
