## About

FastVAE is a lightweight plugin that accelerates diffusers VAE encoding and decoding through parallel implementation while reducing GPU memory footprint.

## Usage
```python
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from fastvae.dist.env import DistributedEnv as dist_env
from fastvae.models.wan.para_wan_vae import apply_wan_dist_patch, remove_wan_dist_patch

# Baseline
vae = AutoencoderKLWan.from_pretrained(...)
encoded = model.encode(video).latent_dist.sample()
decoded = model.decode(encoded).sample

# Parallel (monkey patch)
dist_env.initialize(vae_group)
apply_wan_dist_patch()
vae = AutoencoderKLWan.from_pretrained(...)
encoded = model.encode(video).latent_dist.sample()
decoded = model.decode(encoded).sample
remove_wan_dist_patch()

```

## Performance

5s 720p video, A800 GPU, bf16. Results are measured after one warmup pass, and peak memory is `torch.cuda.max_memory_allocated()` on rank0.

### Wan2_2

| Processes | Encode (s) | Decode (s) | Total (s) | Peak Mem |
| --- | --- | --- | --- | --- |
| 1 | 2.833 | 10.336 | 13.170 | 13.829 GB |
| 2 | 2.088 | 6.158 | 8.247 | 8.335 GB |
| 4 | 1.480 | 3.561 | 5.042 | 5.590 GB |
| 8 | 1.230 | 2.240 | 3.470 | 4.217 GB |

### Wan2_1

| Processes | Encode (s) | Decode (s) | Total (s) | Peak Mem |
| --- | --- | --- | --- | --- |
| 1 | 5.113 | 9.256 | 14.368 | 11.762 GB |
| 2 | 3.387 | 5.452 | 8.839 | 6.761 GB |
| 4 | 2.088 | 3.269 | 5.357 | 4.261 GB |
| 8 | 1.465 | 2.156 | 3.621 | 3.017 GB |
