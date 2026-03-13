## About

FastVAE is a high-performance framework that accelerates VAE encoding and decoding through parallel implementation while significantly reducing GPU memory footprint.

## Performance

720p video, A800 GPU, bf16. Results are measured after one warmup pass, and peak memory is `torch.cuda.max_memory_allocated()` on rank0.

### Wan2_2

| Processes | Encode (s) | Decode (s) | Total (s) | Peak Mem |
| --- | --- | --- | --- | --- |
| 1 | 3.347 | 11.799 | 15.147 | 22.583 GB |
| 2 | 2.209 | 7.073 | 9.282 | 13.609 GB |
| 4 | 1.473 | 4.081 | 5.555 | 9.121 GB |
| 8 | 1.115 | 2.517 | 3.632 | 7.930 GB |

### Wan2_1

| Processes | Encode (s) | Decode (s) | Total (s) | Peak Mem |
| --- | --- | --- | --- | --- |
| 1 | 6.771 | 11.198 | 17.970 | 18.404 GB |
| 2 | 4.024 | 6.552 | 10.576 | 10.437 GB |
| 4 | 2.431 | 3.871 | 6.302 | 6.452 GB |
| 8 | 1.618 | 2.424 | 4.041 | 5.783 GB |
