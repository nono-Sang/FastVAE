# python -m torch.distributed.run --nproc_per_node 8 test/ltx/test_vae_video_e2e.py
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.profiler
from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
from diffusers.utils import export_to_video, load_video
from diffusers.video_processor import VideoProcessor

from fastvae.models.ltx.para_ltx2_vae import apply_ltx2_dist_patch

try:
    from ..utils import get_test_device, init_dist
except ImportError:  # allow running as a script
    import os
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from test.utils import get_test_device, init_dist


def _parse_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def _align_num_frames(video: torch.Tensor, temporal_ratio: int) -> torch.Tensor:
    # LTX2 temporal downsampling expects T ≡ 1 (mod temporal_ratio)
    num_frames = video.shape[2]
    if temporal_ratio <= 1:
        return video
    target_frames = num_frames - ((num_frames - 1) % temporal_ratio)
    if target_frames < 1:
        target_frames = 1
    if target_frames != num_frames:
        video = video[:, :, :target_frames, :, :]
    return video


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    init_method = "env://"  # torchrun
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        init_dist(rank, world_size, init_method)
    apply_ltx2_dist_patch()
    device = get_test_device(rank)
    dtype = _parse_dtype(args.dtype)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ckpt_path = args.ckpt
    video_path = args.input_video

    model = AutoencoderKLLTX2Video.from_pretrained(
        ckpt_path, subfolder="vae", torch_dtype=dtype
    )
    model.to(device).eval()

    video_processor = VideoProcessor(vae_latent_channels=128)
    video = load_video(video_path)
    video = video_processor.preprocess_video(video, args.height, args.width)
    video = _align_num_frames(video, model.temporal_compression_ratio)
    video = video.to(device=device, dtype=dtype)

    # warmup
    warmup_dist = model.encode(video).latent_dist
    warmup_latents = warmup_dist.sample()
    _ = model.decode(warmup_latents).sample
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    enable_profile = args.profile and (args.profile_all_ranks or rank == 0)
    if enable_profile:
        os.makedirs(args.profile_dir, exist_ok=True)
        trace_path = os.path.join(args.profile_dir, f"trace_rank{rank}.json")
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=args.profile_with_stack,
        ) as prof:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.profiler.record_function("encode"):
                encoded = model.encode(video).latent_dist.sample()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            encode_time = time.perf_counter() - start_time

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.profiler.record_function("decode"):
                decoded = model.decode(encoded).sample
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decode_time = time.perf_counter() - start_time
        prof.export_chrome_trace(trace_path)
    else:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        encoded = model.encode(video).latent_dist.sample()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        encode_time = time.perf_counter() - start_time

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        decoded = model.decode(encoded).sample
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_time = time.perf_counter() - start_time

    total_time = encode_time + decode_time

    if rank == 0 and decoded is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        output_videos = video_processor.postprocess_video(decoded, output_type="pil")
        for i, frames in enumerate(output_videos):
            output_path = os.path.join(args.output_dir, f"ltx2_output_{i}.mp4")
            export_to_video(frames, output_path, fps=args.fps)
        if torch.cuda.is_available():
            peak_mem_bytes = torch.cuda.max_memory_allocated()
            peak_mem_gb = peak_mem_bytes / (1024**3)
            peak_mem_str = f"{peak_mem_gb:.3f} GB"
        else:
            peak_mem_str = "N/A"
        print(
            f"[e2e] encode={encode_time:.3f}s decode={decode_time:.3f}s "
            f"total={total_time:.3f}s peak_mem={peak_mem_str}"
        )

    if dist.is_initialized():
        dist.barrier()


def get_args():
    parser = argparse.ArgumentParser()
    # modelscope download --model FastVideo/LTX2-Diffusers --include 'vae/*' --local_dir ./ckpt/ltx2/
    parser.add_argument("--ckpt", default="ckpt/ltx2/")
    parser.add_argument("--input-video", default="video_samples/input_video.mp4")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--width", type=int, default=736)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", default="profiles")
    parser.add_argument("--profile-with-stack", action="store_true")
    parser.add_argument("--profile-all-ranks", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
