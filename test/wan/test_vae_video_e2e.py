import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.profiler
from diffusers.utils import export_to_video, load_video
from diffusers.video_processor import VideoProcessor

from fastvae.dist.env import DistributedEnv
from fastvae.models.wan.para_vae2_1 import DistWan2_1_VAE
from fastvae.models.wan.para_vae2_2 import DistWan2_2_VAE
from fastvae.models.wan.vae2_1 import Wan2_1_VAE
from fastvae.models.wan.vae2_2 import Wan2_2_VAE

MODEL_REGISTRY = {
    "DistWan2_1_VAE": DistWan2_1_VAE,
    "DistWan2_2_VAE": DistWan2_2_VAE,
    "Wan2_1_VAE": Wan2_1_VAE,
    "Wan2_2_VAE": Wan2_2_VAE,
}


def _parse_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def _init_distributed() -> tuple[int, int, int]:
    if not dist.is_available():
        return 0, 0, 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return rank, local_rank, world_size
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    DistributedEnv.initialize(dist.group.WORLD)
    return rank, local_rank, world_size


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    model_cls = MODEL_REGISTRY[args.model]
    rank, local_rank, world_size = _init_distributed()

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else "cpu"
    dtype = _parse_dtype(args.dtype)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ckpt_path = args.ckpt
    video_path = args.input_video

    latent_channels = 48 if "2_2" in args.model else 16
    video_processor = VideoProcessor(vae_latent_channels=latent_channels)

    video = load_video(video_path)
    video = video_processor.preprocess_video(video, args.height, args.width)
    video = video.to(device=device, dtype=dtype)

    model = model_cls(
        vae_pth=ckpt_path,
        dtype=dtype,
        device=device,
    )

    videos = [video[i] for i in range(video.shape[0])]

    # warmup
    warmup_encoded = model.encode(videos)
    _ = model.decode(warmup_encoded)
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
                encoded = model.encode(videos)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            encode_time = time.perf_counter() - start_time

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.profiler.record_function("decode"):
                decoded = model.decode(encoded)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decode_time = time.perf_counter() - start_time
        prof.export_chrome_trace(trace_path)
    else:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        encoded = model.encode(videos)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        encode_time = time.perf_counter() - start_time

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        decoded = model.decode(encoded)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_time = time.perf_counter() - start_time

    total_time = encode_time + decode_time
    decoded_tensor = torch.stack(decoded, dim=0)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        output_videos = video_processor.postprocess_video(
            decoded_tensor, output_type="pil"
        )
        for i, frames in enumerate(output_videos):
            output_path = os.path.join(args.output_dir, f"{args.model}_output_{i}.mp4")
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
    # modelscope download --model Wan-AI/Wan2.1-T2V-14B Wan2.1_VAE.pth --local_dir ./ckpt/wan2_1/
    # modelscope download --model Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local_dir ./ckpt/wan2_2/
    parser.add_argument("--ckpt", default="ckpt/wan2_2/Wan2.2_VAE.pth")
    parser.add_argument("--input-video", default="video_samples/input_video.mp4")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--model", choices=sorted(MODEL_REGISTRY.keys()), default="DistWan2_2_VAE"
    )
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", default="profiles")
    parser.add_argument("--profile-with-stack", action="store_true")
    parser.add_argument("--profile-all-ranks", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
