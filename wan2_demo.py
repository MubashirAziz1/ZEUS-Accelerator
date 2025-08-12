import argparse
import logging
import math
import time
import torch
import lpips

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from zeus import patch

from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video

import torchvision.transforms as T


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a cute cat wearing sunglasses and hawaiian shirt, sipping cocktail by a Spain style mansion pool")
    parser.add_argument("--negative-prompt", type=str, default="bad quality, static")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--model", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers") # Wan-AI/Wan2.1-T2V-14B-Diffusers
    args = parser.parse_args()

    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    lora_path = ""
    seed = args.seed
    prompt = args.prompt
    negative_prompt = args.negative_prompt

    baseline_pipe = WanPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16).to('cuda')
    baseline_pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(baseline_pipe.scheduler.config)

    # Baseline
    logging.info("Running baseline...")
    start_time = time.time()
    set_random_seed(seed)

    ori_output = baseline_pipe(
                 prompt=prompt,
                 negative_prompt=negative_prompt,
                 height=args.height,
                 width=args.width,
                 num_frames=args.frames,
                 guidance_scale=5.0
    ).frames[0]

    baseline_use_time = time.time() - start_time
    logging.info("Baseline: {:.2f} seconds".format(baseline_use_time))

    export_to_video(ori_output, "output_original.mp4", fps=15)

    del baseline_pipe
    torch.cuda.empty_cache()

    # CAP
    pipe = WanPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16).to('cuda')
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    # pipe.load_lora_weights(lora_path)

    patch.apply_patch(pipe,
                      acc_range=(8, 47),

                      interp_mode="psi",
                      denominator=3,
                      modular=(0, 1),

                      lagrange_int=4,
                      lagrange_step=24,
                      lagrange_term=4,

                      max_interval=6)

    logging.info("Running ⚡Zeus...")
    set_random_seed(seed)
    start_time = time.time()

    cap_output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            guidance_scale=5.0
    ).frames[0]
    use_time = time.time() - start_time
    logging.info("⚡Zeus: {:.2f} seconds".format(use_time))

    export_to_video(cap_output, "output.mp4", fps=15)


    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5)),
    ])
    loss_fn_lpips = lpips.LPIPS(net='alex').to('cuda').eval()

    score = 0.0
    for f in range(args.frames):
        ref = transform(ori_output[f]).unsqueeze(0).to('cuda')
        out = transform(cap_output[f]).unsqueeze(0).to('cuda')

        with torch.no_grad():
            score += loss_fn_lpips(ref, out).item()

    print(f"LPIPS: {score / args.frames:.4f}")