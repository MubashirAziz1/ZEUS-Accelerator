import time
import argparse, ast
import numpy as np
import random
import os
from tqdm import tqdm

import torch
from diffusers import DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video

from zeus import patch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tuple_of_ints(s: str) -> tuple[int, ...]:
    try:
        v = ast.literal_eval(s)
    except Exception:
        raise argparse.ArgumentTypeError(
            "--modular must look like a Python tuple, e.g. '(0,1,2,3)'"
        )
    if not isinstance(v, tuple) or not all(isinstance(i, int) for i in v):
        raise argparse.ArgumentTypeError("Expected a tuple of ints, e.g. '(0,1,2,3)'")
    return v

def main(args):
    if args.dataset == 'vbench':
        prompt_file = os.path.join(args.vbench_root, 'all_category.txt')
        with open(prompt_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        prompts = [{"Prompt": p} for p in lines]
    else:
        raise NotImplementedError(f"Unsupported dataset: {args.dataset}")

    prompts = prompts[:args.num_fid_samples]

    if args.model == "THUDM/CogVideoX-5b":
        from diffusers import CogVideoXPipeline
        pipe = CogVideoXPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif args.model == "Wan-AI/Wan2.1-T2V-14B-Diffusers":
        from diffusers import AutoencoderKLWan, WanPipeline
        vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16).to('cuda')
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        raise NotImplementedError(f"Unsupported model: {args.model}")

    if args.method == 'zeus':
        patch.apply_patch(pipe,
                          acc_range=(args.acc_start, args.acc_end),
                          denominator=args.denominator,
                          modular=args.modular,
                          interp_mode="psi",
                          lagrange_int=args.lagrange_int,
                          lagrange_step=args.lagrange_step,
                          lagrange_term=args.lagrange_term,
                          max_interval=args.max_interval)

    os.makedirs(args.experiment_folder, exist_ok=True)
    global_index = 0
    use_time = 0

    num_batch = (len(prompts) + args.batch_size - 1) // args.batch_size
    for i in tqdm(range(num_batch)):
        start = i * args.batch_size
        end = min(start + args.batch_size, len(prompts))
        sample_prompts = [prompts[j]["Prompt"] for j in range(start, end)]

        set_random_seed(args.seed)
        t0 = time.time()
        pipe_output = pipe(
            sample_prompts,
            num_frames=81,
            height=720,
            width=1280,
            output_type='np',
            return_dict=True,
            num_inference_steps=args.steps
        )
        use_time += time.time() - t0

        videos = pipe_output.frames
        for vid in videos:
            export_to_video(vid, f"{args.experiment_folder}/{global_index}.mp4", fps=8)
            global_index += 1

        if args.method == 'zeus':
            patch.reset_cache(pipe)

    print(f"Done: use_time = {round(use_time, 2)} s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # == Sampling setup ==
    parser.add_argument("--model", type=str, default='THUDM/CogVideoX-5b')
    parser.add_argument("--dataset", type=str, default="vbench")
    parser.add_argument("--vbench-root", type=str, default='./experiment', help="root folder of the cloned V-Bench repo")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-fid-samples", type=int, default=5000)
    parser.add_argument("--experiment-folder", type=str, default='samples/inference/zeus')

    # == Acceleration setup ==
    parser.add_argument("--method", type=str, choices=["original", "zeus"], default="zeus")
    parser.add_argument("--acc-start", type=int, default=10)
    parser.add_argument("--acc-end", type=int, default=45)
    parser.add_argument("--denominator", type=int, default=5)
    parser.add_argument("--modular", type=tuple_of_ints, default=(0, 1, 2, 3))

    parser.add_argument("--lagrange-term", type=int, default=3)
    parser.add_argument("--lagrange-step", type=int, default=24)
    parser.add_argument("--lagrange-int", type=int, default=6)
    parser.add_argument("--max-interval", type=int, default=8)

    args = parser.parse_args()
    set_random_seed(args.seed)
    main(args)
