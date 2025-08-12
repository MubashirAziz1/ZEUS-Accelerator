import time
import argparse, ast
import numpy as np
import random

import os
from tqdm import tqdm

import torch
from datasets import load_dataset
from torchvision.transforms.functional import to_pil_image

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
    if args.dataset == 'parti':
        prompts = load_dataset("nateraw/parti-prompts", split="train")
    elif args.dataset == 'coco2017':
        dataset = load_dataset("phiyodr/coco2017")
        prompts = [{"Prompt": sample['captions'][0]} for sample in dataset['validation']]
    else:
        raise NotImplementedError

    prompts = prompts[:args.num_fid_samples]

    if args.model == "black-forest-labs/FLUX.1-dev":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to('cuda')
        if args.solver == 'dpm': raise NotImplementedError("Flux only supports Euler Flow-matching")
    else: raise NotImplementedError

    if args.method == 'zeus':
        patch.apply_patch(pipe,
                          acc_range=(args.acc_start, args.acc_end),

                          denominator=args.denominator,
                          modular=args.modular,

                          interp_mode="psi",
                          caching_mode="reuse_all",

                          lagrange_int=args.lagrange_int,
                          lagrange_step=args.lagrange_step,
                          lagrange_term=args.lagrange_term,

                          max_interval=args.max_interval)

    output_dir = args.experiment_folder
    os.makedirs(output_dir, exist_ok=True)

    num_batch = len(prompts) // args.batch_size
    if len(prompts) % args.batch_size != 0:
        num_batch += 1

    global_image_index = 0  # Tracks unique image indices across batches
    use_time = 0

    for i in tqdm(range(num_batch)):
        start, end = args.batch_size * i, min(args.batch_size * (i + 1), len(prompts))
        sample_prompts = [prompts[i]["Prompt"] for i in range(start, end)]

        set_random_seed(args.seed)
        start_time = time.time()
        if args.method != "deep_cache":
            pipe_output = pipe(
                sample_prompts, output_type='np', return_dict=True,
                num_inference_steps=args.steps
            )
        else:
            pipe_output = pipe(
                sample_prompts, num_inference_steps=args.steps,
                cache_interval=args.update_interval,
                cache_layer_id=args.layer, cache_block_id=args.block,
                uniform=args.uniform, pow=args.pow, center=args.center,
                output_type='np', return_dict=True
            )
        use_time += round(time.time() - start_time, 2)
        images = pipe_output.images

        for image in images:
            image = to_pil_image((image * 255).astype(np.uint8))  # Convert to PIL image
            image.save(f"{output_dir}/{global_image_index}.jpg")  # Use global index
            global_image_index += 1

        if args.method == 'zeus':
            patch.reset_cache(pipe)

    print(f"Done: use_time = {use_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # == Sampling setup ==
    parser.add_argument("--model", type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument("--dataset", type=str, default="coco2017")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-fid-samples", type=int, default=40)
    parser.add_argument('--experiment-folder', type=str, default='samples/inference/original')
    parser.add_argument("--solver", type=str, choices=["euler"], default="euler")

    # == Acceleration Setup ==
    parser.add_argument("--method", type=str, choices=["original", "zeus"], default="original")
    parser.add_argument("--acc-start", type=int, default=7)
    parser.add_argument("--acc-end", type=int, default=47)

    parser.add_argument("--denominator", type=int, default=4)
    parser.add_argument("--modular", type=tuple_of_ints, default=(0,1,2))

    parser.add_argument("--lagrange-term", type=int, default=3)
    parser.add_argument("--lagrange-step", type=int, default=24)
    parser.add_argument("--lagrange-int", type=int, default=6)

    parser.add_argument("--max-interval", type=int, default=8)

    args = parser.parse_args()
    set_random_seed(args.seed)
    main(args)