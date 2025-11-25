# ==========================================================================================
#
# Unified Sampling Script for MIMA Project
# Handles both State Dicts (MIMA) and Full Objects (LoRA Attacks)
# MIT License. To view a copy of the license, visit MIT_LICENSE.md.
#
# ==========================================================================================

import argparse
import sys
import os
import numpy as np
import torch
from PIL import Image

sys.path.append('./')
# We import MIMAPipeline but also standard pipeline for fallback
from model_pipeline import MIMAPipeline
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

def sample(ckpt, delta_ckpt, from_file, prompt, compress, batch_size, outdir, num_inference_steps, guidance_scale, eta, freeze_model, sdxl=False):
    model_id = ckpt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading base model: {model_id}")
    
    # 1. Intelligent Model Loading Logic
    # Strategy: Try to load as MIMA (State Dict) first. If that fails or if it's a full object, switch strategies.
    
    try:
        # Attempt 1: Load as MIMA Pipeline (Expects State Dict in delta_ckpt)
        # This covers: Erase.py output, Train.py output
        if delta_ckpt is not None:
            print(f"Attempting to load delta checkpoint: {delta_ckpt}")
            
            # Peek at the file to see what it is
            loaded_obj = torch.load(delta_ckpt, map_location="cpu")
            
            if isinstance(loaded_obj, dict):
                # It's a dictionary -> Use MIMAPipeline
                print("Detected State Dictionary. Using MIMAPipeline.")
                pipe = MIMAPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
                pipe.load_model(delta_ckpt, compress)
                
            else:
                # It's a Full Object (e.g., from LoRA attack script) -> Use Standard Pipeline
                print("Detected Full Model Object. Using Standard StableDiffusionPipeline.")
                
                # Unwrap if DDP
                if hasattr(loaded_obj, 'module'):
                    unet = loaded_obj.module
                else:
                    unet = loaded_obj
                
                # Inject UNet into standard pipeline
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    safety_checker=None
                ).to(device)
                pipe.unet = unet.to(dtype=torch.float16).to(device)
                
        else:
            # No delta checkpoint -> Just load standard model
            print("No delta checkpoint provided. Loading standard model.")
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to(device)

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Output Setup
    if outdir is None:
        outdir = os.path.dirname(delta_ckpt) if delta_ckpt else "samples"
    os.makedirs(f'{outdir}/samples', exist_ok=True)
    
    generator = torch.Generator(device=device).manual_seed(42)

    # 3. Generation Logic
    prompts_to_run = []
    if prompt is not None:
        prompts_to_run.append(prompt)
    elif from_file:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            prompts_to_run = f.read().splitlines()

    for p in prompts_to_run:
        print(f"Generating for prompt: {p}")
        
        # Generate batch
        images = pipe([p]*batch_size, num_inference_steps, guidance_scale, eta, generator=generator).images
        
        # Stitching (Panel View)
        images_stitched = np.hstack([np.array(x) for x in images])
        panel_image = Image.fromarray(images_stitched)
        
        # Safe Filename
        # takes only first 50 characters of prompt to name the image file
        name = "".join([c for c in p[:50] if c.isalnum() or c in (' ', '-')]).strip().replace(' ', '-')
        panel_image.save(f'{outdir}/{name}.png')
        print(f"Saved panel to {outdir}/{name}.png")

        # Save individual images
        for i, im in enumerate(images):
            im.save(f'{outdir}/samples/{name}_{i}.jpg')


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ckpt', help='target string for query: base model id', default="CompVis/stable-diffusion-v1-4",
                        type=str)
    parser.add_argument('--delta_ckpt', help='target string for query: path to checkpoint (delta or full)', default=None,
                        type=str)
    parser.add_argument('--from-file', help='path to prompt file', default=None, # originally default to './'
                        type=str)
    parser.add_argument('--prompt', help='prompt to generate', default=None,
                        type=str)
    parser.add_argument("--compress", action='store_true', help="Use for compressed (LoRA-style) MIMA checkpoints")
    parser.add_argument("--output_dir", default=None, type=str, help="Directory to save images")
    parser.add_argument("--sdxl", action='store_true') # Unused but kept for compatibility
    parser.add_argument("--batch_size", default=5, type=int, help="Number of images per prompt")
    parser.add_argument('--freeze_model', help='crossattn or crossattn_kv', default='crossattn_kv',
                        type=str) # Unused but kept for compatibility
    # Note: Default These To Official Repo Settings
    parser.add_argument("--num_inference_steps", default=200, type=int)
    parser.add_argument("--guidance_scale", default=6., type=float)
    parser.add_argument("--eta", default=1., type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample(args.ckpt, args.delta_ckpt, args.from_file, args.prompt, args.compress, args.batch_size, args.output_dir, args.num_inference_steps, args.guidance_scale, args.eta, args.freeze_model, args.sdxl)
