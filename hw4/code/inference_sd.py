"""
Stable Diffusion Inference Script

This script demonstrates loading SD 1.x checkpoints using our implementation:
1. Modified UNet (UNetForSD) - based on our blocks.py
2. CLIP text encoder from HuggingFace
3. Weight loading function for SD checkpoints
4. TAESD decoder for latent decoding

Usage:
    python inference_sd.py --checkpoint ../sd-v1-4.ckpt --prompt "a sunset"
"""

import argparse
import torch
import yaml
from pathlib import Path

from libs.stable_diffusion import StableDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion Inference (Our Implementation)")
    parser.add_argument(
        "--checkpoint", 
        type=str,
        required=True,
        help="Path to SD checkpoint (.ckpt or .safetensors)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a beautiful sunset over mountains, highly detailed, oil painting",
        help="Text prompt"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="blurry, bad quality, distorted",
        help="Negative prompt"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output image path"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width"
    )
    parser.add_argument(
        "--taesd_path",
        type=str,
        default="../pretrained/taesd_decoder.pth",
        help="Path to TAESD decoder weights"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (optional)"
    )
    parser.add_argument(
        "--use_diffusers",
        action="store_true",
        help="Use diffusers StableDiffusionPipeline for weight loading/inference (recommended for quality)"
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Optional HuggingFace VAE path (e.g., stabilityai/sd-vae-ft-mse) when using diffusers"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("Stable Diffusion Inference (Our Implementation)")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt: {args.prompt}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Load config if provided
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {args.config}")
    
    # Fast path: use diffusers pipeline for high fidelity if requested
    if args.use_diffusers:
        try:
            from diffusers import StableDiffusionPipeline
            import torch
        except ImportError as e:
            raise ImportError("Please pip install diffusers[torch] transformers safetensors to use --use_diffusers") from e

        print("\nUsing diffusers StableDiffusionPipeline for inference...")
        pipe = StableDiffusionPipeline.from_single_file(
            args.checkpoint,
            torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
            use_safetensors=args.checkpoint.endswith(".safetensors"),
            vae=args.vae_path,
        )
        pipe = pipe.to(args.device)
        pipe.set_progress_bar_config(disable=False)
        g = torch.Generator(device=args.device)
        g.manual_seed(args.seed)
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            generator=g,
        ).images[0]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output)
    else:
        # Load model using our implementation
        sd = StableDiffusion.from_pretrained(
            checkpoint_path=args.checkpoint,
            device=args.device,
            taesd_decoder_path=args.taesd_path,
        )
        
        # Generate image
        print(f"\nGenerating image...")
        images = sd.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        
        # Save
        sd.save_image(images, args.output)
    
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print(f"  Prompt: {args.prompt}")
    print(f"  Size: {args.width}x{args.height}")
    print(f"  Steps: {args.steps}")
    print(f"  Guidance: {args.guidance_scale}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
