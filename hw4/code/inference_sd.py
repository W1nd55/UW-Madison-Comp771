"""
Stable Diffusion Inference Script

This script demonstrates loading and using Stable Diffusion 1.x checkpoints.

Usage:
    python inference_sd.py --checkpoint path/to/sd-v1-4.ckpt --prompt "a photo of an astronaut"
    
    # Using TAESD decoder (faster, smaller)
    python inference_sd.py --checkpoint path/to/sd-v1-4.ckpt --prompt "a cat" --use_taesd

Download SD 1.4/1.5 checkpoints from:
    - https://huggingface.co/CompVis/stable-diffusion-v-1-4-original
    - https://huggingface.co/runwayml/stable-diffusion-v1-5
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from libs.stable_diffusion import StableDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion Inference")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to SD checkpoint (.ckpt or .safetensors)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of an astronaut riding a horse on the moon",
        help="Text prompt for generation"
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
        "--use_taesd",
        action="store_true",
        help="Use TAESD decoder instead of full VAE"
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
        help="Device to use"
    )
    return parser.parse_args()


def tensor_to_pil(tensor):
    """Convert tensor to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.cpu().clamp(0, 1)
    tensor = (tensor * 255).byte()
    tensor = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(tensor)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Stable Diffusion Inference")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt: {args.prompt}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Load model
    print("\nLoading Stable Diffusion...")
    sd = StableDiffusion.from_pretrained(
        args.checkpoint,
        device=args.device,
        use_taesd=args.use_taesd,
        taesd_path=args.taesd_path if args.use_taesd else None,
    )
    
    # Generate
    print(f"\nGenerating image with {args.steps} steps...")
    with torch.no_grad():
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
    image = tensor_to_pil(images)
    image.save(args.output)
    print(f"\nSaved to {args.output}")
    
    # Also display info
    print("\nGeneration complete!")
    print(f"  Prompt: {args.prompt}")
    print(f"  Size: {args.width}x{args.height}")
    print(f"  Steps: {args.steps}")
    print(f"  Guidance: {args.guidance_scale}")
    print(f"  Seed: {args.seed}")


if __name__ == "__main__":
    main()

