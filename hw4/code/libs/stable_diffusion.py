"""
Stable Diffusion 1.x Implementation

This module implements Stable Diffusion by:
1. Modified UNet (UNetForSD) - based on our existing blocks, compatible with SD weights
2. CLIP text encoder from HuggingFace
3. Weight loading function that maps SD checkpoints to our architecture
4. LDM inference pipeline

Requirements:
- transformers (for CLIP)

Usage:
    sd = StableDiffusion.from_pretrained("path/to/sd-v1-4.ckpt")
    images = sd.generate("a photo of an astronaut riding a horse")
"""

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from .unet_sd import UNetForSD, load_sd_checkpoint
from .tiny_autoencoder import TAESD


class CLIPTextEncoder(nn.Module):
    """
    CLIP Text Encoder from OpenAI via HuggingFace.
    
    Uses openai/clip-vit-large-patch14 which outputs 768-dim embeddings,
    matching SD 1.x's context dimension.
    """
    
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.device = device
        self.max_length = max_length
        
        try:
            from transformers import CLIPTokenizer, CLIPTextModel
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        print(f"Loading CLIP text encoder: {version}")
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.transformer.eval()
        
        # Freeze CLIP
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(self, text):
        """
        Encode text to CLIP embeddings.
        
        Args:
            text: str or list of str
            
        Returns:
            [B, 77, 768] tensor of text embeddings
        """
        if isinstance(text, str):
            text = [text]
        
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        tokens = batch_encoding["input_ids"].to(self.device)
        
        with torch.no_grad():
            outputs = self.transformer(input_ids=tokens)
        
        return outputs.last_hidden_state

    def to(self, device):
        self.device = device
        self.transformer = self.transformer.to(device)
        return self


class StableDiffusion(nn.Module):
    """
    Stable Diffusion 1.x Pipeline using our implementation.
    
    Components:
    - UNetForSD: Our modified UNet that loads SD weights
    - CLIP: Text encoder for conditioning
    - TAESD: Lightweight VAE decoder (uses pretrained weights)
    """
    
    def __init__(
        self,
        unet,
        clip_encoder,
        vae_decoder=None,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.unet = unet
        self.clip_encoder = clip_encoder
        self.vae_decoder = vae_decoder
        
        # DDPM schedule parameters (SD 1.x uses scaled linear)
        self.num_train_timesteps = 1000
        self.beta_start = 0.00085
        self.beta_end = 0.012
        
        # Precompute alphas
        betas = torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, 
                               self.num_train_timesteps) ** 2
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    @classmethod
    def from_pretrained(cls, checkpoint_path, device="cuda", taesd_decoder_path=None):
        """
        Load Stable Diffusion from checkpoint.
        
        Args:
            checkpoint_path: Path to SD .ckpt or .safetensors file
            device: Device to use
            taesd_decoder_path: Path to TAESD decoder weights (optional, for decoding)
            
        Returns:
            StableDiffusion model ready for inference
        """
        print("=" * 60)
        print("Loading Stable Diffusion (Our Implementation)")
        print("=" * 60)
        
        # 1. Create and load UNet
        print("\n[1/3] Loading UNet...")
        unet = UNetForSD(
            in_channels=4,
            out_channels=4,
            dim=320,
            context_dim=768,
            dim_mults=(1, 2, 4, 4),
            attn_levels=(1, 2, 3),
            num_res_blocks=2,
        )
        unet = load_sd_checkpoint(unet, checkpoint_path, device="cpu")
        unet = unet.to(device)
        unet.eval()
        
        # 2. Load CLIP text encoder
        print("\n[2/3] Loading CLIP text encoder...")
        clip_encoder = CLIPTextEncoder(device=device)
        clip_encoder = clip_encoder.to(device)
        
        # 3. Load VAE decoder (TAESD - lightweight version)
        print("\n[3/3] Loading VAE decoder...")
        vae_decoder = None
        if taesd_decoder_path:
            try:
                # Pass encoder_path=None to skip loading encoder
                taesd = TAESD(encoder_path=None, decoder_path=taesd_decoder_path)
                vae_decoder = taesd.decoder.to(device)
                vae_decoder.eval()
                print(f"Loaded TAESD decoder from {taesd_decoder_path}")
            except Exception as e:
                print(f"Warning: Could not load TAESD: {e}")
        
        print("\n" + "=" * 60)
        print("Model loaded successfully!")
        print("=" * 60)
        
        return cls(
            unet=unet,
            clip_encoder=clip_encoder,
            vae_decoder=vae_decoder,
            device=device,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt,
        negative_prompt="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=None,
    ):
        """
        Generate images from text prompt using DDIM sampling.
        
        Args:
            prompt: Text prompt or list of prompts
            negative_prompt: Negative prompt for classifier-free guidance
            height: Image height (must be divisible by 8)
            width: Image width (must be divisible by 8)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale (higher = more prompt adherence)
            seed: Random seed
            
        Returns:
            Generated images as tensor [B, 3, H, W] in range [0, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        
        # Encode text with CLIP
        print("Encoding text...")
        text_embeddings = self.clip_encoder(prompt)
        
        # For classifier-free guidance, also encode negative prompt
        if guidance_scale > 1.0:
            uncond_embeddings = self.clip_encoder([negative_prompt] * batch_size)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prepare latent dimensions
        latent_height = height // 8
        latent_width = width // 8
        
        # Start from random noise
        latents = torch.randn(
            batch_size, 4, latent_height, latent_width,
            device=self.device
        )
        
        # Get timesteps for DDIM
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps, device=self.device) * step_ratio
        timesteps = timesteps.flip(0)  # Reverse: go from high noise to low
        
        # Move alphas to device
        alphas_cumprod = self.alphas_cumprod.to(self.device)
        
        # Scale initial noise
        latents = latents * torch.sqrt(1 - alphas_cumprod[int(timesteps[0])])
        
        print(f"Generating with {num_inference_steps} steps...")
        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            t_batch = t.expand(batch_size).long()
            
            # Expand latents for CFG
            if guidance_scale > 1.0:
                latent_input = torch.cat([latents] * 2)
                t_input = t_batch.repeat(2)
            else:
                latent_input = latents
                t_input = t_batch
            
            # Predict noise
            noise_pred = self.unet(latent_input, text_embeddings, t_input)
            
            # Apply classifier-free guidance
            if guidance_scale > 1.0:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            
            # DDIM step
            alpha_t = alphas_cumprod[int(t)]
            if i + 1 < len(timesteps):
                alpha_prev = alphas_cumprod[int(timesteps[i + 1])]
            else:
                alpha_prev = torch.tensor(1.0, device=self.device)
            
            # Predicted x0
            pred_x0 = (latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_prev) * noise_pred
            
            # Update latents
            latents = torch.sqrt(alpha_prev) * pred_x0 + dir_xt
        
        # Decode latents to images
        print("Decoding latents...")
        if self.vae_decoder is not None:
            # Scale latents (SD uses scaling factor 0.18215)
            latents = latents / 0.18215
            images = self.vae_decoder(latents)
        else:
            # Without VAE, just return normalized latents (won't look like real images)
            print("Warning: No VAE decoder - returning raw latents")
            images = latents[:, :3]  # Take first 3 channels
        
        # Normalize to [0, 1]
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        
        return images
    
    def save_image(self, tensor, path):
        """Save tensor as image."""
        from PIL import Image
        import numpy as np
        
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        img = tensor.cpu().permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(path)
        print(f"Saved to {path}")
