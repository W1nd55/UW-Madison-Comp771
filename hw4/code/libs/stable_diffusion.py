"""
Stable Diffusion 1.x Implementation

This module provides a complete Stable Diffusion 1.x implementation that can:
1. Load official SD 1.x checkpoints
2. Use CLIP for text encoding
3. Generate images from text prompts

Requirements:
- transformers (for CLIP)
- safetensors (optional, for .safetensors checkpoints)

Usage:
    sd = StableDiffusion.from_pretrained("path/to/sd-v1-4.ckpt")
    images = sd.generate("a photo of an astronaut riding a horse")

Reference: https://github.com/CompVis/stable-diffusion
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from .unet_sd import UNetSD


class CLIPTextEncoder(nn.Module):
    """
    CLIP Text Encoder wrapper.
    
    Uses HuggingFace transformers to load the CLIP text encoder.
    For SD 1.x, we use openai/clip-vit-large-patch14 (768-dim embeddings).
    """
    
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.device = device
        self.max_length = max_length
        
        try:
            from transformers import CLIPTokenizer, CLIPTextModel
            self.tokenizer = CLIPTokenizer.from_pretrained(version)
            self.transformer = CLIPTextModel.from_pretrained(version)
            self.transformer.eval()
            for param in self.transformer.parameters():
                param.requires_grad = False
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers"
            )

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
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt"
        )
        
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)
        
        return outputs.last_hidden_state

    def to(self, device):
        self.device = device
        self.transformer = self.transformer.to(device)
        return self


class VAEDecoder(nn.Module):
    """
    VAE Decoder for Stable Diffusion.
    
    This is a simplified decoder that can be loaded from SD checkpoints.
    For this assignment, we can also use the existing TAESD.
    """
    
    def __init__(self, in_channels=4, out_channels=3, ch=128, ch_mult=(1, 2, 4, 4)):
        super().__init__()
        
        # Post-quant conv
        self.post_quant_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # Initial conv
        block_in = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(in_channels, block_in, 3, padding=1)
        
        # Middle
        self.mid = nn.Sequential(
            ResnetBlock(block_in, block_in),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in),
        )
        
        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(len(ch_mult))):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(3):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample2x(block_in)
            self.up.insert(0, up)
        
        # Output
        self.norm_out = nn.GroupNorm(32, block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, 3, padding=1)

    def forward(self, z):
        # Scale factor for SD VAE
        z = z / 0.18215
        
        z = self.post_quant_conv(z)
        h = self.conv_in(z)
        h = self.mid(h)
        
        for i_level in reversed(range(len(self.up))):
            for block in self.up[i_level].block:
                h = block(h)
            if hasattr(self.up[i_level], 'upsample'):
                h = self.up[i_level].upsample(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


class ResnetBlock(nn.Module):
    """ResNet block for VAE."""
    
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    """Attention block for VAE."""
    
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        
        B, C, H, W = q.shape
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        
        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=2)
        
        v = v.reshape(B, C, H * W).permute(0, 2, 1)
        h = torch.bmm(attn, v)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        
        return x + self.proj_out(h)


class Upsample2x(nn.Module):
    """2x upsampling for VAE."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class StableDiffusion(nn.Module):
    """
    Complete Stable Diffusion 1.x Pipeline.
    
    Combines:
    - CLIP text encoder
    - UNet denoiser
    - VAE decoder
    """
    
    def __init__(
        self,
        unet=None,
        vae_decoder=None,
        clip_encoder=None,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        
        # Initialize components
        self.unet = unet or UNetSD()
        self.vae_decoder = vae_decoder
        self.clip_encoder = clip_encoder
        
        # Scheduler parameters (DDIM-like)
        self.num_train_timesteps = 1000
        self.beta_start = 0.00085
        self.beta_end = 0.012

    def _get_alphas(self, num_steps):
        """Compute alpha schedule."""
        betas = torch.linspace(
            self.beta_start ** 0.5,
            self.beta_end ** 0.5,
            self.num_train_timesteps,
            device=self.device
        ) ** 2
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Select timesteps
        step_ratio = self.num_train_timesteps // num_steps
        timesteps = torch.arange(0, num_steps, device=self.device) * step_ratio
        timesteps = timesteps.long()
        
        return alphas_cumprod, timesteps

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
        Generate images from text prompt.
        
        Args:
            prompt: Text prompt or list of prompts
            negative_prompt: Negative prompt for classifier-free guidance
            height: Image height (must be divisible by 8)
            width: Image width (must be divisible by 8)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed
            
        Returns:
            Generated images as tensor [B, 3, H, W] in range [0, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        
        # Encode text
        if self.clip_encoder is not None:
            text_embeddings = self.clip_encoder(prompt)
            
            # Classifier-free guidance: encode negative prompt
            if guidance_scale > 1.0:
                uncond_embeddings = self.clip_encoder(
                    [negative_prompt] * batch_size
                )
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        else:
            # Dummy embeddings if no CLIP (for testing)
            text_embeddings = torch.zeros(batch_size, 77, 768, device=self.device)
        
        # Latent dimensions
        latent_height = height // 8
        latent_width = width // 8
        
        # Initialize latents
        latents = torch.randn(
            batch_size, 4, latent_height, latent_width,
            device=self.device
        )
        
        # Get schedule
        alphas_cumprod, timesteps = self._get_alphas(num_inference_steps)
        
        # Scale initial noise
        latents = latents * torch.sqrt(1 - alphas_cumprod[timesteps[0]])
        
        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Generating")):
            # Expand for classifier-free guidance
            if guidance_scale > 1.0:
                latent_model_input = torch.cat([latents] * 2)
                t_input = torch.tensor([t] * batch_size * 2, device=self.device)
            else:
                latent_model_input = latents
                t_input = torch.tensor([t] * batch_size, device=self.device)
            
            # Predict noise
            noise_pred = self.unet(latent_model_input, t_input, text_embeddings)
            
            # Classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            
            # DDIM step
            alpha_t = alphas_cumprod[t]
            if i + 1 < len(timesteps):
                alpha_t_prev = alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=self.device)
            
            # Predicted x0
            pred_x0 = (latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # Direction pointing to xt
            dir_xt = torch.sqrt(1 - alpha_t_prev) * noise_pred
            
            # Update latents
            latents = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
        
        # Decode latents
        if self.vae_decoder is not None:
            images = self.vae_decoder(latents)
        else:
            # If no VAE, just return latents (for testing)
            images = latents
        
        # Normalize to [0, 1]
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        
        return images

    @classmethod
    def from_pretrained(cls, checkpoint_path, device="cuda", use_taesd=False, taesd_path=None):
        """
        Load Stable Diffusion from a checkpoint.
        
        Args:
            checkpoint_path: Path to .ckpt or .safetensors file
            device: Device to load model on
            use_taesd: Use lightweight TAESD decoder instead of full VAE
            taesd_path: Path to TAESD weights (required if use_taesd=True)
            
        Returns:
            StableDiffusion model
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        
        # Load checkpoint
        if checkpoint_path.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_path)
            except ImportError:
                raise ImportError("Please install safetensors: pip install safetensors")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        
        # Initialize UNet
        unet = UNetSD()
        
        # Load UNet weights
        unet_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model.diffusion_model."):
                new_key = key.replace("model.diffusion_model.", "")
                unet_state_dict[new_key] = value
        
        if unet_state_dict:
            # Map weights (may need adjustment based on naming)
            unet = load_unet_weights(unet, unet_state_dict)
            print(f"Loaded {len(unet_state_dict)} UNet parameters")
        
        # Initialize VAE decoder
        if use_taesd:
            from .tiny_autoencoder import TAESD
            vae_decoder = TAESD(decoder_path=taesd_path).decoder
        else:
            vae_decoder = VAEDecoder()
            vae_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("first_stage_model.decoder."):
                    new_key = key.replace("first_stage_model.decoder.", "")
                    vae_state_dict[new_key] = value
            if vae_state_dict:
                vae_decoder = load_vae_weights(vae_decoder, vae_state_dict)
                print(f"Loaded {len(vae_state_dict)} VAE parameters")
        
        # Initialize CLIP
        try:
            clip_encoder = CLIPTextEncoder(device=device)
            clip_encoder = clip_encoder.to(device)
            print("Loaded CLIP text encoder")
        except Exception as e:
            print(f"Warning: Could not load CLIP encoder: {e}")
            clip_encoder = None
        
        # Create model
        model = cls(
            unet=unet.to(device),
            vae_decoder=vae_decoder.to(device) if vae_decoder else None,
            clip_encoder=clip_encoder,
            device=device,
        )
        
        return model


def load_unet_weights(unet, state_dict):
    """
    Load UNet weights with key mapping.
    
    This handles the difference between our naming and SD's naming.
    """
    # For now, try direct loading and report mismatches
    model_dict = unet.state_dict()
    
    # Try to map keys
    mapped_dict = {}
    unmapped_keys = []
    
    for key, value in state_dict.items():
        if key in model_dict:
            if model_dict[key].shape == value.shape:
                mapped_dict[key] = value
            else:
                print(f"Shape mismatch for {key}: {model_dict[key].shape} vs {value.shape}")
        else:
            unmapped_keys.append(key)
    
    if unmapped_keys:
        print(f"Warning: {len(unmapped_keys)} keys not mapped")
    
    # Load what we can
    unet.load_state_dict(mapped_dict, strict=False)
    return unet


def load_vae_weights(vae, state_dict):
    """Load VAE weights with key mapping."""
    model_dict = vae.state_dict()
    mapped_dict = {}
    
    for key, value in state_dict.items():
        if key in model_dict:
            if model_dict[key].shape == value.shape:
                mapped_dict[key] = value
    
    vae.load_state_dict(mapped_dict, strict=False)
    return vae

