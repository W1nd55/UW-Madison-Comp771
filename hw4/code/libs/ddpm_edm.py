"""
EDM (Elucidating the Design Space of Diffusion-Based Generative Models)
Implementation based on Karras et al. (NeurIPS 2022)

Key improvements over vanilla DDPM:
1. Preconditioning: Scale network inputs/outputs based on noise level σ
2. Continuous noise levels: Use σ directly instead of discrete timesteps
3. Log-normal noise sampling: Better coverage of noise levels during training
4. Improved loss weighting: λ(σ) balances loss across noise levels
5. Heun's 2nd-order sampler: Better quality with fewer steps

Reference: https://arxiv.org/abs/2206.00364
Official code: https://github.com/NVlabs/edm
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

from .unet import UNet
from .unet_improved import ImprovedUNet
from .tiny_autoencoder import TAESD


class EDM(nn.Module):
    """
    EDM: Elucidated Diffusion Model
    
    A unified framework for diffusion models with improved:
    - Preconditioning (better gradient flow)
    - Noise schedule (log-normal sampling)
    - Loss weighting (balanced across noise levels)
    - Sampling (Heun's 2nd-order method)
    """

    def __init__(
        self,
        img_shape=(3, 32, 32),
        timesteps=50,  # EDM typically needs fewer steps
        dim=64,
        context_dim=64,
        num_classes=10,
        dim_mults=(1, 2, 4),
        attn_levels=(0, 1),
        use_vae=False,
        vae_encoder_weights=None,
        vae_decoder_weights=None,
        # EDM-specific parameters
        sigma_min=0.002,      # Minimum noise level
        sigma_max=80.0,       # Maximum noise level
        sigma_data=0.5,       # Data standard deviation (for normalized data)
        rho=7.0,              # Schedule parameter
        P_mean=-1.2,          # Log-normal mean for training
        P_std=1.2,            # Log-normal std for training
        # UNet options
        unet_type="standard",
        num_res_blocks=2,
        dropout=0.1,
        attn_depth=1,
        use_skip_scale=True,
    ):
        super().__init__()

        assert len(img_shape) == 3
        self.timesteps = timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.use_vae = use_vae

        # If VAE is considered, input/output dim will be the dim of latent
        if use_vae:
            in_channels = out_channels = 4
            self.img_shape = [4, img_shape[1] // 8, img_shape[2] // 8]
        else:
            in_channels = out_channels = img_shape[0]
            self.img_shape = img_shape

        # The denoising model using UNet
        # Note: EDM uses continuous sigma, so we disable positional encoding for time
        if unet_type == "improved":
            self.model = ImprovedUNet(
                dim,
                context_dim,
                num_classes,
                time_embed_pe=False,  # EDM uses continuous sigma
                in_channels=in_channels,
                out_channels=out_channels,
                dim_mults=dim_mults,
                attn_levels=attn_levels,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                attn_depth=attn_depth,
                use_skip_scale=use_skip_scale,
            )
        else:
            self.model = UNet(
                dim,
                context_dim,
                num_classes,
                time_embed_pe=False,  # EDM uses continuous sigma
                in_channels=in_channels,
                out_channels=out_channels,
                dim_mults=dim_mults,
                attn_levels=attn_levels
            )

        # If we should consider latent EDM
        if use_vae:
            assert vae_encoder_weights is not None
            assert vae_decoder_weights is not None
            self.vae = TAESD(
                encoder_path=vae_encoder_weights,
                decoder_path=vae_decoder_weights
            )
            for param in self.vae.parameters():
                param.requires_grad = False

    # ==================== Preconditioning (Table 1 in EDM paper) ====================
    
    def c_skip(self, sigma):
        """Skip connection scaling: how much of input x to keep"""
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        """Output scaling: how much to scale network output"""
        return sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_in(self, sigma):
        """Input scaling: how much to scale network input"""
        return 1.0 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_noise(self, sigma):
        """Noise conditioning: transform sigma for network input"""
        return 0.25 * torch.log(sigma)

    def loss_weight(self, sigma):
        """Loss weighting λ(σ): balances loss across noise levels"""
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

    # ==================== Denoiser with Preconditioning ====================

    def denoise(self, x_noisy, sigma, label):
        """
        Preconditioned denoiser D(x; σ, c)
        
        D(x; σ) = c_skip(σ) · x + c_out(σ) · F_θ(c_in(σ) · x; c_noise(σ), c)
        
        This formulation ensures:
        - Network sees normalized inputs regardless of noise level
        - Network outputs are properly scaled
        - Skip connection preserves signal at low noise levels
        """
        # Reshape sigma for broadcasting: [B] -> [B, 1, 1, 1]
        sigma = sigma.view(-1, 1, 1, 1)
        
        # Compute preconditioning coefficients
        c_skip = self.c_skip(sigma)
        c_out = self.c_out(sigma)
        c_in = self.c_in(sigma)
        c_noise = self.c_noise(sigma).squeeze()  # [B] for network input
        
        # Preconditioned network forward pass
        # F_θ predicts the "denoised" direction
        F_x = self.model(c_in * x_noisy, label, c_noise)
        
        # Combine skip connection and network output
        D_x = c_skip * x_noisy + c_out * F_x
        
        return D_x

    # ==================== Training ====================

    def compute_loss(self, x_start, label, noise=None):
        """
        EDM training loss with:
        1. Log-normal noise sampling
        2. Preconditioning
        3. Loss weighting
        """
        # Encode to latent space if using VAE
        if self.use_vae:
            with torch.no_grad():
                x_start = self.vae.encoder(x_start)

        batch_size = x_start.shape[0]
        device = x_start.device

        # Sample noise levels from log-normal distribution
        # ln(σ) ~ N(P_mean, P_std²)
        log_sigma = torch.randn(batch_size, device=device) * self.P_std + self.P_mean
        sigma = torch.exp(log_sigma)
        
        # Clamp sigma to valid range
        sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)

        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_start)

        # Add noise: x_noisy = x + σ · ε
        sigma_expanded = sigma.view(-1, 1, 1, 1)
        x_noisy = x_start + sigma_expanded * noise

        # Denoise
        D_x = self.denoise(x_noisy, sigma, label)

        # Compute weighted MSE loss
        # L = λ(σ) · ||D(x + σε; σ) - x||²
        weight = self.loss_weight(sigma).view(-1, 1, 1, 1)
        loss = weight * (D_x - x_start) ** 2
        
        return loss.mean()

    # ==================== Sampling ====================

    @torch.no_grad()
    def get_sigmas(self, n_steps, device):
        """
        Generate noise schedule for sampling using the rho schedule from EDM.
        σ_i = (σ_max^(1/ρ) + i/(n-1) · (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
        """
        rho_inv = 1.0 / self.rho
        steps = torch.linspace(0, 1, n_steps, device=device)
        sigmas = (
            self.sigma_max ** rho_inv + 
            steps * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)
        ) ** self.rho
        
        # Append 0 at the end (final step)
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
        return sigmas

    @torch.no_grad()
    def heun_step(self, x, sigma, sigma_next, label):
        """
        Heun's 2nd-order method (deterministic sampler).
        
        This is a 2nd-order ODE solver that provides better accuracy
        than Euler method with minimal additional cost.
        """
        # Denoised estimate at current noise level
        denoised = self.denoise(x, sigma.expand(x.shape[0]), label)
        
        # Derivative (score direction)
        d = (x - denoised) / sigma.view(1, 1, 1, 1)
        
        # Euler step
        x_next = x + d * (sigma_next - sigma).view(1, 1, 1, 1)
        
        # Heun correction (if not at final step)
        if sigma_next > 0:
            denoised_next = self.denoise(x_next, sigma_next.expand(x.shape[0]), label)
            d_next = (x_next - denoised_next) / sigma_next.view(1, 1, 1, 1)
            
            # Average the two derivatives
            d_avg = 0.5 * (d + d_next)
            x_next = x + d_avg * (sigma_next - sigma).view(1, 1, 1, 1)
        
        return x_next

    @torch.no_grad()
    def euler_step(self, x, sigma, sigma_next, label):
        """
        Euler method (1st-order, faster but less accurate).
        """
        denoised = self.denoise(x, sigma.expand(x.shape[0]), label)
        d = (x - denoised) / sigma.view(1, 1, 1, 1)
        x_next = x + d * (sigma_next - sigma).view(1, 1, 1, 1)
        return x_next

    @torch.no_grad()
    def generate(self, labels, sampler="heun"):
        """
        Generate samples using EDM sampling.
        
        Args:
            labels: Class labels for conditional generation
            sampler: "heun" (2nd-order, better quality) or "euler" (1st-order, faster)
        """
        device = next(self.model.parameters()).device
        
        # Ensure labels are on correct device
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, device=device, dtype=torch.long)
        else:
            labels = labels.to(device)

        batch_size = len(labels)
        shape = [batch_size] + list(self.img_shape)

        # Get noise schedule
        sigmas = self.get_sigmas(self.timesteps, device)

        # Start from pure noise scaled by sigma_max
        x = torch.randn(shape, device=device) * sigmas[0]

        # Select sampler
        step_fn = self.heun_step if sampler == "heun" else self.euler_step

        # Sampling loop
        for i in range(len(sigmas) - 1):
            x = step_fn(x, sigmas[i], sigmas[i + 1], labels)

        # Decode from latent space if using VAE
        if self.use_vae:
            x = self.vae.decoder(x)

        # Post-process
        x = self.postprocess(x)
        return x

    @torch.no_grad()
    def postprocess(self, imgs):
        """Post-process generated images."""
        if self.use_vae:
            imgs.clamp_(min=0.0, max=1.0)
        else:
            imgs.clamp_(min=-1.0, max=1.0)
            imgs = (imgs + 1.0) * 0.5
        return imgs


class EDMStochastic(EDM):
    """
    EDM with stochastic sampling (adds noise during sampling).
    
    This can improve sample diversity and is closer to the original
    DDPM sampling behavior, but with EDM's improved formulation.
    """

    def __init__(self, *args, S_churn=40, S_min=0.05, S_max=50, S_noise=1.003, **kwargs):
        """
        Additional args for stochastic sampling:
            S_churn: Amount of noise to add (0 = deterministic)
            S_min: Minimum sigma for noise injection
            S_max: Maximum sigma for noise injection
            S_noise: Noise multiplier
        """
        super().__init__(*args, **kwargs)
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise

    @torch.no_grad()
    def stochastic_step(self, x, sigma, sigma_next, label):
        """
        Stochastic sampler step with noise injection.
        """
        # Determine if we should add noise at this step
        gamma = min(self.S_churn / self.timesteps, math.sqrt(2) - 1)
        if self.S_min <= sigma <= self.S_max:
            gamma = gamma
        else:
            gamma = 0

        # Add noise
        sigma_hat = sigma * (1 + gamma)
        if gamma > 0:
            eps = torch.randn_like(x) * self.S_noise
            x = x + eps * torch.sqrt(sigma_hat ** 2 - sigma ** 2).view(1, 1, 1, 1)

        # Denoising step
        denoised = self.denoise(x, sigma_hat.expand(x.shape[0]), label)
        d = (x - denoised) / sigma_hat.view(1, 1, 1, 1)
        
        # Euler step
        x_next = x + d * (sigma_next - sigma_hat).view(1, 1, 1, 1)
        
        # Heun correction
        if sigma_next > 0:
            denoised_next = self.denoise(x_next, sigma_next.expand(x.shape[0]), label)
            d_next = (x_next - denoised_next) / sigma_next.view(1, 1, 1, 1)
            d_avg = 0.5 * (d + d_next)
            x_next = x + d_avg * (sigma_next - sigma_hat).view(1, 1, 1, 1)

        return x_next

    @torch.no_grad()
    def generate(self, labels, sampler="stochastic"):
        """Generate with stochastic sampling by default."""
        if sampler == "stochastic":
            device = next(self.model.parameters()).device
            
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, device=device, dtype=torch.long)
            else:
                labels = labels.to(device)

            batch_size = len(labels)
            shape = [batch_size] + list(self.img_shape)
            sigmas = self.get_sigmas(self.timesteps, device)
            x = torch.randn(shape, device=device) * sigmas[0]

            for i in range(len(sigmas) - 1):
                x = self.stochastic_step(x, sigmas[i], sigmas[i + 1], labels)

            if self.use_vae:
                x = self.vae.decoder(x)

            x = self.postprocess(x)
            return x
        else:
            return super().generate(labels, sampler=sampler)

