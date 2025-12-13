# Implementation of flow matching described in https://arxiv.org/abs/2209.03003

import torch
from torch import nn
import torch.nn.functional as F

from .unet import UNet
from .unet_improved import ImprovedUNet
from .tiny_autoencoder import TAESD


class FM(nn.Module):
    """Flow Matching Model"""

    def __init__(
        self,
        img_shape=(3, 32, 32),
        timesteps=100,
        dim=64,
        context_dim=64,
        num_classes=10,
        dim_mults=(1, 2, 4),
        attn_levels=(0, 1),
        use_vae=False,
        vae_encoder_weights=None,
        vae_decoder_weights=None,
        # Improved UNet options
        unet_type="standard",  # "standard" or "improved"
        num_res_blocks=2,
        dropout=0.1,
        attn_depth=1,
        use_skip_scale=True,
    ):
        """
        Args:
            img_shape (tuple/list of int): shape of input image or diffusion
                latent space (C x H x W)
            timesteps (int): number of timesteps in flow matching
            dim (int): base feature dimension in UNet
            context_dim (int): condition dimension (embedding of the label) in UNet
            num_classes (int): number of classes used for conditioning
            dim_mults (tuple/list of int): multiplier of feature dimensions in UNet
                length of this list specifies #blockes in UNet encoder/decoder
                e.g., (1, 2, 4) -> 3 blocks with output dims of 1x, 2x, 4x
                w.r.t. the base feature dim
            attn_levels (tuple/list of int): specify if attention layer is included
                in a block in UNet encoder/decoder
                e.g., (0, 1) -> the first two blocks in the encoder and the last two
            use_vae (bool): if a VAE is used before DDPM (thus latent diffusion)
            vae_encoder_weights (str): path to pre-trained VAE encoder weights
            vae_decoder_weights (str): path to pre-trained VAE encoder weights
        """

        super().__init__()

        assert len(img_shape) == 3
        self.timesteps = timesteps
        self.dt = 1.0 / timesteps

        # if VAE is considered, input / output dim will be the dim of latent
        self.use_vae = use_vae
        if use_vae:
            in_channels = out_channels = 4
            self.img_shape = [4, img_shape[1]//8, img_shape[2]//8]
        else:
            in_channels = out_channels = img_shape[0]
            self.img_shape = img_shape

        # the denoising model using UNet (conditioned on input label)
        if unet_type == "improved":
            self.model = ImprovedUNet(
                dim,
                context_dim,
                num_classes,
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
                in_channels=in_channels,
                out_channels=out_channels,
                dim_mults=dim_mults,
                attn_levels=attn_levels
            )

        # if we should consider latent DDPM
        if use_vae:
            assert vae_encoder_weights is not None
            assert vae_decoder_weights is not None
            self.vae = TAESD(
                encoder_path=vae_encoder_weights,
                decoder_path=vae_decoder_weights
            )
            # freeze the encoder / decoder
            for param in self.vae.parameters():
                param.requires_grad = False

    # compute the simplified loss
    def compute_loss(self, x_start, label, noise=None):
        """
        Compute the rectified flow loss for training the model.
        Note: t in the range of [0, 1]
        """

        """
        Fill in the missing code here. See Algorithm 1 (training) in the
        Rectified Flow paper. Similarly, for latent FMs, an additional encoding
        step will be needed.
        """
        # 1. optional encoding of the input image
        # 2. sample t from U(0, 1), or alternatively from a logit-normal
        #   distribution. See https://arxiv.org/abs/2403.03206
        # 3. sample probability path by generating noise and mix it with input
        # 4. matching the flow by MSE loss

        # 1. optional encoding of the input image (latent FM)
        if self.use_vae:
            # 对 AFHQ 等 latent FM 情况，先用 VAE 编码到 latent 空间
            with torch.no_grad():
                x_start = self.vae.encoder(x_start)

        # 2. sample t from U(0, 1)
        B = x_start.shape[0]
        device = x_start.device
        # t ~ Uniform(0, 1)，形状 [B]
        t = torch.rand(B, device=device)

        # 3. sample probability path: mix clean image with Gaussian noise
        # noise 在这里扮演 x_0（起点，高斯噪声）
        if noise is None:
            x0 = torch.randn_like(x_start)
        else:
            x0 = noise

        # 为了和图像维度广播，将 t reshape 成 [B, 1, 1, 1, ...]
        t_broadcast = t
        while t_broadcast.ndim < x_start.ndim:
            t_broadcast = t_broadcast.view(-1, *([1] * (x_start.ndim - 1)))

        # 直线路径插值：x_t = (1 - t) * x0 + t * x1
        x_t = (1.0 - t_broadcast) * x0 + t_broadcast * x_start

        # 真实速度场：v(x_t, t) = x1 - x0（沿直线恒定）
        v_target = x_start - x0

        # 4. matching the flow by MSE loss
        # UNet 期望的 time 输入是 [B] 的连续标量（0~1）
        v_pred = self.model(x_t, label, t)

        loss = F.mse_loss(v_pred, v_target)
        return loss

    @torch.no_grad()
    def generate(self, labels):
        """
        Sampling from Rectified Flow. This is same as Euler method
        """
        device = next(self.model.parameters()).device
        # shape of the results
        shape = [len(labels)] + list(self.img_shape)
        # start from pure noise (for each example in the batch)
        imgs = torch.randn(shape, device=device)

        """
        Fill in the missing code here. See Algorithm 1 (sampling) in the
        Rectified Flow paper. Similarly, for latent FMs, an additional
        decoding step will be needed.
        """
        # 1. sample dense time steps on the trajectory (t:0->1)
        # 2. draw images by following forward trajectory predicted by learned model
        # 3. optional decoding step

        # 确保 labels 在同一 device 上
        labels = labels.to(device)

        # 1. sample dense time steps on the trajectory (t: 0 -> 1)
        # 2. follow forward trajectory using Euler integration
        for step in range(self.timesteps):
            # 当前时间标量 t ∈ [0,1]
            t_scalar = step * self.dt
            t = torch.full((imgs.shape[0],), t_scalar, device=device)

            # 预测当前速度场 v(x_t, t)
            v = self.model(imgs, labels, t)

            # Euler 方法：x_{t+Δt} = x_t + Δt * fθ(x_t, t)
            imgs = imgs + self.dt * v

        # 3. optional decoding step for latent FM
        if self.use_vae:
            imgs = self.vae.decoder(imgs)

        # postprocessing the images
        imgs = self.postprocess(imgs)
        return imgs

    @torch.no_grad()
    def postprocess(self, imgs):
        """
        Postprocess the sampled images (e.g., normalization / clipping)
        """
        if self.use_vae:
            # already in range, clip the pixel values
            imgs.clamp_(min=0.0, max=1.0)
        else:
            # clip the pixel values within range
            imgs.clamp_(min=-1.0, max=1.0)
            # mapping to range [0, 1]
            imgs = (imgs + 1.0) * 0.5
        return imgs
