# Implementation of flow matching described in https://arxiv.org/abs/2209.03003

import torch
from torch import nn
import torch.nn.functional as F

from .unet import UNet
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

        # return loss

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
