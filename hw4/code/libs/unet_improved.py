"""
Improved U-Net for Diffusion Models

Key improvements over the baseline UNet:
1. Adaptive Group Normalization (AdaGN) - FiLM-style scale+shift conditioning
2. Multiple ResBlocks per resolution level (configurable depth)
3. Attention at all resolution levels (optional)
4. Dropout for regularization
5. Skip connection scaling (1/sqrt(2)) for better gradient flow
6. Zero-initialization for output layers (training stability)
7. Enhanced middle block with more capacity

References:
- EDM: Karras et al. "Elucidating the Design Space of Diffusion-Based Generative Models"
- DiT: Peebles & Xie "Scalable Diffusion Models with Transformers"
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

from .utils import default
from .blocks import (SpatialTransformer, SinusoidalPE,
                     LabelEmbedding, Upsample, Downsample)


class AdaGroupNorm(nn.Module):
    """
    Adaptive Group Normalization with FiLM-style conditioning.
    Applies both scale and shift from the time embedding, not just shift.
    This provides stronger conditioning compared to simple addition.
    """

    def __init__(self, num_channels, time_emb_dim, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        # Project time embedding to scale and shift parameters
        self.proj = nn.Linear(time_emb_dim, num_channels * 2)

    def forward(self, x, t_emb):
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
            t_emb: Time embedding of shape [B, T]
        Returns:
            Normalized and modulated tensor
        """
        # Normalize
        h = self.norm(x)

        # Get scale and shift from time embedding
        t_out = self.proj(F.silu(t_emb))
        scale, shift = t_out.chunk(2, dim=-1)

        # Apply FiLM: scale * norm(x) + shift
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return h


class ImprovedResBlock(nn.Module):
    """
    Improved Residual Block with:
    - Adaptive Group Normalization (FiLM conditioning)
    - Optional dropout for regularization
    - Skip connection scaling for better gradient flow
    """

    def __init__(
        self,
        in_channel,
        time_emb_dim,
        out_channel=None,
        groups=32,
        dropout=0.0,
        skip_scale=1.0,
    ):
        super().__init__()
        out_channel = default(out_channel, in_channel)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.skip_scale = skip_scale

        # First conv block with AdaGN
        self.norm1 = AdaGroupNorm(in_channel, time_emb_dim, groups)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

        # Second conv block with AdaGN
        self.norm2 = AdaGroupNorm(out_channel, time_emb_dim, groups)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)

        # Skip connection
        if in_channel != out_channel:
            self.skip_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

        # Zero-initialize the final conv for residual learning
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, t_emb):
        """
        Args:
            x: Input tensor [B, C, H, W]
            t_emb: Time embedding [B, T]
        """
        h = self.norm1(x, t_emb)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h, t_emb)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Skip connection with optional scaling
        return (h + self.skip_conv(x)) * self.skip_scale


class ImprovedSpatialTransformer(nn.Module):
    """
    Enhanced Spatial Transformer with:
    - Multiple attention layers (stacked transformer blocks)
    - Dropout for regularization
    - Pre-norm architecture
    """

    def __init__(
        self,
        dim,
        context_dim,
        num_heads=8,
        depth=1,
        mlp_ratio=4.0,
        dropout=0.0,
        groups=32,
    ):
        super().__init__()
        self.depth = depth

        self.group_norm = nn.GroupNorm(groups, dim)
        self.proj_in = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1)

        # Stack multiple transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                context_dim=context_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # Zero-init output projection
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, context):
        """
        Args:
            x: Input [B, C, H, W]
            context: Conditioning [B, T, C]
        """
        shortcut = x
        x = self.proj_in(self.group_norm(x))

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, HW, C]

        for block in self.transformer_blocks:
            x = block(x, context)

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        return self.proj_out(x) + shortcut


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention, cross-attention, and MLP."""

    def __init__(
        self,
        dim,
        context_dim,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
            kdim=context_dim, vdim=context_dim
        )

        self.norm3 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context):
        """
        Args:
            x: Input [B, N, C]
            context: Conditioning [B, T, C_ctx]
        """
        # Self-attention
        x_norm = self.norm1(x)
        x = x + self.self_attn(x_norm, x_norm, x_norm, need_weights=False)[0]

        # Cross-attention
        x_norm = self.norm2(x)
        x = x + self.cross_attn(x_norm, context, context, need_weights=False)[0]

        # MLP
        x = x + self.mlp(self.norm3(x))
        return x


class ImprovedUNet(nn.Module):
    """
    Improved U-Net for diffusion models with enhanced architecture.

    Key improvements:
    - AdaGN (Adaptive Group Normalization) with scale+shift
    - Multiple ResBlocks per resolution (configurable via num_res_blocks)
    - Optional attention at all levels
    - Dropout regularization
    - Skip scaling for gradient stability
    - Enhanced middle block

    Args:
        dim: Base feature dimension
        context_dim: Condition embedding dimension
        num_classes: Number of classes for conditioning
        in_channels: Input image channels
        out_channels: Output channels (default: same as in_channels)
        dim_mults: Channel multipliers for each resolution level
        attn_levels: Which levels to apply attention (or 'all')
        num_res_blocks: Number of ResBlocks per resolution level
        num_heads: Number of attention heads
        num_groups: Groups for GroupNorm
        dropout: Dropout rate
        attn_depth: Number of transformer blocks in each attention layer
        use_skip_scale: Whether to scale skip connections by 1/sqrt(2)
    """

    def __init__(
        self,
        dim,
        context_dim,
        num_classes,
        time_embed_pe=True,
        in_channels=3,
        out_channels=None,
        dim_mults=(1, 2, 4),
        attn_levels=(0, 1, 2),
        num_res_blocks=2,
        init_dim=None,
        num_groups=None,
        num_heads=None,
        dropout=0.1,
        attn_depth=1,
        use_skip_scale=True,
    ):
        super().__init__()

        # Dimension setup
        self.in_channels = in_channels
        init_dim = default(init_dim, dim)
        dims = [init_dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        # Auto-configure groups and heads
        time_dim = dim * 4
        num_groups = default(num_groups, min(max(dim // 4, 8), 32))
        num_heads = default(num_heads, max(dim // 32, 4))

        # Skip scale for residual connections
        skip_scale = 1.0 / math.sqrt(2) if use_skip_scale else 1.0

        # Time embedding
        self.time_embed_pe = time_embed_pe
        if time_embed_pe:
            self.time_embd = nn.Sequential(
                SinusoidalPE(dim),
                nn.Linear(dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            self.time_embd = nn.Sequential(
                nn.Linear(1, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )

        # Label embedding
        self.label_embd = nn.Sequential(
            LabelEmbedding(num_classes, dim),
            nn.Linear(dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, init_dim, kernel_size=3, padding=1)

        # Encoder
        self.encoder = nn.ModuleList()
        for level, (dim_in, dim_out) in enumerate(in_out):
            is_last = (level >= num_resolutions - 1)
            use_attn = level in attn_levels or attn_levels == 'all'

            level_blocks = nn.ModuleList()
            for block_idx in range(num_res_blocks):
                # First block handles channel change
                block_in = dim_in if block_idx == 0 else dim_out
                level_blocks.append(
                    ImprovedResBlock(
                        block_in,
                        time_emb_dim=time_dim,
                        out_channel=dim_out,
                        groups=num_groups,
                        dropout=dropout,
                        skip_scale=skip_scale,
                    )
                )
                # Add attention after each resblock if enabled
                if use_attn:
                    level_blocks.append(
                        ImprovedSpatialTransformer(
                            dim_out,
                            context_dim,
                            num_heads=num_heads,
                            depth=attn_depth,
                            dropout=dropout,
                            groups=num_groups,
                        )
                    )

            # Downsampling (except last level)
            downsample = Downsample(dim_out, dim_out) if not is_last else None

            self.encoder.append(nn.ModuleDict({
                'blocks': level_blocks,
                'downsample': downsample,
            }))

        # Middle block (bottleneck)
        mid_dim = dims[-1]
        self.mid_block = nn.ModuleList([
            ImprovedResBlock(mid_dim, time_dim, groups=num_groups, dropout=dropout, skip_scale=skip_scale),
            ImprovedSpatialTransformer(mid_dim, context_dim, num_heads=num_heads, depth=attn_depth, groups=num_groups),
            ImprovedResBlock(mid_dim, time_dim, groups=num_groups, dropout=dropout, skip_scale=skip_scale),
            ImprovedSpatialTransformer(mid_dim, context_dim, num_heads=num_heads, depth=attn_depth, groups=num_groups),
            ImprovedResBlock(mid_dim, time_dim, groups=num_groups, dropout=dropout, skip_scale=skip_scale),
        ])

        # Decoder
        self.decoder = nn.ModuleList()
        for level, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = (level >= num_resolutions - 1)
            orig_level = num_resolutions - level - 1
            use_attn = orig_level in attn_levels or attn_levels == 'all'

            level_blocks = nn.ModuleList()
            for block_idx in range(num_res_blocks):
                # First block handles skip connection (2x channels)
                if block_idx == 0:
                    block_in = dim_out * 2  # concatenated skip
                else:
                    block_in = dim_in

                level_blocks.append(
                    ImprovedResBlock(
                        block_in,
                        time_emb_dim=time_dim,
                        out_channel=dim_in,
                        groups=num_groups,
                        dropout=dropout,
                        skip_scale=skip_scale,
                    )
                )
                if use_attn:
                    level_blocks.append(
                        ImprovedSpatialTransformer(
                            dim_in,
                            context_dim,
                            num_heads=num_heads,
                            depth=attn_depth,
                            dropout=dropout,
                            groups=num_groups,
                        )
                    )

            # Upsampling (except last level)
            upsample = Upsample(dim_in, dim_in) if not is_last else None

            self.decoder.append(nn.ModuleDict({
                'blocks': level_blocks,
                'upsample': upsample,
            }))

        # Output
        self.out_channels = default(out_channels, in_channels)
        self.final_norm = nn.GroupNorm(num_groups, dim)
        self.final_conv = nn.Conv2d(dim, self.out_channels, kernel_size=3, padding=1)

        # Zero-init final conv
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x, label, time):
        """
        Args:
            x: Input image [B, C, H, W]
            label: Class labels [B]
            time: Timestep [B]
        """
        # Embeddings
        if self.time_embed_pe:
            t = self.time_embd(time)
        else:
            t = self.time_embd(time[:, None])
        c = self.label_embd(label).unsqueeze(1)  # [B, 1, context_dim]

        # Initial conv
        x = self.conv_in(x)

        # Encoder with skip connections
        skips = []
        for level in self.encoder:
            for block in level['blocks']:
                if isinstance(block, ImprovedResBlock):
                    x = block(x, t)
                else:  # Transformer
                    x = block(x, c)
            skips.append(x)
            if level['downsample'] is not None:
                x = level['downsample'](x)

        # Middle block
        for block in self.mid_block:
            if isinstance(block, ImprovedResBlock):
                x = block(x, t)
            else:  # Transformer
                x = block(x, c)

        # Decoder with skip connections
        for level in self.decoder:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)

            first_block = True
            for block in level['blocks']:
                if isinstance(block, ImprovedResBlock):
                    x = block(x, t)
                    first_block = False
                else:  # Transformer
                    x = block(x, c)

            if level['upsample'] is not None:
                x = level['upsample'](x)

        # Output
        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)
        return x

