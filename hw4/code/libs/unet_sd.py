"""
Stable Diffusion 1.x Compatible UNet

This UNet is designed to be weight-compatible with Stable Diffusion 1.x checkpoints.
The layer naming and architecture match the original CompVis/Stability AI implementation.

SD 1.x Architecture:
- Base channels: 320
- Channel multipliers: (1, 2, 4, 4) -> (320, 640, 1280, 1280)
- Attention at levels 1, 2, 3 (resolutions 32, 16, 8 for 64x64 latent)
- 2 ResBlocks per level
- Cross-attention with CLIP text embeddings (context_dim=768)
- Latent space: 4 channels

Reference: https://github.com/CompVis/stable-diffusion
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embeddings as used in SD."""
    
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * 
            torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class TimestepEmbedSequential(nn.Sequential):
    """Sequential module that passes timestep embeddings to children."""
    
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block with timestep conditioning.
    Compatible with SD checkpoint naming.
    """
    
    def __init__(self, channels, emb_channels, out_channels=None, groups=32):
        super().__init__()
        out_channels = out_channels or channels
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels),
        )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        
        if channels != out_channels:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)[:, :, None, None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class CrossAttention(nn.Module):
    """Cross-attention layer for text conditioning."""
    
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        context_dim = context_dim or query_dim
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
        )

    def forward(self, x, context=None):
        if context is None:
            context = x
            
        B, N, C = x.shape
        _, S, _ = context.shape  # S = context sequence length
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Compute dim_head from output size
        dim_head = q.shape[-1] // self.heads
        
        # Reshape for multi-head attention: [B, seq, heads, dim_head] -> [B, heads, seq, dim_head]
        q = q.view(B, N, self.heads, dim_head).transpose(1, 2)
        k = k.view(B, S, self.heads, dim_head).transpose(1, 2)
        v = v.view(B, S, self.heads, dim_head).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)


class GEGLU(nn.Module):
    """GEGLU activation as used in SD."""
    
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """Feed-forward network with GEGLU."""
    
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            GEGLU(dim, inner_dim),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    """Transformer block with self-attention, cross-attention, and FFN."""
    
    def __init__(self, dim, n_heads, d_head, context_dim=None):
        super().__init__()
        self.attn1 = CrossAttention(dim, dim, heads=n_heads, dim_head=d_head)  # Self-attn
        self.attn2 = CrossAttention(dim, context_dim, heads=n_heads, dim_head=d_head)  # Cross-attn
        self.ff = FeedForward(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """Spatial transformer block for cross-attention with text."""
    
    def __init__(self, in_channels, n_heads, d_head, depth=1, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        
        self.norm = nn.GroupNorm(32, in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, 1)
        
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, n_heads, d_head, context_dim=context_dim)
            for _ in range(depth)
        ])
        
        self.proj_out = nn.Conv2d(inner_dim, in_channels, 1)

    def forward(self, x, context=None):
        B, C, H, W = x.shape
        x_in = x
        
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        
        for block in self.transformer_blocks:
            x = block(x, context=context)
        
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.proj_out(x)
        
        return x + x_in


class Upsample(nn.Module):
    """Upsampling with conv."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    """Downsampling with strided conv."""
    
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class UNetSD(nn.Module):
    """
    Stable Diffusion 1.x compatible UNet.
    
    This architecture matches the original SD 1.x UNet for weight loading.
    """
    
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        model_channels=320,
        attention_resolutions=(4, 2, 1),  # At which downsampling levels to use attention
        num_res_blocks=2,
        channel_mult=(1, 2, 4, 4),
        num_heads=8,
        context_dim=768,  # CLIP embedding dimension
        use_spatial_transformer=True,
        transformer_depth=1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        
        time_embed_dim = model_channels * 4
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input blocks
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))
        ])
        
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, mult * model_channels)]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        SpatialTransformer(
                            ch, num_heads, dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim
                        )
                    )
                
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch)))
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle block
        dim_head = ch // num_heads
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim),
            SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim),
            ResBlock(ch, time_embed_dim),
        )
        
        # Output blocks
        self.output_blocks = nn.ModuleList([])
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, mult * model_channels)]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        SpatialTransformer(
                            ch, num_heads, dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim
                        )
                    )
                
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        
        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, context):
        """
        Args:
            x: [B, 4, H, W] latent input
            timesteps: [B] timestep values
            context: [B, seq_len, 768] CLIP text embeddings
        """
        # Time embedding
        emb = self.time_embed(timesteps)
        
        # Input blocks with skip connections
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        
        # Middle block
        h = self.middle_block(h, emb, context)
        
        # Output blocks
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        
        return self.out(h)

