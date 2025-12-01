import math

import torch
from torch import nn
import torch.nn.functional as F

from .utils import default


class Attention(nn.Module):
    """Multi-head Self-Attention."""

    def __init__(
        self,
        dim,
        num_heads=4,
        qkv_bias=False,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x (tensor): Input feature map of size B x H x W x C
        """
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)
        return x


class CrossAttention(nn.Module):
    """Multi-head Cross Self-Attention."""

    def __init__(
        self,
        dim,
        context_dim,
        num_heads=4,
        qkv_bias=False,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context):
        """
        Args:
            x (tensor): Input feature map of size B x H x W x C (used for q)
            context (tensor): Input context feature of size B x T x C (used for k, v)
        """
        B, H, W, _ = x.shape
        B, T, _ = context.shape

        # q with shape (B, nHead, H * W, C)
        q = self.q(x).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3)
        # kv with shape (2, B, nHead, T, C)
        kv = self.kv(context).reshape(B, T, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W / T, C)
        k, v = kv.reshape(2, B * self.num_heads, T, -1).unbind(0)
        q = q.reshape(B * self.num_heads, H*W, -1)

        # attention
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)
        return x


class GEGLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x):
        x = self.proj(x)
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class MLP_SD(nn.Module):
    """Multilayer perceptron. This is the MLP used by stable diffusion."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp = nn.Sequential(
            GEGLU(in_features, hidden_features),
            nn.Dropout(0.0),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class SpatialTransformer(nn.Module):
    """
    Spatial Transformer using cross/self-attention. This block include
    one self-attention, one cross-attention, and one MLP.
    """

    def __init__(
        self,
        dim,
        context_dim,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        groups=4
    ):
        """
        Args:
            dim (int): Number of input channels.
            context_dim (int): Number of input context channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
        """
        super().__init__()

        self.group_norm = nn.GroupNorm(groups, dim)
        self.proj_in = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            context_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.mlp = MLP_SD(
            in_features=dim, hidden_features=int(dim * mlp_ratio)
        )
        self.norm3 = norm_layer(dim)

    def forward(self, x, context):
        """
        Args:
            x (tensor): Input feature map of size B x C x H x W (used for q)
            context (tensor): Input context feature of size B x T x C (used for k, v)
        """
        shortcut = x
        x = self.proj_in(self.group_norm(x))
        x = x.permute(0, 2, 3, 1)  # B C H W -> B H W C
        x = self.self_attn(self.norm1(x)) + x
        x = self.cross_attn(self.norm2(x), context) + x
        x = self.mlp(self.norm3(x)) + x
        x = x.permute(0, 3, 1, 2) # B H W C -> B C H W
        return self.proj_out(x) + shortcut


class SinusoidalPE(nn.Module):
    """
    sinusoidal position encoding for time steps as in the Transformer paper
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.max_period=10000

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class ResBlock(nn.Module):
    """A slightly altered residual block with time modulation"""

    def __init__(self, in_channel, time_emb_dim, out_channel=None, groups=4):
        """
        Args:
            in_channel (int): Number of input channels.
            time_emb_dim (int): Number of input time embedding dimensions.
            out_channel (int): Number of output channels (default: same as in_channel)
            groups (int): number of groups used in groupnorm (default: 4)
        """

        super().__init__()
        if out_channel is None:
            out_channel = in_channel
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.norm1 = nn.GroupNorm(groups, in_channel)
        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=1, padding=1
        )
        self.time_emb_proj = nn.Linear(
            in_features=time_emb_dim, out_features=out_channel, bias=True
        )
        self.norm2 = nn.GroupNorm(groups, out_channel)
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, stride=1, padding=1
        )
        self.act = nn.SiLU()
        if in_channel == out_channel:
            self.conv_shortcut = nn.Identity()
        else:
            self.conv_shortcut = nn.Conv2d(
                in_channel, out_channel, kernel_size=1, stride=1
            )

    def forward(self, x, t_emb, cond=None):
        """
        Args:
            x (tensor): Input feature map of size B x C x H x W
            t_emb (tensor): Input time embedding of size B x T x C
        """

        # pre-norm + conv1
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        # Time modulation (with shift only, similar to stable diffusion)
        if t_emb is not None:
            t_hidden = self.time_emb_proj(self.act(t_emb))
            h = h + t_hidden[:, :, None, None]
        # norm
        h = self.norm2(h)
        h = self.act(h)
        # conv2
        h = self.conv2(h)
        # Skip connection
        return h + self.conv_shortcut(x)


class Upsample(nn.Module):
    """Upsampling using nearest neighbor + conv2D"""

    def __init__(self, dim, dim_out=None, scale_factor=2):
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.conv = nn.Conv2d(dim, default(dim_out, dim), kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsampling using strided convolution"""

    def __init__(self, dim, dim_out=None, scale_factor=2):
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.conv = nn.Conv2d(
            dim, default(dim_out, dim), kernel_size=3, stride=self.scale_factor, padding=1
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class LabelEmbedding(nn.Module):
    """Simple label embedding"""

    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)

    def forward(self, label):
        return self.embed(label)
