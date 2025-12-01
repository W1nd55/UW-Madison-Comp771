import torch
from torch import nn

from .utils import default
from .blocks import (ResBlock, SpatialTransformer, SinusoidalPE,
                     LabelEmbedding, Upsample, Downsample)


class UNet(nn.Module):
    """
    UNet as adopted by many diffusion models. This is the conditional version.
    Essentially, this UNet defines a function f(x, c, t), which takes
    (1) the noisy version of an image (x)
    (2) the condition (c) as the class label in our case
    (3) the time step in the diffusion proccess (t)

    Args:
        dim (int): base feature dimension in UNet. This will be multiplied
            by dim_mults across blocks.
        context_dim (int): condition dimension (embedding of the label) in UNet
        num_classes (int): number of classes used for conditioning
        in_channels (int): input channels to the UNet
        in_channels (int): output channels of the UNet
        dim_mults (tuple/list of int): multiplier of feature dimensions in UNet
            length of this list specifies #blockes in UNet encoder/decoder
            e.g., (1, 2, 4) -> 3 blocks with output dims of 1x, 2x, 4x
            w.r.t. the base feature dim
        attn_levels (tuple/list of int): specify if attention layer is included
            in a block in UNet encoder/decoder
            e.g., (0, 1) -> the first two blocks in the encoder and the last two
            blocks in the decoder will include attention layer
        init_dim (int): if specified, a different dimension will be used for
            the first conv layer
        num_groups (int): number of groups used in group norm. Will infer from
            dim if not specified.
        num_heads (int): number of attention heads in self/cross-attention. Will
            infer from dim if not specified.
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
        attn_levels=(0, 1),
        init_dim=None,
        num_groups=None,
        num_heads=None,
    ):
        super().__init__()

        # determine dimensions (input, intermediate feature dimensions)
        self.in_channels = in_channels
        init_dim = default(init_dim, dim)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # figure out all feature dims / num groups / num heads
        # dim of time embeddings
        time_dim = dim * 4
        # num_groups (8, 16, or 32)
        if num_groups is None:
            num_groups = min(max(dim // 64, 1) * 8, 32)
        # num_heads (4 or 8)
        if num_heads is None:
            num_heads = min(max(dim // 128, 1) * 4, 8)

        # mlp for embedding time steps (we will use sinusoidal PE here)
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

        # mlp for embedding labels, similar to time steps
        self.label_embd = nn.Sequential(
            LabelEmbedding(num_classes, dim),
            nn.Linear(dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )

        # initial conv layer
        self.conv_in = nn.Conv2d(
            in_channels, init_dim, kernel_size=3, padding=1
        )

        # layers for unet
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        num_resolutions = len(in_out)

        # encoder (ResBlock + Transformer + Downsampling)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = (ind >= (num_resolutions - 1))
            self.encoder.append(
                nn.ModuleList(
                    [
                        ResBlock(
                            dim_in,
                            out_channel=dim_out,
                            time_emb_dim=time_dim,
                            groups=num_groups
                        ),
                        SpatialTransformer(
                            dim_out,
                            context_dim,
                            num_heads=num_heads,
                            groups=num_groups
                        ) if ind in attn_levels
                        else None,
                        Downsample(dim_out, dim_out)
                        if not is_last
                        else None,
                    ]
                )
            )

        # middle block (ResBlock + Transformer + ResBlock)
        mid_dim = dims[-1]
        self.mid_res_block1 = ResBlock(
            mid_dim,
            time_emb_dim=time_dim,
            groups=num_groups
        )
        self.mid_attn = SpatialTransformer(
            mid_dim,
            context_dim,
            num_heads=num_heads,
            groups=num_groups
        )
        self.mid_res_block2 = ResBlock(
            mid_dim,
            time_emb_dim=time_dim,
            groups=num_groups
        )

        # decoder (ResBlock + Transformer + Upsampling)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = (ind >= (num_resolutions - 1))
            self.decoder.append(
                nn.ModuleList(
                    [
                        ResBlock(
                            # concat using unet
                            dim_out * 2,
                            out_channel=dim_in,
                            time_emb_dim=time_dim,
                            groups=num_groups
                        ),
                        SpatialTransformer(
                            dim_in,
                            context_dim,
                            num_heads=num_heads,
                            groups=num_groups
                        ) if (num_resolutions - ind - 1) in attn_levels
                        else None,
                        Upsample(dim_in, dim_in)
                        if not is_last
                        else None,
                    ]
                )
            )

        # final conv block
        self.out_channels = default(out_channels, in_channels)
        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups, dim),
            nn.SiLU(),
            nn.Conv2d(dim, self.out_channels, 1),
        )

    def forward(self, x, label, time):
        """
        Args:
            x (tensor): input image of shape B x C x H x W
            label (iterable of long): input label of size B
            time (float): input time step
        """

        # first conv
        x = self.conv_in(x)
        # time embedding
        if self.time_embed_pe:
            t = self.time_embd(time)
        else:
            t = self.time_embd(time[:, None])
        # label embedding
        c = self.label_embd(label).unsqueeze(1)
        # cache for encoder output
        encoder_output = []

        # encoder
        for resblock, transformer, downsample in self.encoder:
            x = resblock(x, t)
            if transformer:
                x = transformer(x, c)
            encoder_output.append(x)
            if downsample:
                x = downsample(x)

        # middle block
        x = self.mid_res_block1(x, t)
        x = self.mid_attn(x, c)
        x = self.mid_res_block2(x, t)

        """
        Fill in the missing code here (decoder part of UNet).
        The decoder is similar to the encoder. It applies following operations
        within a single block
        (1) a residual module
        (2) a transformer module
        (3) an upsampling module (if there is one)
        You can use the encoder part as a reference.
        """
        # decoder

        x = self.final_conv(x)
        return x
