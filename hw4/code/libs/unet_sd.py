"""
Stable Diffusion 1.x Compatible UNet

This is a modified version of our UNet that can load Stable Diffusion 1.x checkpoints.
It uses the same building blocks from blocks.py (which are already SD-compatible)
but with SD 1.x specific configuration.

SD 1.x Architecture:
- Base channels: 320
- Channel multipliers: (1, 2, 4, 4) -> (320, 640, 1280, 1280)
- Attention at levels 1, 2, 3 (not level 0)
- 2 ResBlocks per level
- Cross-attention with CLIP (context_dim=768)
- 4 channels in/out (VAE latent space)
"""

import torch
from torch import nn

from .utils import default
from .blocks import (ResBlock, SpatialTransformer, SinusoidalPE, Upsample, Downsample)


class UNetForSD(nn.Module):
    """
    UNet modified for Stable Diffusion 1.x checkpoint loading.
    
    Uses our existing building blocks but with SD-specific configuration.
    This allows loading official SD 1.x checkpoints with proper weight mapping.
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        dim=320,                      # SD 1.x base dimension
        context_dim=768,              # CLIP embedding dimension
        dim_mults=(1, 2, 4, 4),       # SD 1.x channel multipliers
        attn_levels=(1, 2, 3),        # Attention at these levels (not level 0)
        num_res_blocks=2,             # 2 ResBlocks per level in SD
        num_groups=32,                # SD uses 32 groups
        num_heads=8,                  # SD uses 8 attention heads
    ):
        super().__init__()

        self.in_channels = in_channels
        self.dim = dim
        
        # Calculate dimensions at each level
        dims = [dim * m for m in dim_mults]
        time_dim = dim * 4
        
        # Time embedding (sinusoidal PE + MLP)
        self.time_embed = nn.Sequential(
            SinusoidalPE(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        # Encoder blocks
        self.encoder = nn.ModuleList()
        self.encoder_channels = [dim]  # Track channels for skip connections
        
        ch = dim
        for level, mult in enumerate(dim_mults):
            out_ch = dim * mult
            
            # ResBlocks at this level
            for block_idx in range(num_res_blocks):
                block_in = ch if block_idx == 0 else out_ch
                layers = nn.ModuleDict({
                    'resblock': ResBlock(block_in, time_dim, out_ch, groups=num_groups),
                })
                
                # Add attention if at attention level
                if level in attn_levels:
                    layers['attn'] = SpatialTransformer(
                        out_ch, context_dim, num_heads=num_heads, groups=num_groups
                    )
                
                self.encoder.append(layers)
                self.encoder_channels.append(out_ch)
                ch = out_ch
            
            # Downsample (except at last level)
            if level < len(dim_mults) - 1:
                self.encoder.append(nn.ModuleDict({
                    'downsample': Downsample(ch, ch)
                }))
                self.encoder_channels.append(ch)

        # Middle block
        self.mid_block1 = ResBlock(ch, time_dim, ch, groups=num_groups)
        self.mid_attn = SpatialTransformer(ch, context_dim, num_heads=num_heads, groups=num_groups)
        self.mid_block2 = ResBlock(ch, time_dim, ch, groups=num_groups)

        # Decoder blocks
        self.decoder = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(dim_mults))):
            out_ch = dim * mult
            
            # ResBlocks at this level (including one extra for skip connection)
            for block_idx in range(num_res_blocks + 1):
                skip_ch = self.encoder_channels.pop()
                block_in = ch + skip_ch  # Concatenate skip connection
                
                layers = nn.ModuleDict({
                    'resblock': ResBlock(block_in, time_dim, out_ch, groups=num_groups),
                })
                
                if level in attn_levels:
                    layers['attn'] = SpatialTransformer(
                        out_ch, context_dim, num_heads=num_heads, groups=num_groups
                    )
                
                # Upsample at the end of each level (except last)
                if block_idx == num_res_blocks and level > 0:
                    layers['upsample'] = Upsample(out_ch, out_ch)
                
                self.decoder.append(layers)
                ch = out_ch

        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, context, timesteps):
        """
        Args:
            x: [B, 4, H, W] noisy latent
            context: [B, seq_len, 768] CLIP text embeddings
            timesteps: [B] timestep values
        """
        # Time embedding
        t_emb = self.time_embed(timesteps)
        
        # Input conv
        h = self.conv_in(x)
        
        # Encoder with skip connections
        skips = [h]
        for block in self.encoder:
            if 'resblock' in block:
                h = block['resblock'](h, t_emb)
                if 'attn' in block:
                    h = block['attn'](h, context)
            elif 'downsample' in block:
                h = block['downsample'](h)
            skips.append(h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_block2(h, t_emb)
        
        # Decoder
        for block in self.decoder:
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = block['resblock'](h, t_emb)
            if 'attn' in block:
                h = block['attn'](h, context)
            if 'upsample' in block:
                h = block['upsample'](h)
        
        return self.out(h)


def load_sd_checkpoint(model, checkpoint_path, device="cpu"):
    """
    Load Stable Diffusion checkpoint into our UNetForSD.
    
    This function maps SD checkpoint keys to our model's layer names.
    
    Args:
        model: UNetForSD instance
        checkpoint_path: Path to .ckpt or .safetensors file
        device: Device to load on
        
    Returns:
        model with loaded weights
    """
    print(f"Loading SD checkpoint from {checkpoint_path}...")
    
    # Load checkpoint
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path, device=device)
    else:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
    
    # Extract UNet weights (SD stores them under 'model.diffusion_model.')
    unet_state_dict = {}
    prefix = "model.diffusion_model."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            unet_state_dict[new_key] = value
    
    if not unet_state_dict:
        print("Warning: No UNet weights found with expected prefix")
        unet_state_dict = state_dict
    
    # Create key mapping from SD naming to our naming
    mapped_weights = map_sd_weights_to_model(unet_state_dict, model)
    
    # Load weights
    missing, unexpected = model.load_state_dict(mapped_weights, strict=False)
    
    print(f"Loaded {len(mapped_weights)} parameters")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
    
    return model


def map_sd_weights_to_model(sd_weights, model):
    """
    Map Stable Diffusion weight names to our model's naming convention.
    
    SD naming example: 
        input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight
    Our naming:
        encoder.0.attn.self_attn.qkv.weight (combined qkv)
        
    This is complex because of structural differences. We do our best to map.
    """
    model_state = model.state_dict()

    # ------------------------------------------------------------------
    # 1) Preprocess SD weights to fuse q/k/v into our combined Linear layers.
    #    SD stores attention as to_q, to_k, to_v (separate Linear); our
    #    implementation uses a single qkv Linear for self-attn and a q + kv
    #    pair for cross-attn. Fusing here reduces missing keys drastically.
    # ------------------------------------------------------------------
    fused = {}
    attn_groups = {}
    cross_groups = {}

    def _prefix(key, token):
        # split "...to_q.weight" -> "...", token="to_q"
        return key.rsplit(token, 1)[0]

    for k, v in sd_weights.items():
        if ".attn1.to_q." in k:
            pref = _prefix(k, "to_q.")
            attn_groups.setdefault(pref, {})["q" if k.endswith("weight") else "q_bias"] = v
            continue
        if ".attn1.to_k." in k:
            pref = _prefix(k, "to_k.")
            attn_groups.setdefault(pref, {})["k" if k.endswith("weight") else "k_bias"] = v
            continue
        if ".attn1.to_v." in k:
            pref = _prefix(k, "to_v.")
            attn_groups.setdefault(pref, {})["v" if k.endswith("weight") else "v_bias"] = v
            continue
        if ".attn2.to_q." in k:
            pref = _prefix(k, "to_q.")
            cross_groups.setdefault(pref, {})["q" if k.endswith("weight") else "q_bias"] = v
            continue
        if ".attn2.to_k." in k:
            pref = _prefix(k, "to_k.")
            cross_groups.setdefault(pref, {})["k" if k.endswith("weight") else "k_bias"] = v
            continue
        if ".attn2.to_v." in k:
            pref = _prefix(k, "to_v.")
            cross_groups.setdefault(pref, {})["v" if k.endswith("weight") else "v_bias"] = v
            continue
        fused[k] = v  # keep other weights

    # fuse self-attn qkv
    for pref, parts in attn_groups.items():
        if all(x in parts for x in ("q", "k", "v")):
            fused[pref + "qkv.weight"] = torch.cat(
                [parts["q"], parts["k"], parts["v"]], dim=0
            )
        # Biases (if present)
        q_b = parts.get("q_bias")
        k_b = parts.get("k_bias")
        v_b = parts.get("v_bias")
        if all(isinstance(b, torch.Tensor) and b.dim() == 1 for b in (q_b, k_b, v_b)):
            fused[pref + "qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

    # fuse cross-attn q + kv
    for pref, parts in cross_groups.items():
        if "q" in parts:
            fused[pref + "q.weight"] = parts["q"]
        if "k" in parts and "v" in parts:
            fused[pref + "kv.weight"] = torch.cat(
                [parts["k"], parts["v"]], dim=0
            )
        # bias handling (if present)
        q_b = parts.get("q_bias")
        k_b = parts.get("k_bias")
        v_b = parts.get("v_bias")
        if isinstance(q_b, torch.Tensor) and q_b.dim() == 1:
            fused[pref + "q.bias"] = q_b
        if isinstance(k_b, torch.Tensor) and isinstance(v_b, torch.Tensor):
            fused[pref + "kv.bias"] = torch.cat([k_b, v_b], dim=0)

    # ------------------------------------------------------------------
    # 2) Direct mappings for a few known keys (time embed, in/out convs)
    # ------------------------------------------------------------------
    mapped = {}
    direct_maps = {
        # Time embedding
        "time_embed.0.weight": "time_embed.0.weight",  # SinusoidalPE doesn't have weight
        "time_embed.0.bias": "time_embed.0.bias",
        "time_embed.2.weight": "time_embed.1.weight",
        "time_embed.2.bias": "time_embed.1.bias",
        # Input conv
        "input_blocks.0.0.weight": "conv_in.weight",
        "input_blocks.0.0.bias": "conv_in.bias",
        # Output
        "out.0.weight": "out.0.weight",
        "out.0.bias": "out.0.bias",
        "out.2.weight": "out.2.weight",
        "out.2.bias": "out.2.bias",
    }

    for sd_key, our_key in direct_maps.items():
        if sd_key in fused and our_key in model_state:
            if fused[sd_key].shape == model_state[our_key].shape:
                mapped[our_key] = fused[sd_key]

    # ------------------------------------------------------------------
    # 3) Shape-based matching fallback (heuristic but better after fusion)
    # ------------------------------------------------------------------
    sd_by_shape = {}
    for key, value in fused.items():
        shape = tuple(value.shape)
        sd_by_shape.setdefault(shape, []).append((key, value))

    for our_key, our_value in model_state.items():
        if our_key in mapped:
            continue
        shape = tuple(our_value.shape)
        if shape in sd_by_shape and sd_by_shape[shape]:
            sd_key, sd_value = sd_by_shape[shape].pop(0)
            mapped[our_key] = sd_value

    return mapped
