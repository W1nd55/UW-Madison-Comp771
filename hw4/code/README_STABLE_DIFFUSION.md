# Stable Diffusion 1.x Implementation (Bonus 3)

This implementation loads Stable Diffusion 1.x checkpoints using our own code, demonstrating understanding of the SD architecture.

## Components (As Required)

### 1. Modified UNet (`libs/unet_sd.py`)
A new UNet class `UNetForSD` that:
- Uses our existing building blocks from `blocks.py` (which are already SD-compatible with GEGLU, MLP_SD, etc.)
- Matches SD 1.x architecture: 320 base channels, (1,2,4,4) multipliers, attention at levels 1,2,3
- Can load official SD checkpoint weights

### 2. CLIP Text Encoder (`libs/stable_diffusion.py`)
- Wrapper around OpenAI's CLIP from HuggingFace (`openai/clip-vit-large-patch14`)
- Outputs 768-dimensional embeddings matching SD's context_dim
- Tokenizes text to 77 tokens

### 3. Weight Loading Function (`libs/unet_sd.py`)
- `load_sd_checkpoint()`: Loads .ckpt or .safetensors files
- `map_sd_weights_to_model()`: Maps SD's weight names to our architecture

### 4. LDM Config File (`configs/stable_diffusion.yaml`)
- Complete configuration for SD 1.x
- UNet, CLIP, VAE, and diffusion settings

## Requirements

```bash
pip install transformers  # For CLIP text encoder
pip install safetensors   # Optional, for .safetensors files
```

## Usage

```bash
cd code

# Run inference with SD checkpoint
python inference_sd.py \
    --checkpoint ../sd-v1-4.ckpt \
    --prompt "a beautiful sunset over mountains" \
    --taesd_path ../pretrained/taesd_decoder.pth \
    --output sunset.png
```

## Architecture Details

Our implementation matches SD 1.x:

| Component | Configuration |
|-----------|--------------|
| **UNet** | |
| Base channels | 320 |
| Channel multipliers | (1, 2, 4, 4) |
| Attention levels | 1, 2, 3 (not 0) |
| ResBlocks per level | 2 |
| Context dim | 768 (CLIP) |
| **CLIP** | |
| Model | openai/clip-vit-large-patch14 |
| Embedding dim | 768 |
| Sequence length | 77 |
| **VAE** | |
| Latent channels | 4 |
| Scaling factor | 0.18215 |

## Code Structure

```
libs/
├── blocks.py          # Building blocks (already SD-compatible: GEGLU, MLP_SD, etc.)
├── unet_sd.py         # UNetForSD class + weight loading
├── stable_diffusion.py # Full SD pipeline with CLIP
└── tiny_autoencoder.py # TAESD decoder

configs/
└── stable_diffusion.yaml  # LDM config file
```

## How Weight Loading Works

1. Load checkpoint file (.ckpt contains state_dict with 'model.diffusion_model.' prefix)
2. Extract UNet weights by removing prefix
3. Map SD weight names to our model's names
4. Load with `strict=False` to handle any unmapped weights

## Key Implementation Details

### Building Blocks (from blocks.py)
Our `blocks.py` already contains SD-compatible components:
- `GEGLU`: Gated activation used in SD's transformer FFN
- `MLP_SD`: SD-style MLP with GEGLU
- `SpatialTransformer`: Self-attention + cross-attention + FFN
- `ResBlock`: Residual block with time conditioning

### UNetForSD
- Uses same blocks as our original UNet
- Configured for SD 1.x dimensions
- Encoder: 4 levels × 2 ResBlocks + attention (levels 1,2,3) + downsample
- Middle: ResBlock + Attention + ResBlock  
- Decoder: Mirror of encoder with upsampling

### Inference Pipeline
1. Encode text prompt with CLIP → [B, 77, 768]
2. Start from random noise in latent space → [B, 4, 64, 64]
3. DDIM sampling loop (50 steps)
4. Decode with TAESD → [B, 3, 512, 512]

## Notes

- Weight mapping is heuristic-based due to naming differences
- Uses TAESD (lightweight) instead of full SD VAE for decoding
- CLIP encoder is loaded from HuggingFace (downloads ~500MB)
