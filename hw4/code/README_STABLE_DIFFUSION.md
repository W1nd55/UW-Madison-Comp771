# Stable Diffusion 1.x Implementation (Bonus 3)

This directory contains an implementation that can load and run Stable Diffusion 1.x checkpoints.

## Components

### 1. UNet (`libs/unet_sd.py`)
- SD 1.x compatible architecture
- 320 base channels with multipliers (1, 2, 4, 4)
- Cross-attention for text conditioning
- Compatible with official SD checkpoint weight structure

### 2. CLIP Text Encoder (`libs/stable_diffusion.py`)
- Wrapper for HuggingFace `transformers` CLIP model
- Uses `openai/clip-vit-large-patch14` (768-dim embeddings)
- Tokenizes and encodes text prompts

### 3. VAE Decoder (`libs/stable_diffusion.py`)
- Decodes latent space to pixel space
- Can use lightweight TAESD (included in `pretrained/`) for faster inference

### 4. Stable Diffusion Pipeline (`libs/stable_diffusion.py`)
- Complete text-to-image pipeline
- DDIM-style sampling
- Classifier-free guidance support

## Requirements

```bash
pip install transformers  # For CLIP
pip install safetensors   # Optional, for .safetensors checkpoints
```

## Download Checkpoints

Download SD 1.4 or 1.5 checkpoints:

```bash
# Option 1: SD 1.4 from HuggingFace
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt

# Option 2: SD 1.5 from HuggingFace  
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
```

## Usage

### Command Line

```bash
cd code

# Basic usage
python inference_sd.py \
    --checkpoint path/to/sd-v1-4.ckpt \
    --prompt "a photo of an astronaut riding a horse on the moon" \
    --output astronaut.png

# With more options
python inference_sd.py \
    --checkpoint path/to/sd-v1-4.ckpt \
    --prompt "a beautiful sunset over mountains, oil painting" \
    --negative_prompt "blurry, bad quality" \
    --steps 50 \
    --guidance_scale 7.5 \
    --seed 42 \
    --output sunset.png

# Using TAESD decoder (faster, uses included pretrained weights)
python inference_sd.py \
    --checkpoint path/to/sd-v1-4.ckpt \
    --prompt "a cute cat" \
    --use_taesd \
    --taesd_path ../pretrained/taesd_decoder.pth \
    --output cat.png
```

### Python API

```python
from libs.stable_diffusion import StableDiffusion

# Load model
sd = StableDiffusion.from_pretrained(
    "path/to/sd-v1-4.ckpt",
    device="cuda",
    use_taesd=True,  # Use lightweight decoder
    taesd_path="../pretrained/taesd_decoder.pth"
)

# Generate
images = sd.generate(
    prompt="a beautiful landscape",
    negative_prompt="blurry",
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
)

# Save (images is a tensor [B, 3, H, W] in range [0, 1])
from torchvision.utils import save_image
save_image(images, "output.png")
```

## Architecture Details

### SD 1.x UNet Specifications:
- **Input/Output**: 4 channels (VAE latent space)
- **Base channels**: 320
- **Channel multipliers**: (1, 2, 4, 4) → (320, 640, 1280, 1280)
- **Attention resolutions**: At downsampling levels 1, 2, 4 (32x32, 16x16, 8x8 for 64x64 latent)
- **ResBlocks per level**: 2
- **Context dim**: 768 (CLIP ViT-L/14)
- **Attention heads**: 8

### Sampling:
- DDIM deterministic sampling
- Classifier-free guidance with configurable scale
- 50 steps default (can be reduced to 20-30 for faster generation)

## Notes

1. **Memory**: SD 1.x requires ~4GB VRAM for inference at 512x512
2. **Speed**: ~10-20 seconds per image on modern GPU with 50 steps
3. **TAESD**: The lightweight decoder is much faster but slightly lower quality
4. **Weight Loading**: The implementation includes weight mapping functions to handle differences between our architecture and SD's naming conventions

## Troubleshooting

### "transformers not found"
```bash
pip install transformers
```

### "Out of memory"
- Use `--use_taesd` for smaller decoder
- Reduce image size: `--height 256 --width 256`
- Use CPU (slow): `--device cpu`

### "Shape mismatch" warnings
This is expected if weight names don't perfectly match. The model will still work with partial loading.

