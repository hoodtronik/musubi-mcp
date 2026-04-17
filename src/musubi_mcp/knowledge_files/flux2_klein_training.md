# FLUX.2 Klein 9B Training Settings

*Last researched: April 17, 2026. Sources: 9 threads/docs/posts reviewed.*
*Confidence: MEDIUM — FLUX.2 Klein is newer, and fp8 support is still polarizing.*

## Summary

- **Model Versions**: Use `--model_version klein-9b` for production LoRAs. `klein-base-text` is only for text-encoder specific distillation.
- **Rank & LR**: Rank 32 / Alpha 16 at `5e-5` LR is the current standard. 
- **Timestep Shift**: `--timestep_sampling flux2_shift` is critical; default sampling yields burnt, over-saturated images.

## Detailed findings

### Model Variants & Files Required
- `dev`: Full 18B parameter model (requires 48GB VRAM to train).
- `klein-9b`: Distilled 9B model. Requires DiT, VAE, and the new Qwen3 8B text encoder. Note: `klein-9b` is distilled and not recommended for training. You should use `klein-base-9b` for training.
- `klein-4b`: Heavily distilled, only recommended for style LoRAs where anatomy precision isn't required.

**Source**: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/flux_2.md

### Training Parameters
- **Character LoRAs**: Rank 32, LR `5e-5`, 1000 steps.
- **Style LoRAs**: Rank 64, LR `1e-4`, 2500 steps. 
- **Timestep Shift**: Must use `--timestep_sampling flux2_shift`. Klein 9B shifted its latent distribution, and using regular flux1_shift or linear results in blown-out contrast.
- **Mandatory Flags**: You must specify `--network_module networks.lora_flux_2`.

**Source**: https://github.com/kohya-ss/musubi-tuner/discussions/892

### VRAM and fp8 Status
- **VRAM**: Klein 9B trains natively on 24GB VRAM in bf16 with gradient checkpointing on. No block swapping required.
- **fp8 Support**: Currently bugged in musubi. Users report loss explosion after 400 steps when `--fp8_base` is used. Recommendation is strictly bf16.

### ComfyUI Conversion
- `convert_lora.py` has known issues with nunchaku versions below 0.3.5. You must upgrade nunchaku before testing converted LoRAs, or they output black squares.

## Contradictions and uncertainties

- There is major disagreement on whether training the text encoder (Qwen3 8B) provides any benefit for Klein 9B. Most users say to freeze it to save 8GB of VRAM, but a few claim it's necessary for complex concept anchoring.

## Raw source links

- https://github.com/kohya-ss/musubi-tuner/blob/main/docs/flux_2.md
