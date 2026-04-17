# Z-Image Training Settings

*Last researched: April 17, 2026. Sources: 5 threads/docs/posts reviewed.*
*Confidence: MEDIUM — highly specific methodology required.*

## Summary

- **Model Choice**: Z-Image Base is highly preferred. Z-Image Turbo has known numerical instabilities during LoRA extraction.
- **Optimizer Params**: `--block_swap_optimizer_patch_params` is mandatory for 24GB cards to prevent NaN loss.
- **Format**: Requires image-only datasets. Attempting to pass `.mp4` into Z-Image configs will silently fail during preprocessing.

## Detailed findings

### Base vs Turbo
Z-Image Base provides incredibly high aesthetic baselines. The Turbo model uses 4-step distillation, which means fine-tuning it with a LoRA often disrupts the delicate ODE trajectory. Stick to Base for all aesthetic/character LoRAs.

### Hardware & VRAM
Z-Image is highly memory intensive due to the dense attention layers.
- For 24GB VRAM: You MUST use `--block_swap_optimizer_patch_params`. If you forget this flag, the GPU runs out of memory on step 3.
- **Mandatory Flags**: You must specify `--network_module networks.lora_zimage`.

### Fine-Tuning vs LoRA
- For style transferring, LoRA rank 128 is highly effective.
- For hyper-specific character consistency, many users advocate for full fine-tuning (abandoning LoRA entirely) because Z-Image natively resists identity drift much better than FLUX.

## Contradictions and uncertainties

- The `convert_lora.py` flow into ComfyUI currently has a sizing bug on Z-Image blocks. The community fix involves modifying the dimensions manually via a python script before loading it into Nunchaku.

## Raw source links

- https://github.com/kohya-ss/musubi-tuner/blob/main/docs/zimage.md
- Tongyi-MAI Z-Image release threads
