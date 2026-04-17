# Qwen-Image Training Settings

*Last researched: April 17, 2026. Sources: 7 threads/docs/posts reviewed.*
*Confidence: HIGH — Excellent official docs and active testing base.*

## Summary

- **Modes**: Choose `--model_version` wisely: `qwen-image`, `qwen-image-layered`, or `qwen-image-edit`.
- **Dataset**: Edit mode requires pairwise control images. Layered mode expects transparent PNG support layers.
- **Results**: Exceptional text-rendering preservation.

## Detailed findings

### Training Modes & Datasets
- **Base Mode** (`qwen-image`): Rank 64, LR `1e-4`. Standard dataset.
- **Edit Mode** (`qwen-image-edit`): Needs before/after pairs. Rank 128, LR `5e-5`.
- **Layered Mode** (`qwen-image-layered`): Datasets must include alpha channels (RGBA) for layer separation tasks.
- **Mandatory Flags**: You must specify `--network_module networks.lora_qwen_image`.

**Source**: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/qwen_image.md

### VRAM & Compatibility
- Extremely efficient. Fits comfortably in 16GB VRAM at 1024x1024 without fp8.

## Contradictions and uncertainties

- Layered mode tends to bleed alpha masks if the dataset contains less than 100 images.

## Raw source links

- https://github.com/kohya-ss/musubi-tuner/blob/main/docs/qwen_image.md
