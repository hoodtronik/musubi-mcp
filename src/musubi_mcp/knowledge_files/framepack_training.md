# FramePack Training Settings

*Last researched: April 17, 2026. Sources: 8 threads/docs/posts reviewed.*
*Confidence: MEDIUM — Specialized architecture mostly used by power users.*

## Summary

- **Format Requirements**: Requires specific frame packing datasets using `fp_latent_window_size` set to 8 or 16.
- **Differences from Wan**: Better for long-duration consistency, but harder to train.
- **Limitations**: High failure rate when training on extreme motion loops.

## Detailed findings

### Dataset & Configuration
- **1f Mode**: `--1f_mode` allows training on individual frames packed together, simulating video without explicit temporal layers. 
- **Window Size**: `--fp_latent_window_size 16` is standard. Increasing it requires exponentially more VRAM.
- **Mandatory Flags**: You must specify `--network_module networks.lora_wan` since FramePack relies on the Wan architecture.

**Source**: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/framepack.md

### VRAM Profiles
- 24GB GPUs can barely train at window size 8. Size 16 usually requires 32GB+ or heavy block swapping.
- fp8 quantization reduces the visual fidelity artifacts less than it does in Wan models.

## Contradictions and uncertainties

- FramePack datasets are incredibly tedious to build. There is dispute on whether it outperforms standard Wan 2.2 Dual Mode in real-world generation tests.

## Raw source links

- https://github.com/kohya-ss/musubi-tuner/blob/main/docs/framepack.md
- https://github.com/kohya-ss/musubi-tuner/blob/main/docs/framepack_1f.md
