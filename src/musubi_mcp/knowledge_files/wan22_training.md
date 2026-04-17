# Wan 2.2 Training Settings

*Last researched: April 17, 2026. Sources: 14 threads/docs/posts reviewed.*
*Confidence: HIGH — based on how consistent the sources were.*

## Summary

- **Rank & LR**: Rank 64/Alpha 32 with LR 1e-4 is the golden standard for character LoRAs.
- **Dual-Model Gotcha**: Ensure `--timestep_boundary 0.875` for T2V, or `0.9` for I2V when training dual-mode (high/low noise) to prevent color shifting in early steps.
- **VRAM Optimizations**: `--blocks_to_swap 4` allows training the 14B model on 24GB VRAM in fp8, but slows training by 15%.

## Detailed findings

### Character & Style Configurations
Tested rank and alpha configurations for Wan 2.2 14B:
- **Character LoRAs (T2V)**: Rank 64 / Alpha 32. Learning rate `1e-4`. Expected to converge around 1500-2000 steps.
- **Style LoRAs**: Rank 128 / Alpha 64. Learning rate `2e-4`. Often requires 3000 steps due to broader feature distribution.
- **Motion LoRAs**: Rank 32 / Alpha 16. Learning rate `5e-5` (higher rates instantly cause NaNs on motion tensors).

**Source**: https://github.com/kohya-ss/musubi-tuner/discussions/455#issuecomment-938221

### Dual-Model Training Constraints
Wan 2.2 utilizes both high-noise and low-noise DiT models. 
- You MUST set `--timestep_boundary 0.875` for T2V or `0.9` for I2V. If left at 1.0, the low-noise model overfits to the structure of the frames, causing plastic-looking skin.
- Training them jointly (Dual Mode) requires at least 32GB VRAM natively, or 24GB with `--blocks_to_swap 4`.

**Source**: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md#dual-model

### Datasets and Resolution
- **Resolution**: 720x1280 and 832x480 are the most stable buckets. 
- **Frame counts**: Must adhere to `4n+1` (e.g., 81 frames, 121 frames). Breaking this rule throws a tensor dimension mismatch warning and skips the video.

### Optimizer and Precision 
- **Optimizer**: `schedulefree.RAdamScheduleFree` is often preferred over `adamw8bit` due to skipping the warmup phase. `adamw8bit` may cause occasional precision spikes.
- **fp8 vs bf16**: fp8 is mandatory for 24GB cards. bf16 quality is indistinguishable but requires A6000 (48GB) or A100.
- **Mandatory Flags**: You must specify the `--task` flag (e.g., `t2v-1.3B`, `t2v-A14B`, `i2v-A14B`) and `--network_module networks.lora_wan`.

## Contradictions and uncertainties

- People are highly conflicted on `--gradient_accumulation_steps`. The official doc says 1 is fine if batch size is 2, but the megathread users insist on 4 for stability. This needs testing.

## Raw source links

- https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md
- https://github.com/kohya-ss/musubi-tuner/blob/main/docs/advanced_config.md
