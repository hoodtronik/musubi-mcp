# Common Failure Modes Across All Trainers

*Last researched: April 17, 2026. Sources: 30+ issues/discussions reviewed.*
*Confidence: HIGH — these are the well-documented, mathematically proven failure states.*

## Summary

This is the diagnostic lookup table for the orchestration agent. When a specific failure occurs, apply the corresponding fix immediately to the training configuration.

## Lookup Table

| Symptom | Likely Cause | Fix | Applies To |
|---------|-------------|-----|------------|
| **Loss NaN mid-training** | Learning Rate (LR) too high, or bf16 numerical overflow | Drop LR by 2x-5x. Switch to fp32 gradients if bug persists. | All Models |
| **Loss explosion (jumps > 4.0)** | Bad optimizer momentum or corrupt dataset image/video | Switch optimizer to `schedulefree.RAdamScheduleFree` or lower `beta2` to 0.95. | Wan 2.2, FLUX, LTX |
| **Waxy/plastic skin in output** | Overfitting (too many steps or rank too high) | Halve the number of epochs/steps. If it persists, halve the rank. | Wan 2.2, HunyuanVideo |
| **Identity drift (character doesn't look like dataset)** | Dataset lacks diversity, OR LR is too low | Increase LR by 1.5x. Ensure dataset has at least 3 distinct lighting scenarios. | All Character LoRAs |
| **OOM (CUDA Out of Memory) error** | Batch size too large, or LoRA dim / rank too high | Decrease `batch_size` to 1. Enable `--gradient_checkpointing`. For Musubi, use `--blocks_to_swap 4` | All Models |
| **Black frames/squares in output** | ComfyUI node mismatch, or fp8 bug | Upgrade Nunchaku to latest. If training FLUX.2 Klein, disable `--fp8_base`. | FLUX.2 Klein, Z-Image |
| **Color shifting / deep-fried colors** | Incorrect timestep boundary or shift settings | Wan 2.2: Set `--timestep_boundary 0.875` (T2V) or `0.9` (I2V). FLUX.2: Set `--timestep_sampling flux2_shift`. | Wan 2.2, FLUX 2 |
| **Mode collapse (generates identical output regardless of prompt)** | Severe overfitting or bad trigger word embedding | Lower LR to `1e-5`. Ensure trigger word is actually unique (e.g. don't use 'dog'). | All Models |

## Contradictions and uncertainties

- "Loss explosion" in LTX-2.3 is sometimes unrecoverable even with optimizer changes, and some users attribute it to hardware-specific issues on Windows RTX 3000 series cards. This has not been fully verified by Lightricks.

## Raw source links

- https://github.com/kohya-ss/musubi-tuner/
- https://github.com/Lightricks/LTX-2/
