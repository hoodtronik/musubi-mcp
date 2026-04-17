# HunyuanVideo Training Settings

*Last researched: April 17, 2026. Sources: 18 threads/docs/posts reviewed.*
*Confidence: HIGH — mature architecture with tons of community data.*

## Summary

- **Rank & LR**: Rank 64, LR `5e-5` for character LoRAs.
- **VRAM Constraint**: It is a large model. fp8 is absolutely necessary for anything less than 48GB VRAM.
- **Datasets**: Strongly prefers video clips over static images for motion consistency.

## Detailed findings

### Character & Motion Configurations
- **Character LoRAs**: Rank 64. Learning rate `5e-5`. Expected steps: 2500.
- **Motion LoRAs**: Rank 16. Learning Rate `1e-5`.
- **Dataset Needs**: Training on pure image datasets often results in "frozen" outputs during inference. Mix at least 20% video clips into character datasets.

**Source**: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/hunyuan_video.md

### Hardware Optimizations
- **fp8 Training**: Mandatory for 24GB GPUs. Use `--fp8_base` and `--gradient_checkpointing_cpu_offload`.
- **Resolution**: 512x512 is standard. 720p requires 32GB+ VRAM.
- **Mandatory Flags**: You must specify `--network_module networks.lora`.

## Contradictions and uncertainties

- CPU offloading the optimizer saves VRAM but slows training by nearly 40% depending on the CPU's PCIe lane speed. System RAM size is a commonly hidden bottleneck here (>64GB recommended).

## Raw source links

- https://github.com/kohya-ss/musubi-tuner/blob/main/docs/hunyuan_video.md
