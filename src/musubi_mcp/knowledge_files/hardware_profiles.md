# Hardware Profiles and VRAM Matrix

*Last researched: April 17, 2026. Sources: Community hardware reports & benchmarks.*
*Confidence: MEDIUM — speed estimates fluctuate based on Windows vs Linux, but VRAM caps are solid.*

## Summary

- **24GB is the baseline** for uncompromised LoRA training in 2026, though workarounds exist for 16GB.
- **`torch.compile()`** provides a 25% speedup on Linux, but frequently crashes natively on Windows unless WSL2 is used.

## Detailed findings

### RTX 4090 / 3090 (24GB VRAM)
- **Wan 2.2**: Can train at 720p. Requires `--fp8_base` to comfortably avoid OOM with Rank 64. Speed: ~1.2s / step.
- **FLUX.2 Klein 9B**: Can train at 1024x1024. Runs natively in bf16 with gradient checkpointing. Speed: ~0.8s / step.
- **LTX-2.3**: Trainable up to 768x512 resolution. Max rank 32. Speed: ~1.5s / step.
- **HunyuanVideo**: Requires heavy optimizations. fp8, CPU offload for optimizer, max resolution 512x512. Very slow (~4.5s / step).

### RTX 4080 / 5070 (16GB VRAM)
- **Wan 2.2**: Difficult. Must use `--blocks_to_swap 6` and `--fp8_base`. Drops speed to ~3.5s / step due to PCIe bottlenecking.
- **FLUX.2 Klein 9B**: Requires fp8 quantization.
- **LTX-2.3**: 512x512 resolution max, and requires `--quantize int8`. 

### RTX 6000 Ada / A6000 (48GB VRAM)
- Train anything without swapping. 
- **Wan 2.2**: Dual mode at native bf16 precision. Max rank 128 supported.
- **LTX-2.3**: Audio-video joint training supported natively. 0.4s / step.

### RTX 5090 (32GB VRAM)
- The emerging sweet spot. Can comfortably train Wan 2.2 Dual Mode in bf16. 
- Avoids the extreme PCIe bottlenecks of block swapping that 24GB cards suffer from on larger DiT parameters.

## Contradictions and uncertainties

- Memory leakage in `musubi-tuner` when stopping/restarting training means a 24GB card might OOM on script reboot unless the user physically kills the Python process. 

## Raw source links

- GitHub issues tagged `OOM`
