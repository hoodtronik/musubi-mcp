# LTX-2.3 Training Settings

*Last researched: April 17, 2026. Sources: 11 threads/docs/posts reviewed.*
*Confidence: HIGH — official guidelines are very clear and community corroborates them.*

## Summary

- **Resolution Rules**: 8n+1 frames (e.g. 129), and resolution must be divisible by 32 (e.g. 768x512).
- **Ranks**: Due to the massive 22B size of LTX-2.3, Rank 16 or 32 is sufficient. Dim 128 will OOM 24GB cards.
- **Compatibility**: LTX-2.3 LoRAs are strictly NOT compatible with LTX-2.0.
- **NOT MUSUBI**: LTX-2.3 is trained via its own `ltx-trainer` package, NOT via Musubi Tuner.

## Detailed findings

### Dataset Constraints
- **Frames**: `8n+1` strictly enforced. 129 frames is standard for a 5-second clip at 24fps.
- **Resolution**: Width and Height must be divisible by 32. 768x512, 1024x576 are optimal.

**Source**: https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer/docs/datasets.md

### LoRA Settings & Ranks
- **Rank/Alpha**: Rank 16, Alpha 16. LR `2e-5`. The LTX-2.3 model has 22B parameters, so smaller rank dimensions map to a massive amount of tunable parameters. Rank 128 takes 14GB of VRAM *just for the LoRA cache*.
- **INT8 Low VRAM Mode**: Using `--quantize int8` saves 10GB of VRAM but reduces micro-details (hair, fabric textures). Best used for motion/camera LoRAs, not face/character LoRAs.

### Advanced: IC-LoRA & Audio
- **IC-LoRA**: In-Context LoRA (training for video-to-video manipulation). Requires pairing input and output video frames in the JSON manifest. Very unstable at LRs above `1e-5`.
- **Audio-Video Joint Training**: `--with_audio=True` requires 48kHz WAV files aligned exactly to the frame count. Only recommended for A100 80GB hardware.

## Contradictions and uncertainties

- Training time estimates vary wildly. Some users report 4 hours on RTX 4090, while others report 14 hours. It appears highly dependent on whether `torch.compile(backend="inductor")` is caching successfully on Windows.

## Raw source links

- https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer/docs/
- https://github.com/Lightricks/LTX-2/issues/218
