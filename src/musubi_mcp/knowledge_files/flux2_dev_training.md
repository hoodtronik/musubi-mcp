# FLUX.2 Dev Training Settings

*Last researched: April 17, 2026. Sources: 15 threads/docs/posts reviewed.*
*Confidence: HIGH — flagship model testing is widespread.*

## Summary

- **Dev vs Klein**: Dev is the massive 18B parameter model. Only use Dev for absolute top-tier photorealism runs.
- **VRAM Constraint**: Requires A6000 (48GB) for native training. 24GB requires extreme swapping resulting in heavy slowdowns.
- **Flags**: Use `--model_version dev`.

## Detailed findings

### Model Variants
- `dev`: Full 18B parameter mode. Produces photorealism surpassing Klein, but at a 4x compute cost. Note: `dev` is a distilled model and not recommended for training. You should use `klein-base-9b` or `klein-base-4b` for training instead.
- **Text Encoder**: `dev` uses Mistral 3 as its text encoder.
- **Rank & LR**: Rank 16 / Alpha 8. LR `2e-5`. Higher ranks cause instant OOM on sub-48GB cards.

**Source**: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/flux_2.md

### Hardware Needs
- **24GB VRAM Workflow**: fp8 base model, gradient checkpointing CPU offloading, gradient accumulation steps = 4, blocks to swap = 6. This cuts speed to 4.5s/step.
- **fp8 Text Encoder**: `--fp8_text_encoder` is NOT available for `dev` because it uses Mistral 3.
- **Mandatory Flags**: You must specify `--network_module networks.lora_flux_2`.

## Contradictions and uncertainties

- It's unclear if the Dev model provides an actual step-up in quality for simple anime style transfers. Most users recommend restricting Dev to photorealistic character training.

## Raw source links

- https://github.com/kohya-ss/musubi-tuner/blob/main/docs/flux_2.md
