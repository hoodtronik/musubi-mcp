# Dataset Quality and Captioning Best Practices

*Last researched: April 17, 2026. Sources: 12 threads/docs/posts reviewed.*
*Confidence: HIGH — industry consensus on tagging has solidified heavily around sparse captioning in 2026.*

## Summary

- **Count Sweet Spots**: 20-50 high-res images for Character is better than 200 mediocre ones. Motion LoRAs require 15-30 video clips.
- **Caption Style**: Do NOT use flowery language. Direct, factual, token-dense statements rule.
- **Captioning Models**: Gemini 1.5 Pro and JoyCaption 2 (Ollama) beat legacy VLM models.

## Detailed findings

### Image Counts & Bucketing
- **Character**: 20-50 images. 
- **Style**: 50-150 images.
- **Motion/Vid**: 15-30 clips (3-10 seconds each).
- **Rule of Thumb**: Multi-bucket training is mandatory (allow the script to bucket resolutions dynamically). Cropping destroys structural learning for video models like Wan and LTX.

**Source**: https://github.com/alvdansen/klippbok/blob/main/docs/PIPELINES.md

### Captioning Methodology
- **The Modern Approach**: Sparse, structured tagging. "A woman with red hair walking. She is wearing a blue coat." is superior to "A cinematic, 8k masterpiece beautiful photo of a ginger lady strolling down the street in a navy jacket, masterpiece, trending on artstation". 
- **Trigger Words**: Place the trigger word at the absolute front of the caption (e.g. `zst character, a woman...`).

### Identity Leaking & Diversity
- If you train a character who is only wearing a white shirt in the dataset, the white shirt becomes baked into the identity. You must include varied clothing, backgrounds, and angles. If that's impossible, aggressively caption the white shirt in every frame so the model associates the shirt with the background text rather than the trigger word.

## Contradictions and uncertainties

- There is debate on whether to caption "obvious" things. E.g., if the character has a specific mole, do you caption it? The community split is 50/50. Some say captioning it allows you to remove it via negative prompt later; others say you should leave it uncaptioned so it binds strictly to the trigger word.

## Raw source links

- https://github.com/alvdansen/klippbok/blob/main/docs/CAPTIONING.md (non-authoritative community reference)
- https://github.com/kohya-ss/musubi-tuner/blob/main/docs/dataset_config.md
