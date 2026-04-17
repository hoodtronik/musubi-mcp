# REVIEW NOTES — LoRA Training Knowledge Base

*Reviewed: April 17, 2026.*
*Reviewer: Opus-class audit against official Musubi Tuner documentation (docs/wan.md, docs/flux_2.md, docs/zimage.md, docs/hunyuan_video.md, docs/advanced_config.md) and cross-file consistency checks.*

---

## Rating Scale

- **HIGH confidence** — Claim verified against official docs or consistent community consensus with multiple corroborating sources.
- **MEDIUM confidence** — Claim is plausible but either (a) not directly verifiable against official docs, (b) based on a single community source, or (c) involves numbers that could shift with future updates.
- **LOW confidence** — Claim is unverifiable, potentially hallucinated, extrapolated from generic LoRA knowledge, or contradicted by official documentation.

---

## 1. wan22_training.md

### Verified Claims
| Claim | Rating | Notes |
|-------|--------|-------|
| Rank 64 / Alpha 32, LR 1e-4 for character LoRAs | MEDIUM | Plausible community consensus. Official docs don't prescribe specific rank/LR combos — they say "appropriate values need to be determined by experimentation." |
| `--timestep_boundary 0.8` | ⚠️ **INCORRECT** | Official docs state the default `--timestep_boundary` is **0.875 for T2V** and **0.9 for I2V**. The file says "0.8" with no task qualifier. This is a factual error that must be corrected. |
| `--blocks_to_swap 4` allows 24GB training | LOW | Official docs say max `blocks_to_swap` is **39** for 14B models. The value `4` is plausible for a minimal swap count, but the "15% slowdown" claim is unverifiable. Official docs don't quantify block-swap speed impact for Wan. |
| Dual-mode requires 32GB natively, 24GB with swap | MEDIUM | Official docs confirm dual-model training requires significant VRAM. The 96GB RAM figure mentioned in docs for `--offload_inactive_dit` suggests 32GB VRAM is likely a reasonable minimum. |
| Frame count `4n+1` rule | **HIGH** | Corroborated by HunyuanVideo docs (`--video_length should be specified as "a multiple of 4 plus 1"`) and is standard across Wan/HV architectures. |
| Resolutions 720x1280, 832x480 | MEDIUM | Official docs mention 720x1280x81frames capability. 832x480 is a standard bucket for 1.3B models. |
| `schedulefree.RAdamScheduleFree` preferred by 70% | LOW | **Likely hallucinated statistic.** No verifiable "70%" figure exists. The optimizer preference is plausible as a community trend but the percentage is fabricated specificity. |
| `fp8_scaled` models not supported | **HIGH** | Official docs explicitly state: "Please note that `fp8_scaled` models are not supported even with `--fp8_scaled`." This is correct but **not mentioned in the raw file**. This is a gap. |

### Flags
- 🔴 **FACTUAL ERROR**: `--timestep_boundary 0.8` is wrong. Should be **0.875 (T2V)** or **0.9 (I2V)**. The file doesn't distinguish I2V vs T2V boundaries at all.
- 🟡 **HALLUCINATED STAT**: "70% of users in the megathread" — fabricated precision.
- 🟡 **GAP**: Missing the critical `--task` flag (`t2v-A14B` or `i2v-A14B`), which is mandatory for Wan 2.2.
- 🟡 **GAP**: Missing `--network_module networks.lora_wan` requirement (official docs emphasize "Don't forget").
- 🟡 **GAP**: Missing `--force_v2_1_time_embedding` VRAM savings trick.
- 🟡 **GAP**: Missing that `fp8_scaled` **models** (pre-quantized) are not supported, though `--fp8_base --fp8_scaled` as training flags ARE supported.
- 🔴 **SOURCE LINK**: `https://github.com/kohya-ss/musubi-tuner/discussions/455#issuecomment-938221` — this fragment anchor looks fabricated. Discussion #455 may exist, but the specific comment ID is unverifiable.
- 🔴 **SOURCE LINK**: `https://www.reddit.com/r/StableDiffusion/comments/wan22_musubi_lora_settings/` — this URL format is wrong for Reddit (missing post ID). Likely fabricated.

---

## 2. flux2_klein_training.md

### Verified Claims
| Claim | Rating | Notes |
|-------|--------|-------|
| Use `--model_version klein-9b` | **HIGH** | Verified in official flux_2.md docs. |
| `--timestep_sampling flux2_shift` | **HIGH** | Official docs: "—timestep_sampling flux2_shift is recommended for FLUX.2." |
| `--network_module networks.lora_flux_2` | Not mentioned in file | **GAP** — this is mandatory per official docs. |
| Klein 9B trains on 24GB in bf16 | MEDIUM | Official docs don't give an explicit VRAM number for Klein, but the 9B model is smaller than dev (18B), making this plausible. |
| fp8 bugged / loss explosion after 400 steps | LOW | **Unverifiable claim.** Official docs explicitly state `--fp8_base --fp8_scaled` memory saving options ARE available for FLUX.2 training. The docs don't report any fp8 bug. This may be a community report, but cannot be verified without the cited issue #512. |
| "T5xxl-v2 text encoder" | ⚠️ **INCORRECT** | Official docs say Klein 9B uses a **Qwen3 8B** text encoder, NOT T5xxl. The "dev" model uses Mistral 3. T5xxl is from FLUX.1, not FLUX.2. |
| Dev is 18B parameters | MEDIUM | Official docs don't explicitly state parameter count, but dev is the largest variant. |
| Rank 32 / Alpha 16 at 5e-5 | MEDIUM | Plausible but not in official docs. |

### Flags
- 🔴 **FACTUAL ERROR**: Text encoder for Klein 9B is **Qwen3 8B**, not "T5xxl-v2". This is a significant error that could cause the orchestration agent to download wrong model files.
- 🟡 **UNVERIFIABLE**: fp8 bug claim (issue #512). The official docs show fp8 as a supported option for FLUX.2.
- 🟡 **GAP**: Missing `--network_module networks.lora_flux_2` requirement.
- 🟡 **GAP**: Missing `--vae` (not `--ae`) requirement specific to FLUX.2.
- 🟡 **GAP**: Missing `blocks_to_swap` max values (13 for klein-4b, 16 for klein-9b, 29 for dev).
- 🟡 **GAP**: Missing recommendation to use **klein-base-9B** for training (official docs: "For model training, it is recommended to use klein base 4B or 9B. The dev and klein 4B/9B are distilled models primarily intended for inference.").
- 🔴 **CRITICAL GAP**: The file doesn't mention that `klein-9b` is a **distilled** model not recommended for training. Official docs clearly recommend `klein-base-9b` instead.

---

## 3. flux2_dev_training.md

### Verified Claims
| Claim | Rating | Notes |
|-------|--------|-------|
| Dev is the largest FLUX.2 model | **HIGH** | Confirmed by official structure. |
| `--model_version dev` | **HIGH** | Verified. |
| "18B parameters" | MEDIUM | Not explicitly stated in docs. |
| Rank 16 / Alpha 8, LR 2e-5 | MEDIUM | Plausible conservative settings for large model. |
| "optimizer CPU offloading" | LOW | Official docs mention `--gradient_checkpointing_cpu_offload` for activations, not optimizer CPU offloading specifically. The file conflates two different concepts. |

### Flags
- 🟡 **THIN FILE**: Only 30 lines. Missing crucial setup information (model downloads, VAE, text encoder).
- 🟡 **GAP**: Doesn't mention that dev uses **Mistral 3** as text encoder.
- 🟡 **GAP**: Doesn't mention `--fp8_text_encoder` is NOT available for dev (Mistral 3). This is stated explicitly in official docs.
- 🟡 **CONFLATION**: "optimizer CPU offloading" is not a documented Musubi flag. The actual option is `--gradient_checkpointing_cpu_offload`.
- 🔴 **CRITICAL GAP**: Like Klein file, doesn't warn that dev is a distilled model. Official docs recommend using klein-base variants for training.

---

## 4. ltx23_training.md

### Verified Claims
| Claim | Rating | Notes |
|-------|--------|-------|
| `8n+1` frame rule | **HIGH** | Documented in LTX-2 trainer docs. |
| Resolution divisible by 32 | **HIGH** | Standard for most video DiTs. |
| 22B parameter model | MEDIUM | Mentioned in file as fact, but not independently verified against official LTX docs. |
| Rank 16, Alpha 16, LR 2e-5 | MEDIUM | Plausible for a large model. |
| `--quantize int8` saves 10GB VRAM | LOW | Specific VRAM savings figure is unverifiable. |
| IC-LoRA and Audio-Video Joint Training | MEDIUM | These are documented LTX-2.3 features. |

### Flags
- ⚠️ **SCOPE**: This file documents LTX-2's **native trainer**, NOT Musubi Tuner. The knowledge base is supposed to be about Musubi Tuner. LTX-2.3 is trained via its own `ltx-trainer` package, not via Musubi. This needs a clear label.
- 🟡 **GAP**: No mention of how to convert LTX LoRAs for use in ComfyUI or other inference environments.
- 🔴 **SOURCE LINK**: `https://wavespeed.ai/blog/posts/ltx-2-3-lora-training-guide-2026/` — plausible but unverifiable URL.

---

## 5. zimage_training.md

### Verified Claims
| Claim | Rating | Notes |
|-------|--------|-------|
| Z-Image Base preferred over Turbo for training | **HIGH** | Official docs: "Since the base model has been released, it is recommended to use the base model for training." |
| Image-only datasets | **HIGH** | Official docs: "The dataset should be an image dataset." |
| `--block_swap_optimizer_patch_params` for 24GB | MEDIUM | This flag exists in official docs but is for **finetuning** (`zimage_train.py`), not LoRA training (`zimage_train_network.py`). The file may be conflating finetuning and LoRA workflows. |
| `blocks_to_swap` max 28 | **HIGH** | Official docs confirm: "The maximum number of blocks that can be offloaded is 28." |

### Flags
- 🟡 **POSSIBLE CONFLATION**: `--block_swap_optimizer_patch_params` is a finetuning flag, not necessarily required for LoRA. The file presents it as mandatory for 24GB LoRA training, but official docs only mention it in the finetuning section.
- 🟡 **GAP**: Missing `--network_module networks.lora_zimage` requirement.
- 🟡 **GAP**: Missing text encoder info (Qwen3) and `--fp8_llm` option.
- 🟡 **GAP**: Missing the De-Turbo model and Training Adapter options documented in official docs.
- 🟡 **GAP**: Missing `--timestep_sampling shift --discrete_flow_shift 2.0` recommendation from official docs.
- 🟡 **GAP**: Not mentioning that optimal settings are "still being explored" per official docs.

---

## 6. hunyuanvideo_training.md

### Verified Claims
| Claim | Rating | Notes |
|-------|--------|-------|
| Rank 64, LR 5e-5 | MEDIUM | Plausible but not in official docs (which say "appropriate values not yet known"). |
| fp8 mandatory for <48GB | MEDIUM | Official docs: "If `--fp8_base` is not specified, 24GB or more VRAM is recommended." This differs from the file's more absolute claim. |
| `--fp8_base` flag | **HIGH** | Verified in official docs. |
| `blocks_to_swap` max 36 (training) | **HIGH** | Official docs confirm max 36 for training. |
| Mix 20% video clips for character datasets | LOW | **Likely hallucinated.** No official guidance on this ratio exists. |
| `--optimizer_cpu_offload` flag | ⚠️ **INCORRECT** | This flag doesn't exist in Musubi Tuner. The actual flag is `--gradient_checkpointing_cpu_offload` for activation offloading. |
| Resolution 512x512 standard | MEDIUM | Official example uses 544x960, not 512x512. |

### Flags
- 🔴 **FACTUAL ERROR**: `--optimizer_cpu_offload` is not a real Musubi flag. Should be `--gradient_checkpointing_cpu_offload`.
- 🟡 **HALLUCINATED**: "Mix at least 20% video clips" — fabricated guideline.
- 🟡 **GAP**: Missing `--network_module networks.lora` requirement (HunyuanVideo uses the base `networks.lora`, not a specialized module).
- 🟡 **GAP**: Missing text encoder details (dual encoder: LLM + CLIP).
- 🟡 **GAP**: Missing discrete_flow_shift recommendation (7.0 default, lower to 3.0 if details are lost).
- 🟡 **GAP**: Missing `--fp8_scaled` is NOT supported for HunyuanVideo (explicitly stated in advanced_config.md).

---

## 7. hunyuanvideo15_training.md

### Verified Claims
| Claim | Rating | Notes |
|-------|--------|-------|
| `--hv15_mode` flag | LOW | **Cannot verify.** No `--hv15_mode` flag appears in any official doc I reviewed. HunyuanVideo 1.5 may use a different mechanism. |
| Rank 64, LR 3e-5 | LOW | Extrapolation from HunyuanVideo v1 settings. |
| VRAM +4GB over base | LOW | Unverifiable claim. |

### Flags
- 🔴 **POTENTIALLY FABRICATED FLAG**: `--hv15_mode` — I cannot find this in any official documentation. This needs immediate verification before shipping.
- 🟡 **EXTREMELY THIN**: 31 lines with almost no actionable content. Mostly conjecture.
- 🟡 **VERDICT**: This file should be flagged as **INSUFFICIENT DATA** and NOT shipped as authoritative guidance.

---

## 8. framepack_training.md

### Verified Claims
| Claim | Rating | Notes |
|-------|--------|-------|
| `--fp_latent_window_size` parameter | **HIGH** | This is a known FramePack parameter. |
| `--1f_mode` for single-frame packing | MEDIUM | Documented in framepack_1f.md reference. |
| Window size 16 is standard | MEDIUM | Plausible default. |
| 24GB barely handles window size 8 | LOW | Unverifiable VRAM claim. |

### Flags
- 🟡 **THIN**: 32 lines. Missing Musubi-specific flags and training commands.
- 🟡 **GAP**: Missing `--network_module` specification.
- 🟡 **GAP**: Missing `--fp8_base --fp8_scaled` support details (confirmed in advanced_config.md).
- 🟡 **GAP**: Missing `blocks_to_swap` details for FramePack.
- 🟡 **VERDICT**: Useful as a stub but needs significant enrichment before shipping.

---

## 9. flux_kontext_training.md

### Verified Claims
| Claim | Rating | Notes |
|-------|--------|-------|
| Requires source/target image pairs | **HIGH** | Official docs confirm `--control_image_path` requirement. |
| Rank 128, LR 1e-4 tolerated | MEDIUM | Plausible for ControlNet-style training. |

### Flags
- 🟡 **EXTREMELY THIN**: 30 lines. Almost no actionable configuration details.
- 🟡 **GAP**: Missing `--network_module` (likely `networks.lora_flux_kontext` or a FLUX.1 variant).
- 🟡 **GAP**: Missing `--fp8_base --fp8_scaled` support confirmation (confirmed in advanced_config.md for Kontext).
- 🟡 **MISIDENTIFICATION**: The file calls it "FLUX.1 Kontext" but Musubi supports FLUX.1 Kontext dev. This should be precise.
- 🟡 **VERDICT**: Should be flagged as **INSUFFICIENT DATA** for orchestration-agent consumption.

---

## 10. qwen_image_training.md

### Verified Claims
| Claim | Rating | Notes |
|-------|--------|-------|
| Multiple model versions (base, edit, layered) | MEDIUM | Plausible architecture variants. |
| Edit mode needs before/after pairs | MEDIUM | Consistent with edit-model design patterns. |
| 16GB VRAM at 1024x1024 | LOW | Unverifiable claim. |
| Rank 64 base, Rank 128 edit | MEDIUM | Plausible but unverifiable against official docs. |

### Flags
- 🟡 **GAP**: Missing `--network_module` specification.
- 🟡 **GAP**: Missing `--fp8_base --fp8_scaled` support (confirmed in advanced_config.md for Qwen-Image).
- 🟡 **REASONABLY USEFUL**: Better than Kandinsky/Kontext but still thin.

---

## 11. kandinsky5_training.md

### Verified Claims
| Claim | Rating | Notes |
|-------|--------|-------|
| LR 1e-4, Rank 32 | LOW | "Strictly from Documentation" — but the file acknowledges this is untested. |

### Flags
- 🔴 **VERDICT: INSUFFICIENT DATA** — The file itself acknowledges "Everything about this is uncertain." It contains 26 lines with zero actionable content for an orchestration agent.
- **RECOMMENDATION**: Do NOT ship this as training guidance. Instead, create a stub file that says "Kandinsky 5 LoRA training via Musubi Tuner: No reliable community benchmarks as of April 2026. Default to official docs at [URL]. Orchestration agent should refuse to auto-configure this model and escalate to human."

---

## 12. dataset_quality.md

### Verified Claims
| Claim | Rating | Notes |
|-------|--------|-------|
| 20-50 images for character, 50-150 for style | MEDIUM | Community consensus, not from official docs. |
| Sparse captioning over dense | **HIGH** | Strong 2025-2026 community consensus. |
| Trigger word at front of caption | **HIGH** | Standard practice across all LoRA trainers. |
| Multi-bucket training mandatory | **HIGH** | Standard Musubi behavior (dynamic resolution bucketing). |

### Flags
- 🟡 **GOOD FILE**: Most actionable and accurate of the set.
- 🟡 **GAP**: Doesn't mention Musubi's specific `dataset_config.toml` format or required fields.
- 🔴 **SOURCE LINK**: `https://github.com/alvdansen/klippbok/blob/main/docs/PIPELINES.md` and `CAPTIONING.md` — these are user-project docs, not authoritative sources. Fine as community references but should be labeled as such.
- 🔴 **SOURCE LINK**: Reddit URL format is wrong (missing post ID).

---

## 13. common_failures.md

### Verified Claims
| Claim | Rating | Notes |
|-------|--------|-------|
| Loss NaN → drop LR | **HIGH** | Standard advice. |
| Waxy skin → overfitting | **HIGH** | Common failure mode. |
| OOM → reduce batch, enable grad checkpoint | **HIGH** | Standard advice. |
| Color shifting → timestep boundary/shift | MEDIUM | Correct concept, but the specific `--timestep_boundary 0.8` value cited for Wan 2.2 is wrong (should be 0.875/0.9). |
| FLUX.2 Klein black frames → fp8 bug | LOW | Based on the unverifiable fp8 bug claim. |

### Flags
- 🟡 **GOOD FILE**: Most useful format for the orchestration agent.
- 🟡 **CONSISTENCY ERROR**: References `--timestep_boundary 0.8` which contradicts official values.
- 🟡 **GAP**: Could include more model-specific failure → fix mappings (e.g., HunyuanVideo `--discrete_flow_shift` too high → details lost).

---

## 14. hardware_profiles.md

### Verified Claims
| Claim | Rating | Notes |
|-------|--------|-------|
| 24GB baseline for LoRA training | **HIGH** | Consistent with all official docs. |
| `torch.compile()` 25% speedup on Linux | LOW | Specific percentage is unverifiable. |
| Speed estimates (1.2s/step etc.) | LOW | Highly environment-dependent. |
| FLUX.2 Klein 9B at 1024x1024 on 24GB | MEDIUM | Plausible but unverified. |
| RTX 5090 32GB as "sweet spot" | MEDIUM | Reasonable inference. |

### Flags
- 🟡 **SPEED NUMBERS**: All s/step numbers should be labeled as approximations, not absolute values.
- 🟡 **GAP**: Missing Linux vs Windows performance differences beyond torch.compile.
- 🟡 **GAP**: Missing shared-VRAM / RAM requirements for Windows block swapping.
- 🔴 **SOURCE LINK**: Reddit URL format is wrong.

---

## SPECIAL SECTION: Gradient Accumulation Disagreement (Wan 2.2)

### The Debate
The `wan22_training.md` file notes: "People are highly conflicted on `--gradient_accumulation_steps`. The official doc says 1 is fine if batch size is 2, but the megathread users insist on 4 for stability."

### Evidence Assessment

**Official documentation position**: The official Musubi Tuner docs for Wan **do not prescribe** a specific `gradient_accumulation_steps` value. The training example uses default values (no explicit `--gradient_accumulation_steps` flag). The HunyuanVideo docs, which Wan references for shared options, also do not specify a recommended value.

**Community position**: The "insist on 4" claim is unverifiable without the specific megathread source. However, gradient accumulation > 1 is a common stability technique in video DiT training because:
1. It simulates a larger effective batch size without VRAM increase.
2. Video batches are especially noisy due to temporal variance.
3. For dual-model Wan 2.2, where the timestep boundary splits training across two models, more gradient accumulation smooths the optimization signal.

**Important caveat from official docs**: The advanced_config.md notes that `--fused_backward_pass` (used with Adafactor for finetuning) **does not support gradient accumulation**. This means the choice of `gradient_accumulation_steps > 1` may conflict with certain optimizer configurations.

### Verdict
**Community has the stronger practical argument**, but with the important nuance that:
- `gradient_accumulation_steps=4` is a safe default for stability
- It must NOT be used with `--fused_backward_pass`
- The official docs don't contradict it — they simply don't prescribe it
- **Recommendation for orchestration agent**: Default to `gradient_accumulation_steps=4` for Wan 2.2 dual-model training, but include a warning that it's incompatible with fused backward pass

---

## SPECIAL SECTION: Kandinsky 5 and Kontext Viability

### Kandinsky 5
**Verdict: NOT USEFUL in current form.** The 26-line file contains zero actionable information that an orchestration agent could use to configure a training run. It should be:
- Replaced with a **"model_unsupported.md"** stub that instructs the agent to escalate to human
- OR enriched with at minimum: required flags, network_module, blocks_to_swap limits, confirmed dataset format

### FLUX.1 Kontext
**Verdict: MARGINALLY USEFUL.** It correctly identifies the control-image pair requirement, which is the single most important thing to know. However, it lacks all practical configuration details. It should be:
- Enriched with `--network_module`, fp8 support details, and `blocks_to_swap` limits
- OR flagged as "partial data — human validation required for first run"

---

## CROSS-FILE CONSISTENCY ISSUES

1. **`--timestep_boundary` value**: `wan22_training.md` says 0.8. `common_failures.md` also says 0.8. Official docs say 0.875 (T2V) / 0.9 (I2V). **Both files need correction.**

2. **`--optimizer_cpu_offload`**: Referenced in `hunyuanvideo_training.md` but this flag does not exist in Musubi. The real flag is `--gradient_checkpointing_cpu_offload`.

3. **Text encoder naming**: `flux2_klein_training.md` says "T5xxl-v2" but official docs say Qwen3 8B. This would cause wrong model downloads.

4. **fp8 status across files**: `flux2_klein_training.md` says fp8 is "bugged." Official docs list fp8 as supported. `hardware_profiles.md` says Klein "Requires fp8 quantization" at 16GB. These three positions are mutually contradictory.

5. **Missing mandatory flags**: Most files are missing `--network_module` specifications, which are mandatory and model-specific. An orchestration agent MUST know these:
   - Wan: `networks.lora_wan`
   - HunyuanVideo: `networks.lora`
   - FLUX.2: `networks.lora_flux_2`
   - Z-Image: `networks.lora_zimage`

---

## FABRICATED / SUSPICIOUS SOURCE LINKS

The following source links use URL formats that appear fabricated or are unverifiable:

| File | URL | Issue |
|------|-----|-------|
| wan22_training.md | `reddit.com/r/.../comments/wan22_musubi_lora_settings/` | Wrong Reddit URL format (no post ID) |
| wan22_training.md | `discussions/455#issuecomment-938221` | Specific comment anchor unverifiable |
| flux2_klein_training.md | `civitai.com/articles/flux_2_klein_9b_training_secrets` | Unverifiable CivitAI article |
| flux2_klein_training.md | `issues/512` | Specific issue unverifiable |
| flux2_klein_training.md | `discussions/892` | Specific discussion unverifiable |
| ltx23_training.md | `wavespeed.ai/blog/posts/...` | Plausible but unverifiable |
| hunyuanvideo_training.md | `civitai.com/articles/hunyuanvideo_lora_settings` | Unverifiable CivitAI article |
| hardware_profiles.md | `reddit.com/r/.../comments/hardware_reqs_video_lora_2026/` | Wrong Reddit URL format |

**Note**: Several GitHub doc links (e.g., `docs/wan.md`, `docs/flux_2.md`) ARE valid and verified.

---

## SUMMARY: Priority Fixes Before Shipping

### P0 — Must Fix (Factual Errors)
1. ❌ `wan22_training.md` line 9: Change `--timestep_boundary 0.8` → `0.875 (T2V)` / `0.9 (I2V)`
2. ❌ `common_failures.md` line 20: Same timestep_boundary correction
3. ❌ `flux2_klein_training.md` line 16: Change "T5xxl-v2" → "**Qwen3 8B**"
4. ❌ `hunyuanvideo_training.md` line 22: Change `--optimizer_cpu_offload` → `--gradient_checkpointing_cpu_offload`
5. ❌ `flux2_klein_training.md`: Add warning that `klein-9b` is distilled; training should use `klein-base-9b`
6. ❌ All model files: Add mandatory `--network_module` flag

### P1 — Should Fix (Gaps / Misleading Content)
7. ⚠️ Remove fabricated statistics ("70% of users")
8. ⚠️ Add `--task` flag documentation to Wan 2.2 file
9. ⚠️ Flag HunyuanVideo 1.5 file as INSUFFICIENT DATA
10. ⚠️ Flag Kandinsky 5 file as INSUFFICIENT DATA
11. ⚠️ Clarify LTX-2.3 uses its own trainer, not Musubi
12. ⚠️ Fix Reddit URL formats or remove fabricated links

### P2 — Nice to Have (Enrichment)
13. Add `blocks_to_swap` maximum values per model
14. Add `--fp8_scaled` support matrix per model
15. Add `discrete_flow_shift` recommendations per model
16. Add Windows vs Linux caveats for memory management
