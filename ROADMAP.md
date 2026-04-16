# Roadmap

This file captures planned expansions for `musubi-mcp`. For shipped
functionality, see [README.md](README.md) and [AGENTS.md](AGENTS.md).

**Ground rules for additions:**
- Every new tool's parameter schema is audited against `docs/cli_help.txt`
  **before** the tool code is written. `docs/cli_help.txt` is the source
  of truth; memory and training cutoffs are not.
- New architectures move from placeholder to live by filling in their
  `ArchitectureConfig` entry (`vae_arg`, `text_encoders`, `network_module`,
  `tasks`, `model_versions`) and flipping `live=True`. Tests for the
  registry must include the new entry.
- Every argv-building tool gets a corresponding monkey-patched argv test
  in `tests/test_server_tools.py` or `tests/test_extra_tools.py`, plus
  one live smoke test against the real Musubi venv.

---

## 1. Activate placeholder architectures

Priority order (rationale): community adoption, VRAM-feasible on single
consumer GPUs, and overlap with the existing FastMCP arg patterns.

### 1a. HunyuanVideo (`hv`) — high priority

- Scripts: `hv_train_network.py`, `hv_train.py` (full fine-tune),
  `hv_generate_video.py`. Caching uses the generic repo-root
  `cache_latents.py` / `cache_text_encoder_outputs.py` (not `hv_`-prefixed
  — already encoded in the placeholder entry).
- Required model paths (audit against `docs/cli_help.txt`):
  `--dit`, `--vae`, `--text_encoder1` (LLaVA), `--text_encoder2` (CLIP).
  The two-encoder pattern diverges from Wan's `--t5 + --clip` and will
  need a new `TextEncoderSpec` pair in the registry.
- `--task` values: audit; HunyuanVideo's original release had a single
  task, but community forks (e.g. SkyReels I2V) extend the space.
- `network_module`: `networks.lora_hunyuan_video` (verify).
- 4n+1 rule applies — already encoded in
  `musubi_validate_dataset_config` when `architecture="hv"`.
- Full fine-tune is live-ready via `hv_train.py` (registry already has
  `train_full_script="hv_train.py"`).

### 1b. HunyuanVideo 1.5 (`hv_1_5`)

- Own `hv_1_5_*` scripts. Verify arg drift from HunyuanVideo original —
  1.5 likely changed encoders and DiT channel counts.

### 1c. Qwen-Image family (`qwen_image`)

- Qwen-Image, Qwen-Image-Layered, Qwen-Image-Edit share
  `qwen_image_*` scripts plus a `--multiple_target` flag on the
  dataset config (for Layered) and a `--control_directory` path (for
  Edit). Confirm whether a single `qwen_image` arch entry handles all
  three, or whether we need three sub-entries with different
  capabilities / defaults.
- `qwen_extract_lora.py` at repo root is already in the script
  inventory but not surfaced as a tool — add `musubi_qwen_extract_lora`
  alongside this activation.
- Full fine-tune available via `qwen_image_train.py`.

### 1d. FramePack (`fpack`)

- `fpack_*` scripts. Distinguishing flags: `fp_1f_*` frame controls for
  one-frame training (documented in
  `musubi://docs/framepack_1f`). Registry entry needs a `notes` line
  explaining the one-frame vs full-clip modes.
- 4n+1 rule applies with a window-multiplier twist — update
  `musubi_validate_dataset_config` to handle FramePack's
  `window * 4 + 1` pattern when the arch is `fpack`.

### 1e. FLUX.1 Kontext (`flux_kontext`)

- Image arch. Control-image driven (mirrors FLUX.2 control path). Audit
  `--text_encoder` count — FLUX.1 Kontext uses T5 + CLIP, not Mistral3.
- `network_module`: `networks.lora_flux` (likely shared with FLUX.1
  base, not FLUX.2).

### 1f. Kandinsky 5 (`kandinsky5`)

- Lower priority until community demand materializes. Audit encoder
  setup (Kandinsky historically uses mCLIP + T5).

---

## 2. Dataset config extensions

- **Control datasets**: `musubi_create_dataset_config` already accepts
  `control_directory`, `control_resolution`, `no_resize_control`. Add
  a helper that builds a FLUX.2 Kontext-style config with pinned 2024x2024
  control resolution when `--model_version` is dev/klein-9b.
- **JSONL metadata**: expose a separate tool
  `musubi_build_jsonl_metadata` that walks a directory and emits the
  `image_jsonl_file` / `video_jsonl_file` format Musubi consumes.
  Useful for pairing with `klippbok-mcp` which produces JSONL manifests.
- **Wan-specific 4n+1 + latent-window validation** for FramePack: the
  current `enforce_4n_plus_1` flag is all-or-nothing. Split into a
  per-arch validator with `fpack_window` parameter.

---

## 3. Runner + observability

- **Streaming logs** back to the MCP client via `ctx.info()` /
  `ctx.report_progress()`. Training can run for hours; a silent
  subprocess is painful. Add a `stream_progress: bool = False`
  parameter to `musubi_train` and `musubi_finetune`.
- **Parse torch OOM lines** in stderr and return structured
  `oom_suggestions` in the tool response (e.g. "peak allocation 23.8 GiB
  exceeded 24 GiB — try `blocks_to_swap=32` or `resolution=[768,768]`").
- **Log rotation / tailing**: currently we truncate training stdout to
  the last 8 KB. Add a helper tool `musubi_tail_training_log` that reads
  the latest log file from `--logging_dir` on demand.

---

## 4. Prompt expansion

- `plan_memory_budget` — takes `architecture` + `hardware` + `resolution`
  and returns the max model size + precision combo that fits.
- `critique_training_config` — takes a dataset TOML + proposed
  `musubi_train` params and reports likely regressions (e.g. learning
  rate too high for FLUX.2 Klein fine-tune, or `batch_size * num_repeats`
  imbalance).
- `suggest_lora_merge_plan` — takes a list of checkpoint paths and
  picks between `musubi_merge_lora` and `musubi_ema_merge` based on
  whether the user wants a single-model merge or a smooth checkpoint
  blend.

---

## 5. Developer ergonomics

- `scripts/regen_cli_help.sh` — one-liner that re-dumps
  `docs/cli_help.txt` from a given Musubi checkout with
  `PYTHONUTF8=1`. Currently documented in README but not scripted.
- Project-local `.claude/settings.local.json` with narrow allowlist
  for `Bash(uv run pytest)`, `Bash(uv sync *)`, `Bash(git status)`,
  `Bash(git diff *)` to reduce permission prompts during Claude Code
  sessions.
- Pre-commit hook: `ruff check src/ tests/` + `uv run pytest` before
  allowing a commit.
- CI: a GitHub Actions workflow that runs `pytest` on Linux with
  Python 3.10, 3.11, 3.12 (Musubi supports <3.13).

---

## 6. Upstream compatibility

- **Pin Musubi Tuner version** that `docs/cli_help.txt` was generated
  against in a frontmatter block at the top of that file. Makes drift
  detection mechanical.
- **Weekly automated `--help` diff** (via the CronCreate scheduling
  tool) to catch upstream flag renames before users hit them.

---

## 7. Not in scope

- GUI / Gradio: Musubi's `gui/` lives upstream and stays upstream.
- Model downloading: users bring their own checkpoints; we never shell
  out to `huggingface-cli download` from a tool.
- Training data cleaning / filtering: that's `klippbok-mcp`'s job.
  `musubi_validate_dataset` stays a *validator*, not a mutator.
