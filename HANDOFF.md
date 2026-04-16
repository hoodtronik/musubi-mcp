# Handoff notes

For continuing or onboarding agents. Read alongside [README.md](README.md),
[AGENTS.md](AGENTS.md), and [ROADMAP.md](ROADMAP.md) — this file fills in
context those documents don't.

---

## Current state (2026-04-16)

- **Surface shipped:** 14 tools + 2 prompts + 17 doc resources. Full
  training pipeline wired (cache_latents → cache_text_encoder → train)
  plus generate + finetune + utilities + dataset tooling.
- **Tests:** 84 passing in ~0.6 s.
- **Live-verified end-to-end** against the local Musubi Tuner venv
  (`f:/__PROJECTS/musubi-tuner/.venv/`): install check reports torch
  2.11.0+cu128, CUDA 12.8, RTX 6000 Ada; `create_dataset_config` →
  `validate_dataset_config` → `cache_latents` spawns the real
  `wan_cache_latents.py` and fails cleanly on a fake VAE path; prompts
  render with context substitutions.
- **Live architectures:** `wan`, `flux_2`, `zimage`.
- **Placeholder architectures** (registered but not wired):
  `hv`, `hv_1_5`, `fpack`, `flux_kontext`, `qwen_image`, `kandinsky5`.

## Companion repos

| Repo | Role |
|---|---|
| [musubi-tuner.pinokio](https://github.com/hoodtronik/musubi-tuner.pinokio) | Interactive one-click launcher for the same upstream (Pinokio + Musubi's built-in Gradio GUI). |
| [klippbok-mcp](https://github.com/hoodtronik/klippbok-mcp) | Style reference — this server's FastMCP patterns, `CommandResult` dataclass, env discipline, and AGENTS.md convention all come from there. |
| [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) | Upstream. `docs/cli_help.txt` in this repo is dumped from a local clone at v0.2.15. |

## Non-obvious gotchas

1. **FLUX.2 uses `--vae` not `--ae`.** Despite the model being called "AE"
   in docs, the actual CLI flag is `--vae`. Regression-guarded in
   [test_architectures.py](tests/test_architectures.py) — don't "fix" it.

2. **Z-Image uses `--fp8_llm` not `--fp8_text_encoder`.** The Qwen3
   encoder gets the LLM flag. FLUX.2 uses `--fp8_text_encoder` (and
   FLUX.2 `dev` rejects that combo — Mistral3 doesn't support fp8).
   All three are encoded + tested.

3. **Wan2.2 I2V does NOT use `--clip`** (unlike Wan2.1 I2V). The
   `clip` TextEncoderSpec is marked `required=False` in the registry
   and the per-arch validation only requires `--t5`.

4. **Training script `--help` crashes on Windows cp1252.** Musubi's
   argparse help text includes Japanese characters. The runner
   forces `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` so runtime is
   fine; it was only the dev-time `cli_help.txt` dump that had to
   be re-run with those env vars. If you regenerate that file,
   export them first (see `scripts/regen_cli_help.sh` in ROADMAP).

5. **All Musubi scripts MUST run with cwd = `MUSUBI_TUNER_DIR`.**
   They use relative imports from their package. The runner sets
   this on every call; tool implementations just say which script.

6. **Training scripts go through `accelerate launch`**, not plain
   `python`. The runner has a dedicated `run_musubi_training`
   helper that handles the `accelerate launch
   --num_cpu_threads_per_process 1 --mixed_precision bf16` prefix.

7. **stdout/stderr truncated to last 8 KB for training results.**
   Training produces multi-MB logs; the MCP response stays under
   typical client limits. Full logs still land on disk via
   `--logging_dir`.

## How to test

```bash
# From f:/__PROJECTS/MusubiMCPSever
uv sync --extra dev
uv run pytest -q              # 84 passing

# Live end-to-end
MUSUBI_TUNER_DIR=f:/__PROJECTS/musubi-tuner \
MUSUBI_PYTHON=f:/__PROJECTS/musubi-tuner/.venv/Scripts/python.exe \
  uv run python -c "
import asyncio
from musubi_mcp.server import musubi_check_installation
print(asyncio.run(musubi_check_installation()))
"
```

## When flags drift upstream

Musubi Tuner is in active development. If a tool starts failing on
"unknown flag," the fix is in this order:

1. Regenerate `docs/cli_help.txt` with `PYTHONUTF8=1` env set,
   running `--help` on every affected script.
2. Compare to the new output and update either:
   - [architectures.py](src/musubi_mcp/architectures.py) for
     changes to task enums / model_version enums / encoder flags
   - The tool definition in
     [server.py](src/musubi_mcp/server.py) for changed flag names
3. Update or add a test in
   [test_server_tools.py](tests/test_server_tools.py) that asserts
   the new flag is in the generated argv.

## What's next

See [ROADMAP.md](ROADMAP.md). Highest-leverage next move is
activating the HunyuanVideo placeholder architecture — the flag
audit is the only meaningful work; registry + tests follow the
established pattern.
