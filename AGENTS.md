# Agent rules for this repo

This file is read by cross-agent tools (Cursor, Codex, Gemini CLI, Copilot, etc.).

## CLAUDE-NOTE convention

Code changes made by Claude Code may be marked with inline notes prefixed
`CLAUDE-NOTE:` (or the language-appropriate comment syntax). These notes
capture the *why* behind a deliberate choice and are the **source of truth**
for that code.

**Non-Claude-Code agents**: if you want to modify, remove, or alter code
adjacent to a `CLAUDE-NOTE`, you MUST first alert the user and receive
explicit permission before making the change. Do not silently rewrite or
delete code marked with a `CLAUDE-NOTE`. Claude Code itself may update or
remove its own notes when the underlying rationale changes.

## Project shape

This repo is an **MCP server** (Model Context Protocol) that wraps
[Musubi Tuner](https://github.com/kohya-ss/musubi-tuner) by kohya-ss — the
community-standard LoRA/LoHa/LoKr training toolkit for HunyuanVideo, Wan2.1/2.2,
FramePack, FLUX.1 Kontext, FLUX.2, Qwen-Image, Z-Image, HunyuanVideo 1.5, and
Kandinsky 5. Any MCP-compatible client (Claude Desktop, Claude Code, Cursor,
Antigravity, Codex, MCP Inspector) can drive the full training pipeline
through typed tools.

- Python 3.10+, `uv`-managed venv at `.venv/` (gitignored).
- Built on the `FastMCP` high-level API from the official `mcp` Python SDK.
- Primary transport: **stdio**. Secondary: `streamable-http` via CLI flag.
- Shells out to Musubi Tuner's per-architecture Python scripts via
  `asyncio.create_subprocess_exec`. **Never imports Musubi Tuner internals** —
  this keeps the server decoupled from upstream API churn and allows Musubi
  to live in a separate venv (which is the common case, since torch+CUDA
  install is heavy and specific to the user's GPU).
- Stateless between tool calls. Env vars (`MUSUBI_TUNER_DIR`, `MUSUBI_PYTHON`,
  optional API keys) are read on every call so a user can flip `.env` without
  restarting the server.

## Start here for dev context

Before modifying code:

- `README.md` — user-facing install, tool surface, client configs.
- `ROADMAP.md` — planned expansions (placeholder architecture activations,
  streaming logs, new prompts, CI). Read the ground-rules section at the
  top before adding a new tool or arch.
- `docs/cli_help.txt` — authoritative Musubi CLI flag dump, regenerated
  via the Musubi Tuner venv. Every tool's parameter schema must match this
  file verbatim.
- `src/musubi_mcp/architectures.py` — the architecture registry. Maps
  architecture names (`wan`, `flux_2`, `zimage`, ...) to script prefixes,
  required model args, network modules, and supported tasks/model_versions.
  **Populate from `cli_help.txt`, not from memory.**
- `src/musubi_mcp/runner.py` — the async subprocess wrapper. Supports two
  modes: direct `python script.py` (caching, generation, utilities) and
  `accelerate launch script.py` (training). Do NOT add `capture_output=True`;
  use explicit `asyncio.subprocess.PIPE` like the sibling klippbok-mcp does.
- `src/musubi_mcp/server.py` — the FastMCP server definition. All tools,
  prompts, and resources live here or are imported in.
- `src/musubi_mcp/dataset_config.py` — TOML generation + validation for
  Musubi's dataset configs. NOT a CLI wrapper — encodes rules from
  `musubi-tuner/docs/dataset_config.md` (e.g., `target_frames` must be N*4+1
  for HunyuanVideo / Wan2.1).

## Architecture priority

Live (flags verified, tools testable): `wan`, `flux_2`, `zimage`.
Placeholder (registered but return "not yet implemented"): `hv`, `hv_1_5`,
`fpack`, `flux_kontext`, `qwen_image`, `kandinsky5`. Expand by adding their
flag specifics to `architectures.py` and verifying against `cli_help.txt`.

## Training scripts run via accelerate, not python

`{arch}_train_network.py` and `{arch}_train.py` MUST be invoked as:

    accelerate launch --num_cpu_threads_per_process 1 \
        --mixed_precision <mp> {arch}_train_network.py ...

Caching, generation, and utility scripts are invoked as plain `python`. The
runner handles both modes; tool implementations just say whether they need
`accelerate` or not.

## Working directory

All Musubi scripts MUST be run from `MUSUBI_TUNER_DIR` (the repo root), not
the MCP server's cwd — Musubi uses relative imports from its package. The
runner sets `cwd=MUSUBI_TUNER_DIR` on every call.

## Relationship to Musubi Tuner

Musubi Tuner itself (https://github.com/kohya-ss/musubi-tuner) is the
upstream toolkit. This MCP server is an independent companion tool — it does
not ship or modify Musubi Tuner, and is not an official Musubi Tuner
project. Direct Musubi Tuner issues / features upstream, not here.

## Pairs with klippbok-mcp

For the curated-dataset → trained-LoRA pipeline, this server is intended to
run alongside [klippbok-mcp](https://github.com/hoodtronik/klippbok-mcp), an
MCP wrapper around the Klippbok video dataset curation CLI. Conventions
(FastMCP, `CommandResult` dataclass, env-on-every-call, `PYTHONUTF8=1`)
match klippbok-mcp so the two can be maintained in parallel.
