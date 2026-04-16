# musubi-mcp

> **Status: alpha.** All core tools shipped. Three priority architectures
> (Wan2.1/2.2, FLUX.2, Z-Image) are live; others are placeholders that
> return "not yet implemented" until their flags are audited.

MCP server for [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner) by
kohya-ss — train LoRA / LoHa / LoKr models for Wan2.1/2.2, FLUX.2, Z-Image,
HunyuanVideo, HunyuanVideo 1.5, FramePack, FLUX.1 Kontext, Qwen-Image, and
Kandinsky 5 from any MCP-compatible AI agent (Claude Desktop, Claude Code,
Cursor, Antigravity, Codex, MCP Inspector).

Pairs with [klippbok-mcp](https://github.com/hoodtronik/klippbok-mcp) for
end-to-end dataset-curation → LoRA-training pipelines driven by an agent.

**This is an independent companion tool. Musubi Tuner must be installed
separately; this server shells out to its Python scripts.**

---

## Supported architectures

| Prefix | Architecture | Type | Training | Status |
|--------|-------------|------|----------|--------|
| `wan` | Wan2.1 / Wan2.2 | Video | LoRA | ✅ live |
| `flux_2` | FLUX.2 (dev / klein) | Image | LoRA | ✅ live |
| `zimage` | Z-Image (Base / Turbo) | Image | LoRA + full | ✅ live |
| `hv` | HunyuanVideo | Video | LoRA + full | 🚧 placeholder |
| `hv_1_5` | HunyuanVideo 1.5 | Video | LoRA | 🚧 placeholder |
| `fpack` | FramePack | Video | LoRA | 🚧 placeholder |
| `flux_kontext` | FLUX.1 Kontext | Image | LoRA | 🚧 placeholder |
| `qwen_image` | Qwen-Image / -Edit / -Layered | Image | LoRA + full | 🚧 placeholder |
| `kandinsky5` | Kandinsky 5 | Video | LoRA | 🚧 placeholder |

---

## Quick start

### 1. Install Musubi Tuner

```bash
git clone https://github.com/kohya-ss/musubi-tuner
cd musubi-tuner
uv sync --extra cu128   # or --extra cu124 / --extra cu130 for your GPU
```

Verify: `.venv/Scripts/accelerate.exe --help` (Windows) or
`.venv/bin/accelerate --help` (Linux/macOS) should print usage.

Run `accelerate config` once before any training tool will work.

### 2. Install musubi-mcp

```bash
git clone https://github.com/hoodtronik/musubi-mcp
cd musubi-mcp
uv sync
```

### 3. Configure the client

See [Client configs](#client-configs) below.

### 4. Verify

Call the `musubi_check_installation` tool. It reports Python version, torch
+ CUDA, which architectures are wired up, whether `accelerate config` has
been run, and any missing deps.

---

## Available tools

**Pipeline** — the three-step LoRA flow, in order:

| Tool | Mode | Purpose |
|---|---|---|
| `musubi_cache_latents` | python | Step 1: pre-cache VAE latents. |
| `musubi_cache_text_encoder` | python | Step 2: pre-cache text encoder outputs. |
| `musubi_train` | accelerate | Step 3: LoRA / LoHa / LoKr training. |
| `musubi_finetune` | accelerate | Full fine-tuning (Z-Image today; HunyuanVideo + Qwen-Image when placeholders go live). |
| `musubi_generate` | python | Sample image / video from a trained checkpoint (+optional LoRA). |

**Dataset** — build and audit dataset configs:

| Tool | Mode | Purpose |
|---|---|---|
| `musubi_create_dataset_config` | pure | Generate a valid dataset TOML. |
| `musubi_validate_dataset_config` | pure | Structural + 4n+1 frame validation. |
| `musubi_validate_dataset` | pure | Filesystem walk: images/videos + caption coverage. |
| `musubi_caption_images` | python | Auto-caption with Qwen2.5-VL. |

**LoRA utilities:**

| Tool | Mode | Purpose |
|---|---|---|
| `musubi_convert_lora` | python | Convert LoRA between default ↔ other (ComfyUI) formats. |
| `musubi_merge_lora` | python | Merge LoRAs into a base DiT. |
| `musubi_ema_merge` | python | Post-Hoc EMA merge of multiple LoRA checkpoints. |

**System:**

| Tool | Mode | Purpose |
|---|---|---|
| `musubi_check_installation` | — | Report Python / torch / CUDA / accelerate status. |
| `musubi_list_architectures` | — | Capability report for every registered architecture. |

**Prompts:**

| Prompt | Purpose |
|---|---|
| `plan_training_run` | Produce a concrete training plan from (architecture, training_type, source_data, hardware, goal). |
| `diagnose_training_issue` | Root-cause a failed / stuck training run from its log tail. |

**Resources:** `musubi://docs/{name}` serves Musubi Tuner's architecture
docs verbatim (`wan`, `flux_2`, `zimage`, `dataset_config`,
`advanced_config`, `loha_lokr`, `torch_compile`, ...).

---

## Configuration

All configuration goes through environment variables — never baked into
code.

| Variable | Required | Purpose |
|----------|----------|---------|
| `MUSUBI_TUNER_DIR` | **yes** | Absolute path to your Musubi Tuner checkout. All scripts run with this as cwd. |
| `MUSUBI_PYTHON` | recommended | Python interpreter inside Musubi's venv. Falls back to `sys.executable`. |
| `MUSUBI_ACCELERATE` | optional | `accelerate` binary path. Defaults to the one on `MUSUBI_PYTHON`'s PATH. |
| `GEMINI_API_KEY` | optional | Forwarded to captioning tools. |
| `REPLICATE_API_TOKEN` | optional | Forwarded to captioning tools. |
| `HF_TOKEN` | optional | Forwarded for Hugging Face model downloads. |

Copy `.env.example` → `.env` and fill it in.

---

## Client configs

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "musubi": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/musubi-mcp", "musubi-mcp"],
      "env": {
        "MUSUBI_TUNER_DIR": "/path/to/musubi-tuner",
        "MUSUBI_PYTHON": "/path/to/musubi-tuner/.venv/Scripts/python.exe"
      }
    }
  }
}
```

### Claude Code

```bash
claude mcp add musubi -- uv run --directory /path/to/musubi-mcp musubi-mcp
```

### Combined with klippbok-mcp (full pipeline)

```json
{
  "mcpServers": {
    "klippbok": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/klippbok-mcp", "klippbok-mcp"],
      "env": {
        "KLIPPBOK_PYTHON": "/path/to/klippbok-venv/bin/python",
        "GEMINI_API_KEY": "your-key"
      }
    },
    "musubi": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/musubi-mcp", "musubi-mcp"],
      "env": {
        "MUSUBI_TUNER_DIR": "/path/to/musubi-tuner",
        "MUSUBI_PYTHON": "/path/to/musubi-tuner/.venv/Scripts/python.exe"
      }
    }
  }
}
```

---

## Development

```bash
uv sync --extra dev
uv run pytest
```

Regenerate `docs/cli_help.txt` after upgrading Musubi Tuner:

```bash
# from the musubi-mcp repo
./scripts/regen_cli_help.sh /path/to/musubi-tuner
```

Run with MCP Inspector:

```bash
npx @modelcontextprotocol/inspector uv run --directory . musubi-mcp
```

---

## How the server is organized

- `src/musubi_mcp/server.py` — FastMCP server; tool / resource / prompt definitions.
- `src/musubi_mcp/architectures.py` — registry mapping architecture names to script prefixes, required args, network modules.
- `src/musubi_mcp/runner.py` — async subprocess wrapper. Two modes: plain `python` and `accelerate launch`.
- `src/musubi_mcp/dataset_config.py` — TOML generation + validation for Musubi's dataset configs.
- `src/musubi_mcp/resources.py` — serves Musubi's architecture docs as MCP resources.
- `src/musubi_mcp/constants.py` — script name mappings, default hyperparameters.
- `docs/cli_help.txt` — `--help` dump from every Musubi script (authoritative flag list).

---

## Roadmap

Planned expansions — architecture activations, runner streaming,
additional prompts, and developer ergonomics — are tracked in
[ROADMAP.md](ROADMAP.md). The highest-priority items are:

- Activate the six placeholder architectures (HunyuanVideo first, then
  Qwen-Image, FramePack, FLUX.1 Kontext, HunyuanVideo 1.5, Kandinsky 5).
- Streaming training progress back to the MCP client via `ctx.info()` /
  `ctx.report_progress()` so long runs aren't silent.
- A `scripts/regen_cli_help.sh` helper that re-dumps `docs/cli_help.txt`
  with `PYTHONUTF8=1` whenever Musubi Tuner upstream updates.

Contributions should follow the ground rules at the top of
[ROADMAP.md](ROADMAP.md): audit `docs/cli_help.txt` before writing new
tool code, cover every argv-building tool with a monkey-patched test,
and include one live smoke test against a real Musubi venv.

## License

Apache-2.0. Same license as Musubi Tuner upstream.
