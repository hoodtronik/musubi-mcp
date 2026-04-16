# musubi-mcp

> **Status: early alpha.** The three priority architectures (Wan2.1/2.2,
> FLUX.2, Z-Image) are wired up; others are placeholders returning "not yet
> implemented" until their flags are audited.

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

> *This section is filled in as tools land.* Currently:

| Tool | Purpose |
|------|---------|
| `musubi_check_installation` | Report Musubi Tuner availability and config state. |
| `musubi_list_architectures` | List all registered architectures with capabilities. |
| `musubi_cache_latents` | Pre-cache VAE latents for a chosen architecture + dataset. |

Planned (not yet shipped):
`musubi_cache_text_encoder`, `musubi_train`, `musubi_finetune`,
`musubi_generate`, `musubi_convert_lora`, `musubi_merge_lora`,
`musubi_ema_merge`, `musubi_create_dataset_config`,
`musubi_validate_dataset`, `musubi_caption_images`.

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

## License

Apache-2.0. Same license as Musubi Tuner upstream.
