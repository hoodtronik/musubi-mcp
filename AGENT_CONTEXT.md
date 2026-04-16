# Agent context

PC-agnostic context for any agent (Claude Code, Cursor, Codex, Gemini
CLI, other human developers) picking up work on this repo. Read this
alongside [AGENTS.md](AGENTS.md), [HANDOFF.md](HANDOFF.md), and
[ROADMAP.md](ROADMAP.md).

This file is intentionally portable — no local paths, no machine
names, no per-OS assumptions. If you're resuming work on a different
PC than the one where prior work happened, this file has everything
you need.

---

## The developer

- Publishes the full ecosystem under the **`hoodtronik`** GitHub org.
  All companion repos (`musubi-mcp`, `musubi-tuner.pinokio`,
  `klippbok-mcp`) are public.
- Operates as a product lead / architect — writes specs up front,
  delegates implementation to agents, reviews at explicit checkpoints.
- Wants concrete verification (live smoke tests against real
  subprocesses / real venvs), not just diff reads or unit-test claims.
- Expects end-of-turn summaries in table / bullet format with
  clickable file + line refs, not prose paragraphs.

---

## How to collaborate

### Git flow

**Init locally, commit per logical step, wait for an explicit
"push it" before creating a GitHub remote.**

- When scaffolding a new repo: `git init -b main` + commits, but NOT
  `gh repo create` until asked.
- When the developer says *"push it to the repo"* / *"push X to my
  github"* — that is the authorization. Default to **public**
  visibility (all three companion repos are public) unless asked
  otherwise. Typical invocation:
  ```bash
  gh repo create hoodtronik/<name> --public \
      --source=. --remote=origin \
      --description "..." --push
  ```
- Never force-push, rewrite history, change `.gitconfig`, or skip
  hooks. Create new commits rather than amending.
- Every commit includes a `Co-Authored-By: Claude Opus 4.7 (1M
  context) <noreply@anthropic.com>` trailer.

### Response style

- **Tight summaries over prose.** Tables for multi-item status;
  bullets for sequential steps; clickable markdown file refs.
- **Live-verify before saying "done."** Pair unit tests with at least
  one end-to-end invocation against a real subprocess / real venv.
  "Tests pass" alone reads as incomplete.
- **Clarifying questions up front for new specs.** Before writing
  code, ask 3–5 focused questions that can't be answered from the
  spec + environment. Offer a recommended answer per question so the
  developer can confirm fast. This has been validated across every
  major task in the ecosystem.
- **Form-based UX for N-option selects.** When a user has to pick
  from several expensive options (e.g. per-architecture model
  bundles, per-feature toggles), present a form with individual
  labeled options — not all-or-nothing.
- **Auto-detect with fallback beats hardcoded defaults.** For
  user-facing tools, prefer detection + a widest-compatibility
  fallback over either static defaults or manual configuration.

### Executing actions with care

- Local reversible edits (code changes, tests) don't need confirmation.
- **Visible / hard-to-reverse actions always need confirmation unless
  explicitly authorized**: creating GitHub repos, pushing to remote,
  force-pushing, deleting branches, opening PRs, posting comments.
  The word "push" in the developer's message is typically the green
  light for the narrow push action specified; it does not authorize
  unrelated pushes.

---

## The ecosystem

Four repos building an interconnected toolkit around
[kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner):

| Repo | Role |
|---|---|
| [hoodtronik/musubi-mcp](https://github.com/hoodtronik/musubi-mcp) | MCP server wrapping Musubi Tuner's training scripts as typed tools for AI agents (Claude Desktop / Code / Cursor / Antigravity). **This repo.** |
| [hoodtronik/musubi-tuner.pinokio](https://github.com/hoodtronik/musubi-tuner.pinokio) | One-click Pinokio launcher for Musubi's built-in Gradio GUI. Interactive counterpart to the MCP server. |
| [hoodtronik/klippbok-mcp](https://github.com/hoodtronik/klippbok-mcp) | MCP server wrapping the Klippbok video dataset curation CLI. **Style reference** for this server's FastMCP patterns, `CommandResult` dataclass, env discipline, AGENTS.md convention. |
| [alvdansen/klippbok](https://github.com/alvdansen/klippbok) (upstream) | The dataset curation CLI klippbok-mcp wraps. A Pinokio companion (`klippbok-pinokio`) is a plausible future addition. |

**Why the split:** The developer wants both interactive (Pinokio +
GUI) and agent-driven (MCP servers) entry points into the same
underlying training stack. Any workflow — human-paced exploration OR
LLM-scripted pipelines — should be possible without switching tools.

**Naming conventions:**
- Dot-form (`musubi-tuner.pinokio`) for Pinokio launchers
- Dash-form (`musubi-mcp`, `klippbok-mcp`) for MCP servers
- Python package names: `musubi_mcp`, `klippbok_mcp` (underscore, not dash)

---

## This repo's design pillars

See [AGENTS.md](AGENTS.md) for the full set. The ones that matter
most for resumption work:

1. **Never import Musubi Tuner internals.** The server shells out
   via `asyncio.create_subprocess_exec`. This keeps us decoupled from
   upstream API churn and supports separate-venv installs (the common
   case — torch+CUDA is a heavy per-GPU install).
2. **All Musubi scripts run with cwd = `MUSUBI_TUNER_DIR`.** They use
   relative imports from their package. The runner sets this on
   every call.
3. **Training scripts go through `accelerate launch`**, not plain
   `python`. Use `run_musubi_training` (not `run_musubi`).
4. **Env variables are read on every call.** The developer can flip
   `.env` between calls without restarting the server.
5. **The FastMCP tool layer uses native Python type hints** — no
   Pydantic models in signatures. Literals for enums. Tools return
   plain `dict[str, Any]`. Never raise from tool handlers; return
   `{"ok": False, "error": ...}` instead.
6. **`docs/cli_help.txt` is the source of truth for CLI flag names.**
   Every tool's parameter schema is audited against it. Regenerate
   with `PYTHONUTF8=1` on Windows (Musubi's argparse help strings
   include Japanese characters that crash cp1252).

---

## Current state

See [HANDOFF.md](HANDOFF.md) for the snapshot (shipped surface, test
count, live architectures, placeholder architectures, gotchas).

See [ROADMAP.md](ROADMAP.md) for planned expansions (architecture
activations starting with HunyuanVideo, streaming logs via
`ctx.info` / `report_progress`, `scripts/regen_cli_help.sh`,
additional prompts, CI).

---

## When the developer references "the other repo"

| Phrase | Usually means |
|---|---|
| "the launcher" / "the Pinokio one" | [musubi-tuner.pinokio](https://github.com/hoodtronik/musubi-tuner.pinokio) |
| "the MCP one" / "this server" | This repo, `musubi-mcp` |
| "klippbok" (no suffix) | Ambiguous; ask whether they mean the upstream CLI, the MCP server, or the (potential) Pinokio launcher |
| "the upstream" / "Musubi" | [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) |
