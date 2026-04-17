"""Serve the community LoRA training knowledge base as MCP resources.

URIs:
    ``knowledge://index``         — table of contents + confidence notes.
    ``knowledge://<name>``        — one resource per curated markdown file
                                    (e.g. ``knowledge://wan22_training``).

File discovery priority:

  1. ``$KNOWLEDGE_BASE_DIR`` env var. **Live-sync** — if the user
     maintains a central knowledge base (e.g.
     ``F:/__PROJECTS/LoRAKnowledgeBase/reviewed/``) and points this var
     at it, edits are visible on the next tool call with no server
     restart needed. All three knowledge-exposing servers (klippbok-mcp,
     musubi-mcp, ltx-trainer-mcp) read the same env var.

  2. Bundled copy under ``<this package>/knowledge_files/`` — shipped
     with every pip install / git clone so a fresh install has the
     reference material available even with no env setup.

Files flagged ``INSUFFICIENT DATA`` in their contents are served
**verbatim**. The orchestration agent is expected to see the flag and
ask the user to fill in the gap rather than guess. Do not filter.
"""
from __future__ import annotations

import os
from pathlib import Path


_PKG_DIR = Path(__file__).resolve().parent
_BUNDLED_DIR = _PKG_DIR / "knowledge_files"


def _source_dir() -> Path:
    """Current knowledge-file source — env override wins, else bundled copy."""
    override = os.environ.get("KNOWLEDGE_BASE_DIR")
    if override:
        p = Path(override)
        if p.exists() and p.is_dir():
            return p
    return _BUNDLED_DIR


def _uri_name(filename: str) -> str:
    """``wan22_training.md`` -> ``wan22_training``. URI-path-safe name."""
    return filename[:-3] if filename.endswith(".md") else filename


def knowledge_uri(name: str) -> str:
    return f"knowledge://{name}"


def all_knowledge_names() -> list[str]:
    """Enumerate available knowledge resources (no .md suffix), sorted.

    Called once at server startup to register one @mcp.resource per file.
    If the source dir is missing or empty, returns an empty list and the
    server simply won't have any knowledge resources (the index resource
    still serves and explains the problem).
    """
    src = _source_dir()
    if not src.exists() or not src.is_dir():
        return []
    return sorted(_uri_name(p.name) for p in src.glob("*.md"))


def read_knowledge(name: str) -> str:
    """Return the raw markdown for a knowledge resource, or an error marker.

    Error markers (not exceptions) — MCP resource reads shouldn't raise.
    Rereads the source dir each call so env-var changes take effect
    without restart.
    """
    src = _source_dir()
    candidate = src / f"{name}.md"
    if not candidate.is_file():
        available = ", ".join(all_knowledge_names()) or "(none)"
        return (
            f"[musubi-mcp] unknown knowledge resource: {name!r}. "
            f"Available: {available}"
        )
    try:
        return candidate.read_text(encoding="utf-8")
    except OSError as exc:
        return f"[musubi-mcp] failed to read knowledge/{name}.md: {exc}"


def knowledge_index() -> str:
    """Markdown index served at ``knowledge://index``.

    Lists available resources and reminds the agent what the confidence
    ratings + INSUFFICIENT DATA flags mean.
    """
    names = all_knowledge_names()
    src = _source_dir()
    override = os.environ.get("KNOWLEDGE_BASE_DIR")
    if override and Path(override).exists():
        source_note = f"live-sync from `$KNOWLEDGE_BASE_DIR = {src}`"
    else:
        source_note = "bundled copy shipped with musubi-mcp"

    if not names:
        return (
            "# LoRA training knowledge base — not available\n\n"
            f"No .md files found under `{src}`.\n\n"
            "Either reinstall the package to restore the bundled copy, or "
            "set `KNOWLEDGE_BASE_DIR` to a folder of .md files."
        )

    lines = [
        "# LoRA training knowledge base",
        "",
        f"Source: {source_note} ({len(names)} file{'s' if len(names) != 1 else ''})",
        "",
        "## Available resources",
        "",
    ]
    lines.extend(f"- `knowledge://{n}`" for n in names)
    lines += [
        "",
        "## How to use this",
        "",
        "Read each resource as raw markdown. Every file starts with a",
        "confidence header — **HIGH / MEDIUM / LOW** — plus a source summary.",
        "",
        "### When `INSUFFICIENT DATA` appears",
        "",
        "Some entries in a file may be flagged `INSUFFICIENT DATA`. This flag",
        "is load-bearing: it means community consensus did not exist at",
        "research time. **Ask the user** for the missing value rather than",
        "guessing. Don't paper over the flag — the user knows their machine",
        "and training intent better than a guess would.",
        "",
        "### Typical reading order for a new training run",
        "",
        "1. `knowledge://<arch>_training` — per-architecture hyperparameters.",
        "2. `knowledge://dataset_quality` — captioning / count / resolution.",
        "3. `knowledge://hardware_profiles` — rank / blocks_to_swap / fp8 for the target VRAM.",
        "4. `knowledge://common_failures` — when loss misbehaves mid-run.",
        "",
        "5. `knowledge://sources` — audit trail if the user asks where a",
        "   recommendation came from.",
    ]
    return "\n".join(lines)
