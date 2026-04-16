"""Serve Musubi Tuner's architecture docs as MCP resources.

Each resource URI (``musubi://docs/{name}``) maps to a file under
``<MUSUBI_TUNER_DIR>/docs/``. Agents read these to ground architecture
choices before planning a training run, so the resource responses are
raw markdown — no transformation.

If MUSUBI_TUNER_DIR is unset or the doc file is missing, the resource
returns a short error marker rather than raising — resource reads in
MCP aren't supposed to throw for missing content.
"""
from __future__ import annotations

import os
from typing import Optional

from .runner import musubi_tuner_dir


# CLAUDE-NOTE: Keep this list in lockstep with the files actually shipping
# under musubi-tuner/docs/. Adding a new arch doc upstream means adding a
# line here; the server registers a resource per entry at startup.
DOC_FILES: dict[str, str] = {
    "wan": "wan.md",
    "wan_1f": "wan_1f.md",
    "flux_2": "flux_2.md",
    "flux_kontext": "flux_kontext.md",
    "zimage": "zimage.md",
    "hunyuan_video": "hunyuan_video.md",
    "hunyuan_video_1_5": "hunyuan_video_1_5.md",
    "framepack": "framepack.md",
    "framepack_1f": "framepack_1f.md",
    "qwen_image": "qwen_image.md",
    "kandinsky5": "kandinsky5.md",
    "dataset_config": "dataset_config.md",
    "advanced_config": "advanced_config.md",
    "loha_lokr": "loha_lokr.md",
    "torch_compile": "torch_compile.md",
    "sampling_during_training": "sampling_during_training.md",
    "tools": "tools.md",
}


def doc_uri(name: str) -> str:
    """Canonical URI for a Musubi doc resource."""
    return f"musubi://docs/{name}"


def read_doc(name: str) -> str:
    """Return the raw markdown content of a named doc, or an error marker."""
    filename = DOC_FILES.get(name)
    if filename is None:
        return f"[musubi-mcp] unknown doc resource: {name!r}"

    root = musubi_tuner_dir()
    if not root:
        return (
            "[musubi-mcp] MUSUBI_TUNER_DIR is unset — cannot serve "
            f"docs/{filename}. Set the env var to your Musubi Tuner checkout."
        )

    path = os.path.join(root, "docs", filename)
    if not os.path.isfile(path):
        return (
            f"[musubi-mcp] docs/{filename} not found under {root}. "
            "This Musubi Tuner version may not ship that doc."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as exc:
        return f"[musubi-mcp] failed to read docs/{filename}: {exc}"


def all_doc_names() -> list[str]:
    return list(DOC_FILES.keys())
