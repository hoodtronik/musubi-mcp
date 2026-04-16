"""Architecture-agnostic constants and script name helpers.

The script inventory was dumped from a fresh clone of
https://github.com/kohya-ss/musubi-tuner (v0.2.15) in April 2026. If a later
version adds a new architecture or splits a script, update both this module
AND ``architectures.py``, then re-run ``docs/cli_help.txt`` generation.
"""
from __future__ import annotations

from typing import Final, Literal

# ---------------------------------------------------------------------------
# Accelerate launch defaults
# ---------------------------------------------------------------------------

# CLAUDE-NOTE: 1 CPU thread per process matches kohya-ss's documented invocation
# across every training doc. It only affects Python-side thread hinting — GPU
# work is unaffected. Users can override via an explicit launcher flag if they
# know what they want.
DEFAULT_CPU_THREADS_PER_PROCESS: Final[int] = 1

# CLAUDE-NOTE: bf16 is the documented recommendation for every supported arch
# (Wan, FLUX.2, Z-Image). fp16 is allowed but loses precision on some DiT
# blocks. fp32 is a memory fire. Leaving this as a default keeps tool calls
# short — callers can still override explicitly.
DEFAULT_MIXED_PRECISION: Final[str] = "bf16"

# ---------------------------------------------------------------------------
# Timeouts (seconds). Training dwarfs everything else; cache runs rarely
# exceed an hour even on large datasets; generation is GPU-bound and usually
# seconds-to-minutes.
# ---------------------------------------------------------------------------

TIMEOUT_CACHE_SECONDS: Final[float] = 3600.0            # 1 hour
TIMEOUT_GENERATE_SECONDS: Final[float] = 1800.0         # 30 min (video can be slow)
TIMEOUT_TRAIN_SECONDS: Final[float] = 86400.0           # 24 hours
TIMEOUT_UTILITY_SECONDS: Final[float] = 900.0           # 15 min

# ---------------------------------------------------------------------------
# Architecture types
# ---------------------------------------------------------------------------

ArchType = Literal["image", "video"]
"""Whether an architecture produces images or videos. Drives which generate
script (``_generate_image.py`` vs ``_generate_video.py``) is routed to."""

NetworkType = Literal["lora", "loha", "lokr"]
"""Network modules supported across architectures. LoHa/LoKr are recent
additions in Musubi Tuner and work through the same ``--network_module``
flag with different module paths (``networks.lora_wan`` etc. — the 'lora_'
prefix is historical and covers all three network types)."""

# ---------------------------------------------------------------------------
# Utility script names (not tied to any architecture)
# ---------------------------------------------------------------------------

UTILITY_SCRIPTS: Final[dict[str, str]] = {
    "convert_lora": "convert_lora.py",
    "merge_lora": "merge_lora.py",
    "ema_merge": "lora_post_hoc_ema.py",
    "qwen_extract_lora": "qwen_extract_lora.py",
    "caption_images": "caption_images_by_qwen_vl.py",
}
