"""Musubi Tuner dataset TOML generation and validation.

Unlike every other module in this server, these helpers do NOT shell out to
Musubi Tuner — they read and write TOML files directly, using only the
rules documented in ``musubi-tuner/docs/dataset_config.md``. This is the
right trade-off because:

  - Musubi has no CLI for generating a dataset TOML; users hand-edit them.
  - Validation errors are much more useful if surfaced before caching
    starts (caching takes minutes to hours; a typo in num_repeats that
    only shows up after loading 10k images is painful).
  - The format is small and stable.

Schema reference (from Musubi's dataset_config.md):

    [general]
    resolution          = [W, H]    # default [960, 544]
    caption_extension   = ".txt"
    batch_size          = 1
    num_repeats         = 1
    enable_bucket       = false
    bucket_no_upscale   = false

    [[datasets]]
    # Image dataset:
    image_directory     = "..."      # OR image_jsonl_file
    image_jsonl_file    = "..."
    cache_directory     = "..."      # optional (defaults to image_directory)
    control_directory   = "..."      # optional (FLUX.1 Kontext / FLUX.2 / Qwen-Image-Edit)
    control_resolution  = [W, H]
    no_resize_control   = false
    multiple_target     = false      # Qwen-Image-Layered only

    # Video dataset:
    video_directory     = "..."      # OR video_jsonl_file
    video_jsonl_file    = "..."
    cache_directory     = "..."      # required for metadata JSONL
    target_frames       = [1, 25, 45]  # MUST be N*4+1 for Hunyuan / Wan2.1
    frame_extraction    = "head"       # head | chunk | slide | uniform | full
    frame_stride        = 1
    frame_sample        = 1
    max_frames          = 129
    source_fps          = 30.0         # note: decimal, not int

    # Any of the above fields may also appear under [general] as defaults.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import tomli_w

# CLAUDE-NOTE: tomllib is stdlib as of Python 3.11; tomli is the backport.
# We target 3.10+ so we import conditionally. Both expose ``loads()`` with
# the same bytes-in signature, so the call site doesn't care which is live.
if sys.version_info >= (3, 11):
    import tomllib as _toml_reader
else:
    import tomli as _toml_reader


_FRAME_EXTRACTION_MODES = ("head", "chunk", "slide", "uniform", "full")


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Outcome of validating a dataset config or directory.

    ``errors`` are hard blockers (training/caching will fail). ``warnings``
    are things the user may have done by accident (e.g. ``num_repeats=1``
    on a tiny dataset). ``info`` is neutral observations (resolution
    distribution, image count).
    """

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }


# ---------------------------------------------------------------------------
# TOML generation
# ---------------------------------------------------------------------------


def build_dataset_toml(
    *,
    # Either image or video path — exactly one is required.
    image_directory: Optional[str] = None,
    video_directory: Optional[str] = None,
    image_jsonl_file: Optional[str] = None,
    video_jsonl_file: Optional[str] = None,
    # Cache dir (optional for image, required for video+jsonl).
    cache_directory: Optional[str] = None,
    # [general] fields (applied at the top level).
    resolution: tuple[int, int] = (960, 544),
    caption_extension: str = ".txt",
    batch_size: int = 1,
    num_repeats: int = 1,
    enable_bucket: bool = False,
    bucket_no_upscale: bool = False,
    # Video-only.
    target_frames: Optional[Iterable[int]] = None,
    frame_extraction: Optional[str] = None,
    frame_stride: Optional[int] = None,
    frame_sample: Optional[int] = None,
    max_frames: Optional[int] = None,
    source_fps: Optional[float] = None,
    # Control image/video (FLUX.1 Kontext, FLUX.2, Qwen-Image-Edit, Wan Fun-Control).
    control_directory: Optional[str] = None,
    control_resolution: Optional[tuple[int, int]] = None,
    no_resize_control: bool = False,
    # Qwen-Image-Layered only.
    multiple_target: bool = False,
) -> dict[str, Any]:
    """Build a dataset config dict ready to be serialized with ``dumps``.

    Raises ``ValueError`` for structural problems the caller must fix
    before writing the file (e.g. providing both image and video sources,
    or neither).
    """
    sources = [
        p
        for p in (image_directory, video_directory, image_jsonl_file, video_jsonl_file)
        if p
    ]
    if len(sources) == 0:
        raise ValueError(
            "provide one of image_directory, video_directory, image_jsonl_file, "
            "or video_jsonl_file"
        )
    if len(sources) > 1:
        raise ValueError(
            "provide exactly ONE source — mixing image and video datasets in a "
            "single [[datasets]] block is not supported"
        )

    is_video = bool(video_directory or video_jsonl_file)
    uses_jsonl = bool(image_jsonl_file or video_jsonl_file)

    if uses_jsonl and is_video and not cache_directory:
        raise ValueError(
            "cache_directory is required when using video_jsonl_file "
            "(Musubi writes cached metadata next to the source, and jsonl "
            "sources have no implicit directory)"
        )

    general: dict[str, Any] = {
        "resolution": list(resolution),
        "caption_extension": caption_extension,
        "batch_size": int(batch_size),
        "num_repeats": int(num_repeats),
        "enable_bucket": bool(enable_bucket),
        "bucket_no_upscale": bool(bucket_no_upscale),
    }

    dataset: dict[str, Any] = {}
    if image_directory:
        dataset["image_directory"] = image_directory
    if video_directory:
        dataset["video_directory"] = video_directory
    if image_jsonl_file:
        dataset["image_jsonl_file"] = image_jsonl_file
    if video_jsonl_file:
        dataset["video_jsonl_file"] = video_jsonl_file
    if cache_directory:
        dataset["cache_directory"] = cache_directory

    # Video-specific fields.
    if is_video:
        if target_frames is None:
            raise ValueError("target_frames is required for video datasets")
        frames = [int(f) for f in target_frames]
        if not frames:
            raise ValueError("target_frames must contain at least one frame count")
        dataset["target_frames"] = frames
        if frame_extraction is not None:
            if frame_extraction not in _FRAME_EXTRACTION_MODES:
                raise ValueError(
                    f"frame_extraction must be one of {_FRAME_EXTRACTION_MODES}, "
                    f"got {frame_extraction!r}"
                )
            dataset["frame_extraction"] = frame_extraction
        if frame_stride is not None:
            dataset["frame_stride"] = int(frame_stride)
        if frame_sample is not None:
            dataset["frame_sample"] = int(frame_sample)
        if max_frames is not None:
            dataset["max_frames"] = int(max_frames)
        if source_fps is not None:
            dataset["source_fps"] = float(source_fps)

    # Control image/video (optional for any architecture that supports it).
    if control_directory:
        dataset["control_directory"] = control_directory
    if control_resolution is not None:
        dataset["control_resolution"] = list(control_resolution)
    if no_resize_control:
        dataset["no_resize_control"] = True
    if multiple_target:
        dataset["multiple_target"] = True

    return {"general": general, "datasets": [dataset]}


def dumps(config: dict[str, Any]) -> str:
    """Serialize a config dict to TOML text."""
    return tomli_w.dumps(config)


def write_dataset_toml(config: dict[str, Any], path: str) -> None:
    """Write a config dict to ``path`` as TOML. Creates parent dirs."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(config, f)


def loads(toml_text: str) -> dict[str, Any]:
    """Parse TOML text into a config dict.

    Uses stdlib ``tomllib`` on 3.11+, falls back to ``tomli`` on 3.10.
    """
    return _toml_reader.loads(toml_text)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_4n_plus_1(n: int) -> bool:
    """True if ``n`` satisfies ``n == 4k + 1`` for some non-negative integer k."""
    return n >= 1 and (n - 1) % 4 == 0


def validate_dataset_config(
    config: dict[str, Any],
    *,
    enforce_4n_plus_1: bool = False,
) -> ValidationResult:
    """Validate a parsed dataset config against Musubi's documented rules.

    ``enforce_4n_plus_1``: when True, video ``target_frames`` must satisfy
    ``N*4+1``. This is required for HunyuanVideo, Wan2.1, and (softly) FramePack;
    other video architectures may have different constraints. The tool layer
    should set this based on the chosen architecture.
    """
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {}

    if "datasets" not in config or not isinstance(config["datasets"], list):
        return ValidationResult(
            ok=False,
            errors=["config is missing a top-level [[datasets]] array"],
        )
    if not config["datasets"]:
        return ValidationResult(
            ok=False,
            errors=["[[datasets]] is empty — add at least one dataset block"],
        )

    for idx, ds in enumerate(config["datasets"]):
        prefix = f"datasets[{idx}]"
        if not isinstance(ds, dict):
            errors.append(f"{prefix}: not a table")
            continue

        sources = [
            k
            for k in ("image_directory", "video_directory", "image_jsonl_file", "video_jsonl_file")
            if ds.get(k)
        ]
        if len(sources) == 0:
            errors.append(
                f"{prefix}: needs one of image_directory, video_directory, "
                f"image_jsonl_file, video_jsonl_file"
            )
            continue
        if len(sources) > 1:
            errors.append(
                f"{prefix}: has multiple sources {sources} — pick exactly one"
            )

        is_video = bool(ds.get("video_directory") or ds.get("video_jsonl_file"))

        if is_video:
            frames = ds.get("target_frames")
            if frames is None:
                errors.append(f"{prefix}: video dataset missing target_frames")
            elif not isinstance(frames, list) or not frames:
                errors.append(f"{prefix}: target_frames must be a non-empty list")
            else:
                bad = [f for f in frames if not isinstance(f, int) or f < 1]
                if bad:
                    errors.append(f"{prefix}: invalid target_frames {bad}")
                if enforce_4n_plus_1:
                    violators = [f for f in frames if isinstance(f, int) and not _is_4n_plus_1(f)]
                    if violators:
                        errors.append(
                            f"{prefix}: target_frames {violators} violate the "
                            f"N*4+1 rule (required by HunyuanVideo / Wan2.1 / FramePack). "
                            f"Valid values: 1, 5, 9, 13, ..., 77, 81."
                        )
            mode = ds.get("frame_extraction")
            if mode is not None and mode not in _FRAME_EXTRACTION_MODES:
                errors.append(
                    f"{prefix}: frame_extraction={mode!r}, must be one of "
                    f"{_FRAME_EXTRACTION_MODES}"
                )

        # num_repeats sanity.
        nr = ds.get("num_repeats", config.get("general", {}).get("num_repeats", 1))
        if isinstance(nr, int) and nr < 1:
            errors.append(f"{prefix}: num_repeats must be >= 1 (got {nr})")

    # [general] sanity.
    general = config.get("general", {})
    res = general.get("resolution")
    if res is not None and (not isinstance(res, list) or len(res) != 2):
        errors.append("[general].resolution must be a list of two integers [W, H]")
    bs = general.get("batch_size", 1)
    if isinstance(bs, int) and bs < 1:
        errors.append(f"[general].batch_size must be >= 1 (got {bs})")

    info["dataset_count"] = len(config.get("datasets", []))
    return ValidationResult(
        ok=not errors, errors=errors, warnings=warnings, info=info
    )


def validate_dataset_toml_file(
    path: str,
    *,
    enforce_4n_plus_1: bool = False,
) -> ValidationResult:
    """Load a TOML file from disk and run ``validate_dataset_config``."""
    if not os.path.isfile(path):
        return ValidationResult(
            ok=False,
            errors=[f"file not found: {path}"],
        )
    try:
        with open(path, "rb") as f:
            config = _toml_reader.load(f)
    except Exception as exc:  # noqa: BLE001 — tomllib raises TOMLDecodeError on 3.11+, tomli's flavour differs
        return ValidationResult(
            ok=False,
            errors=[f"failed to parse TOML at {path}: {exc}"],
        )
    return validate_dataset_config(config, enforce_4n_plus_1=enforce_4n_plus_1)
