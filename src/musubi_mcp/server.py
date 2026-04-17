"""FastMCP server for Musubi Tuner.

This module is the user-visible surface of musubi-mcp. Three tools are
shipped in this first checkpoint:

    - musubi_check_installation   : sanity-check the Musubi Tuner install.
    - musubi_list_architectures   : capability report for every arch.
    - musubi_cache_latents        : pre-cache VAE latents (step 1 of 3 in
                                    every training pipeline).

More tools (cache_text_encoder, train, generate, convert_lora, merge_lora,
dataset tooling, prompts) land after the first checkpoint review.

Design follows the sibling ``klippbok-mcp``:

- FastMCP high-level API; tools are ``@mcp.tool(...)`` decorated async defs.
- Native Python type hints for parameters — no Pydantic models in the
  tool signatures. Literal types for enums.
- Tools return plain ``dict`` results (JSON-serializable). Subprocess
  failures surface as ``{"ok": False, "error": ..., "result": {...}}``
  rather than raising.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from . import __version__
from .architectures import (
    all_architectures,
    architecture_names,
    get_architecture,
    live_architectures,
)
from .constants import (
    DEFAULT_MIXED_PRECISION,
    TIMEOUT_CACHE_SECONDS,
    TIMEOUT_GENERATE_SECONDS,
    TIMEOUT_TRAIN_SECONDS,
    TIMEOUT_UTILITY_SECONDS,
)
from .dataset_config import (
    build_dataset_toml,
    dumps as toml_dumps,
    validate_dataset_toml_file,
    write_dataset_toml,
)
from .resources import DOC_FILES, doc_uri, read_doc
from .knowledge import (
    all_knowledge_names,
    knowledge_index,
    knowledge_uri,
    read_knowledge,
)
from .runner import (
    accelerate_executable,
    musubi_tuner_dir,
    python_executable,
    run_command,
    run_musubi,
    run_musubi_training,
)


SERVER_INSTRUCTIONS = """\
Wraps the Musubi Tuner LoRA training toolkit (https://github.com/kohya-ss/musubi-tuner).

Typical workflow for training a LoRA:
  1. musubi_check_installation           — verify env is ready
  2. musubi_list_architectures           — pick an architecture
  3. (build a dataset TOML — tool coming soon)
  4. musubi_cache_latents                — pre-cache VAE latents
  5. musubi_cache_text_encoder (soon)    — pre-cache text encoder outputs
  6. musubi_train (soon)                 — run the accelerate-launched training
  7. musubi_generate (soon)              — test the trained LoRA

Configure via env:
  MUSUBI_TUNER_DIR  = absolute path to the Musubi Tuner checkout  (required)
  MUSUBI_PYTHON     = python inside Musubi's venv                 (recommended)
  MUSUBI_ACCELERATE = accelerate binary                           (auto-discovered)
"""


mcp = FastMCP("musubi", instructions=SERVER_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# Tool: musubi_check_installation
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Check whether Musubi Tuner is installed and reachable. Reports "
        "Python interpreter, torch + CUDA availability, accelerate config "
        "status, which architectures are live vs placeholder, and which "
        "env vars are set. Call this first when debugging setup issues."
    ),
)
async def musubi_check_installation() -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": True,
        "version": __version__,
        "env": {
            "MUSUBI_TUNER_DIR": os.environ.get("MUSUBI_TUNER_DIR"),
            "MUSUBI_PYTHON": os.environ.get("MUSUBI_PYTHON"),
            "MUSUBI_ACCELERATE": os.environ.get("MUSUBI_ACCELERATE"),
            "GEMINI_API_KEY_set": bool(os.environ.get("GEMINI_API_KEY")),
            "REPLICATE_API_TOKEN_set": bool(os.environ.get("REPLICATE_API_TOKEN")),
            "HF_TOKEN_set": bool(os.environ.get("HF_TOKEN")),
        },
        "architectures": {
            "live": [a.name for a in live_architectures()],
            "placeholder": [a.name for a in all_architectures() if not a.live],
        },
        "issues": [],
        "suggestions": [],
    }

    tuner_dir = musubi_tuner_dir()
    report["resolved"] = {
        "musubi_tuner_dir": tuner_dir,
        "python_executable": python_executable(),
        "accelerate_executable": accelerate_executable(),
    }

    # --- MUSUBI_TUNER_DIR ---
    if not tuner_dir:
        report["ok"] = False
        report["issues"].append(
            "MUSUBI_TUNER_DIR is not set — every tool that invokes Musubi "
            "will fail."
        )
        report["suggestions"].append(
            "Set MUSUBI_TUNER_DIR to the absolute path of your Musubi "
            "Tuner checkout. See README § Configuration."
        )
    elif not os.path.isdir(tuner_dir):
        report["ok"] = False
        report["issues"].append(
            f"MUSUBI_TUNER_DIR points to a path that does not exist: {tuner_dir}"
        )
    else:
        # Spot-check a well-known script so we catch pointer-to-wrong-repo errors.
        sentinel = os.path.join(tuner_dir, "wan_train_network.py")
        if not os.path.isfile(sentinel):
            report["ok"] = False
            report["issues"].append(
                f"MUSUBI_TUNER_DIR={tuner_dir} does not contain "
                f"wan_train_network.py — is this really a Musubi Tuner checkout?"
            )

    # --- Python executable + versions ---
    py = python_executable()
    py_probe = await run_command(
        [py, "--version"], timeout=15.0, mode="python",
    )
    report["python_version"] = (
        py_probe.stdout.strip() or py_probe.stderr.strip()
    ) if py_probe.succeeded else f"(could not run {py}: {py_probe.stderr.strip()})"
    if not py_probe.succeeded:
        report["ok"] = False
        report["issues"].append(
            f"MUSUBI_PYTHON interpreter not runnable: {py_probe.stderr.strip()}"
        )
        report["suggestions"].append(
            "Point MUSUBI_PYTHON at Musubi Tuner's venv Python "
            "(e.g. /path/to/musubi-tuner/.venv/Scripts/python.exe on Windows)."
        )

    # --- torch + CUDA probe ---
    torch_probe = await run_command(
        [
            py,
            "-c",
            (
                "import json, sys\n"
                "info = {'python': sys.version.split()[0]}\n"
                "try:\n"
                "    import torch\n"
                "    info['torch'] = torch.__version__\n"
                "    info['cuda_available'] = bool(torch.cuda.is_available())\n"
                "    info['cuda_version'] = torch.version.cuda\n"
                "    info['device_count'] = torch.cuda.device_count()\n"
                "    info['device_name'] = (\n"
                "        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None\n"
                "    )\n"
                "except Exception as exc:\n"
                "    info['torch_import_error'] = repr(exc)\n"
                "try:\n"
                "    import accelerate as _a\n"
                "    info['accelerate'] = _a.__version__\n"
                "except Exception as exc:\n"
                "    info['accelerate_import_error'] = repr(exc)\n"
                "print(json.dumps(info))\n"
            ),
        ],
        timeout=60.0,
        mode="python",
    )
    if torch_probe.succeeded:
        import json as _json

        try:
            probed = _json.loads(torch_probe.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            probed = {"raw": torch_probe.stdout[-400:]}
        report["torch"] = probed
        if not probed.get("cuda_available"):
            report["suggestions"].append(
                "torch reports cuda_available=False — GPU training will fail. "
                "Reinstall Musubi with the right cu12x extra for your CUDA driver."
            )
        if "accelerate_import_error" in probed:
            report["ok"] = False
            report["issues"].append(
                f"accelerate not importable in MUSUBI_PYTHON: {probed['accelerate_import_error']}"
            )
    else:
        report["torch"] = {"probe_failed": torch_probe.stderr[-400:]}

    # --- accelerate binary + config ---
    accel = accelerate_executable()
    report["resolved"]["accelerate_executable"] = accel
    if not (os.path.isfile(accel) or shutil.which(accel)):
        report["ok"] = False
        report["issues"].append(
            f"accelerate binary not found at {accel!r}. Training tools will fail."
        )
        report["suggestions"].append(
            "Install Musubi's deps (uv sync --extra cu128 inside the Musubi "
            "checkout), or set MUSUBI_ACCELERATE to the absolute path."
        )
    else:
        accel_config_probe = await run_command(
            [accel, "env"], timeout=30.0, mode="accelerate",
        )
        # CLAUDE-NOTE: `accelerate env` prints "Not found" or similar when
        # no config exists. We don't hard-fail — many simple single-GPU
        # setups can run with defaults — but we surface the status so the
        # user knows to run `accelerate config` if multi-GPU is intended.
        report["accelerate_env"] = {
            "stdout_tail": accel_config_probe.stdout[-1000:],
            "ok": accel_config_probe.succeeded,
        }
        if not accel_config_probe.succeeded:
            report["suggestions"].append(
                "`accelerate env` failed — run `accelerate config` once inside "
                "Musubi's venv to create a default config."
            )

    return report


# ---------------------------------------------------------------------------
# Tool: musubi_list_architectures
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "List every registered architecture (wan, flux_2, zimage, ...) "
        "with its capabilities: which scripts exist, which text-encoder "
        "flags it expects, the LoRA network module to use, and any "
        "`--task` or `--model_version` enum values. Architectures marked "
        "live are fully wired and tested; placeholders are registered for "
        "discovery but their tool calls will return a "
        "'not yet implemented' error."
    ),
)
async def musubi_list_architectures() -> dict[str, Any]:
    return {
        "count": len(all_architectures()),
        "live_count": len(live_architectures()),
        "architectures": [a.to_public_dict() for a in all_architectures()],
    }


# ---------------------------------------------------------------------------
# Tool: musubi_cache_latents
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Pre-cache VAE latents for a dataset. This is step 1 of the 3-step "
        "Musubi training pipeline (cache latents → cache text encoder → "
        "train). Routes to the correct per-architecture cache script "
        "(wan_cache_latents.py, flux_2_cache_latents.py, "
        "zimage_cache_latents.py). Pass `vae` to override the VAE path; "
        "otherwise the path set in the dataset TOML is used. For FLUX.2, "
        "`model_version` is REQUIRED. Wan I2V training needs `clip` set."
    ),
)
async def musubi_cache_latents(
    architecture: str,
    dataset_config: str,
    vae: Optional[str] = None,
    vae_dtype: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    skip_existing: bool = False,
    keep_cache: bool = False,
    disable_cudnn_backend: bool = False,
    # FLUX.2 specific
    model_version: Optional[str] = None,
    # Wan specific
    i2v: bool = False,
    clip: Optional[str] = None,
    one_frame: bool = False,
    vae_cache_cpu: bool = False,
    # Escape hatch for rare flags not exposed above.
    extra_args: Optional[list[str]] = None,
) -> dict[str, Any]:
    arch = get_architecture(architecture)
    if arch is None:
        return {
            "ok": False,
            "error": (
                f"unknown architecture {architecture!r}. "
                f"Known: {architecture_names()}"
            ),
        }
    if not arch.live:
        return {
            "ok": False,
            "error": (
                f"architecture {architecture!r} is registered as a placeholder "
                f"but not yet wired up. Live architectures: "
                f"{[a.name for a in live_architectures()]}"
            ),
        }
    if not arch.has_cache_latents():
        return {
            "ok": False,
            "error": f"architecture {architecture!r} has no cache_latents script",
        }
    if not os.path.isfile(dataset_config):
        return {
            "ok": False,
            "error": f"dataset_config not found: {dataset_config}",
        }

    # FLUX.2 requires --model_version at cache time.
    if arch.name == "flux_2":
        if not model_version:
            return {
                "ok": False,
                "error": (
                    "flux_2 cache_latents requires model_version — one of "
                    f"{arch.model_versions}"
                ),
            }
        if arch.model_versions and model_version not in arch.model_versions:
            return {
                "ok": False,
                "error": (
                    f"invalid model_version {model_version!r} for flux_2. "
                    f"Choose one of {list(arch.model_versions)}"
                ),
            }

    # --- Build argv ---
    args: list[str] = ["--dataset_config", dataset_config]
    if vae:
        args += [arch.vae_arg, vae]
    if vae_dtype:
        args += ["--vae_dtype", vae_dtype]
    if device:
        args += ["--device", device]
    if batch_size is not None:
        args += ["--batch_size", str(batch_size)]
    if num_workers is not None:
        args += ["--num_workers", str(num_workers)]
    if skip_existing:
        args.append("--skip_existing")
    if keep_cache:
        args.append("--keep_cache")
    if disable_cudnn_backend:
        args.append("--disable_cudnn_backend")

    # Architecture-specific.
    if arch.name == "flux_2":
        args += ["--model_version", model_version]  # type: ignore[list-item]
    if arch.name == "wan":
        if i2v:
            args.append("--i2v")
        if clip:
            args += ["--clip", clip]
        if one_frame:
            args.append("--one_frame")
        if vae_cache_cpu:
            args.append("--vae_cache_cpu")

    if extra_args:
        args.extend(extra_args)

    # --- Run ---
    extra_env = {}
    for passthrough in ("HF_TOKEN",):
        if os.environ.get(passthrough):
            extra_env[passthrough] = os.environ[passthrough]  # type: ignore[assignment]

    result = await run_musubi(
        arch.cache_latents_script,  # type: ignore[arg-type]
        args,
        timeout=TIMEOUT_CACHE_SECONDS,
        extra_env=extra_env or None,
    )
    return {
        "ok": result.succeeded,
        "architecture": arch.name,
        "script": arch.cache_latents_script,
        "summary": result.short_summary(),
        "result": result.to_dict(),
    }


# ---------------------------------------------------------------------------
# Tool: musubi_cache_text_encoder
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Pre-cache text encoder outputs for a dataset. Step 2 of 3 in the "
        "Musubi training pipeline. Routes to "
        "{arch}_cache_text_encoder_outputs.py. Per architecture: Wan needs "
        "`t5` (T5 UMT5-XXL) and optionally `clip` (required for Wan2.1 I2V). "
        "FLUX.2 needs `text_encoder` and `model_version`. Z-Image needs "
        "`text_encoder` (Qwen3). The fp8_* bool flags quantize the encoder "
        "to fp8 to save VRAM (fp8_text_encoder is NOT available for FLUX.2 "
        "dev, and Z-Image uses `fp8_llm` instead)."
    ),
)
async def musubi_cache_text_encoder(
    architecture: str,
    dataset_config: str,
    # Encoder paths — one of these is required depending on arch.
    t5: Optional[str] = None,               # Wan
    text_encoder: Optional[str] = None,     # FLUX.2, Z-Image
    clip: Optional[str] = None,             # Wan2.1 I2V only
    # fp8 knobs (names differ by arch — we normalise in the tool layer).
    fp8_t5: bool = False,                   # Wan
    fp8_text_encoder: bool = False,         # FLUX.2 (not available for dev)
    fp8_llm: bool = False,                  # Z-Image
    # FLUX.2 required discriminator.
    model_version: Optional[str] = None,
    # Shared args.
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    skip_existing: bool = False,
    keep_cache: bool = False,
    extra_args: Optional[list[str]] = None,
) -> dict[str, Any]:
    arch = get_architecture(architecture)
    if arch is None:
        return {
            "ok": False,
            "error": f"unknown architecture {architecture!r}. Known: {architecture_names()}",
        }
    if not arch.live:
        return {
            "ok": False,
            "error": (
                f"architecture {architecture!r} is a placeholder. "
                f"Live: {[a.name for a in live_architectures()]}"
            ),
        }
    if not arch.has_cache_text_encoder():
        return {
            "ok": False,
            "error": f"architecture {architecture!r} has no cache_text_encoder script",
        }
    if not os.path.isfile(dataset_config):
        return {"ok": False, "error": f"dataset_config not found: {dataset_config}"}

    # --- Per-arch required encoder validation ---
    if arch.name == "wan":
        if not t5:
            return {"ok": False, "error": "wan cache_text_encoder requires `t5` (T5 UMT5-XXL)"}
    elif arch.name in ("flux_2", "zimage"):
        if not text_encoder:
            return {
                "ok": False,
                "error": f"{arch.name} cache_text_encoder requires `text_encoder`",
            }
        if arch.name == "flux_2":
            if not model_version:
                return {
                    "ok": False,
                    "error": (
                        "flux_2 cache_text_encoder requires model_version — "
                        f"one of {list(arch.model_versions or ())}"
                    ),
                }
            if arch.model_versions and model_version not in arch.model_versions:
                return {
                    "ok": False,
                    "error": (
                        f"invalid model_version {model_version!r} for flux_2. "
                        f"Choose one of {list(arch.model_versions)}"
                    ),
                }
            # CLAUDE-NOTE: Musubi docs explicitly say fp8_text_encoder is not
            # available for FLUX.2 dev (Mistral3); we reject the combo up front
            # rather than letting the script fail cryptically.
            if fp8_text_encoder and model_version == "dev":
                return {
                    "ok": False,
                    "error": (
                        "fp8_text_encoder is not supported for flux_2 model_version='dev' "
                        "(Mistral3). Use a klein model_version or omit the flag."
                    ),
                }

    # --- Build argv ---
    args: list[str] = ["--dataset_config", dataset_config]

    if arch.name == "wan":
        args += ["--t5", t5]  # type: ignore[list-item]
        if clip:
            args += ["--clip", clip]
        if fp8_t5:
            args.append("--fp8_t5")
    elif arch.name == "flux_2":
        args += ["--text_encoder", text_encoder]  # type: ignore[list-item]
        args += ["--model_version", model_version]  # type: ignore[list-item]
        if fp8_text_encoder:
            args.append("--fp8_text_encoder")
    elif arch.name == "zimage":
        args += ["--text_encoder", text_encoder]  # type: ignore[list-item]
        if fp8_llm:
            args.append("--fp8_llm")

    if device:
        args += ["--device", device]
    if batch_size is not None:
        args += ["--batch_size", str(batch_size)]
    if num_workers is not None:
        args += ["--num_workers", str(num_workers)]
    if skip_existing:
        args.append("--skip_existing")
    if keep_cache:
        args.append("--keep_cache")
    if extra_args:
        args.extend(extra_args)

    extra_env = {
        k: os.environ[k]
        for k in ("HF_TOKEN",)
        if os.environ.get(k)
    }

    result = await run_musubi(
        arch.cache_text_encoder_script,  # type: ignore[arg-type]
        args,
        timeout=TIMEOUT_CACHE_SECONDS,
        extra_env=extra_env or None,
    )
    return {
        "ok": result.succeeded,
        "architecture": arch.name,
        "script": arch.cache_text_encoder_script,
        "summary": result.short_summary(),
        "result": result.to_dict(),
    }


# ---------------------------------------------------------------------------
# Tool: musubi_create_dataset_config
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Create a valid Musubi Tuner dataset TOML config. For image datasets "
        "pass `image_directory`; for video datasets pass `video_directory` + "
        "`target_frames`. The file is written to `output_path`, and the "
        "generated TOML text is returned in `toml_text` so the agent can "
        "preview it without a second read. Video datasets for HunyuanVideo / "
        "Wan2.1 / FramePack should use target_frames of 4n+1 (1, 5, 9, ..., 81)."
    ),
)
async def musubi_create_dataset_config(
    output_path: str,
    # Source — pick one.
    image_directory: Optional[str] = None,
    video_directory: Optional[str] = None,
    image_jsonl_file: Optional[str] = None,
    video_jsonl_file: Optional[str] = None,
    cache_directory: Optional[str] = None,
    # [general] fields.
    resolution_w: int = 960,
    resolution_h: int = 544,
    caption_extension: str = ".txt",
    batch_size: int = 1,
    num_repeats: int = 1,
    enable_bucket: bool = False,
    bucket_no_upscale: bool = False,
    # Video-specific.
    target_frames: Optional[list[int]] = None,
    frame_extraction: Optional[str] = None,
    frame_stride: Optional[int] = None,
    frame_sample: Optional[int] = None,
    max_frames: Optional[int] = None,
    source_fps: Optional[float] = None,
    # Control image/video (FLUX.1 Kontext, FLUX.2, Qwen-Image-Edit, Wan Fun-Control).
    control_directory: Optional[str] = None,
    control_resolution_w: Optional[int] = None,
    control_resolution_h: Optional[int] = None,
    no_resize_control: bool = False,
    multiple_target: bool = False,
) -> dict[str, Any]:
    try:
        control_resolution = (
            (control_resolution_w, control_resolution_h)
            if control_resolution_w is not None and control_resolution_h is not None
            else None
        )
        config = build_dataset_toml(
            image_directory=image_directory,
            video_directory=video_directory,
            image_jsonl_file=image_jsonl_file,
            video_jsonl_file=video_jsonl_file,
            cache_directory=cache_directory,
            resolution=(resolution_w, resolution_h),
            caption_extension=caption_extension,
            batch_size=batch_size,
            num_repeats=num_repeats,
            enable_bucket=enable_bucket,
            bucket_no_upscale=bucket_no_upscale,
            target_frames=target_frames,
            frame_extraction=frame_extraction,
            frame_stride=frame_stride,
            frame_sample=frame_sample,
            max_frames=max_frames,
            source_fps=source_fps,
            control_directory=control_directory,
            control_resolution=control_resolution,
            no_resize_control=no_resize_control,
            multiple_target=multiple_target,
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    try:
        write_dataset_toml(config, output_path)
    except OSError as exc:
        return {
            "ok": False,
            "error": f"failed to write {output_path}: {exc}",
            "toml_text": toml_dumps(config),
        }

    return {
        "ok": True,
        "path": os.path.abspath(output_path),
        "toml_text": toml_dumps(config),
        "dataset_type": "video" if (video_directory or video_jsonl_file) else "image",
    }


# ---------------------------------------------------------------------------
# Tool: musubi_validate_dataset_config
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Validate an existing Musubi Tuner dataset TOML config. Reports "
        "structural errors (missing source, no target_frames on videos, "
        "invalid frame_extraction) and, when `architecture` is given, "
        "architecture-specific constraints like the 4n+1 `target_frames` "
        "rule required by HunyuanVideo, Wan2.1, and FramePack."
    ),
)
async def musubi_validate_dataset_config(
    path: str,
    architecture: Optional[str] = None,
) -> dict[str, Any]:
    # CLAUDE-NOTE: 4n+1 constraint is documented for HunyuanVideo, Wan2.1,
    # and FramePack. Wan2.2 inherits from Wan2.1; FramePack auto-trims but
    # still prefers compliant frame counts. Other video archs (hv_1_5,
    # kandinsky5) are not audited yet so we don't enforce for them.
    archs_needing_4n1 = {"hv", "fpack"}
    enforce = architecture == "wan" or architecture in archs_needing_4n1

    result = validate_dataset_toml_file(path, enforce_4n_plus_1=enforce)
    out = result.to_dict()
    out["path"] = os.path.abspath(path)
    out["enforced_4n_plus_1"] = enforce
    return out


# ---------------------------------------------------------------------------
# Tool: musubi_train
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Launch LoRA/LoHa/LoKr training via `accelerate launch "
        "{arch}_train_network.py`. This is the long-running step — default "
        "timeout is 24 hours. The tool validates required model paths per "
        "architecture, builds the command, and returns once training exits. "
        "For Wan2.2 dual-model training pass `dit_high_noise` and "
        "`timestep_boundary`. For FLUX.2 pass `model_version`. For Wan pass "
        "`task` (one of the valid enum values from musubi_list_architectures). "
        "`network_module` defaults to the architecture's canonical LoRA "
        "module; override to train LoHa/LoKr. Use `extra_args` for any "
        "training flag not surfaced as a named parameter (see cli_help.txt "
        "in the repo for the full list)."
    ),
)
async def musubi_train(
    architecture: str,
    dataset_config: str,
    output_dir: str,
    output_name: str,
    # Model paths.
    dit: str,
    vae: Optional[str] = None,           # can come from dataset config too
    # Text encoder paths (one-of depending on arch — see musubi_cache_text_encoder).
    t5: Optional[str] = None,            # Wan
    text_encoder: Optional[str] = None,  # FLUX.2, Z-Image
    clip: Optional[str] = None,          # Wan2.1 I2V
    # Network config.
    network_module: Optional[str] = None,
    network_dim: Optional[int] = None,
    network_alpha: Optional[float] = None,
    network_args: Optional[list[str]] = None,
    # Training schedule.
    learning_rate: Optional[float] = None,
    max_train_epochs: Optional[int] = None,
    max_train_steps: Optional[int] = None,
    save_every_n_epochs: Optional[int] = None,
    save_every_n_steps: Optional[int] = None,
    optimizer_type: Optional[str] = None,
    optimizer_args: Optional[list[str]] = None,
    lr_scheduler: Optional[str] = None,
    lr_warmup_steps: Optional[int] = None,
    seed: Optional[int] = None,
    # Memory saving.
    mixed_precision: str = DEFAULT_MIXED_PRECISION,
    fp8_base: bool = False,
    fp8_scaled: bool = False,
    gradient_checkpointing: bool = False,
    gradient_checkpointing_cpu_offload: bool = False,
    blocks_to_swap: Optional[int] = None,
    # Flow/timestep (architecture-recommended values differ).
    timestep_sampling: Optional[str] = None,
    discrete_flow_shift: Optional[float] = None,
    weighting_scheme: Optional[str] = None,
    # Architecture discriminators.
    task: Optional[str] = None,            # Wan
    model_version: Optional[str] = None,   # FLUX.2
    # Wan2.2 dual-model.
    dit_high_noise: Optional[str] = None,
    timestep_boundary: Optional[float] = None,
    # Logging / sampling.
    logging_dir: Optional[str] = None,
    log_with: Optional[str] = None,
    sample_every_n_epochs: Optional[int] = None,
    sample_every_n_steps: Optional[int] = None,
    sample_prompts: Optional[str] = None,
    # Escape hatch.
    extra_args: Optional[list[str]] = None,
    # Launcher knob.
    num_cpu_threads_per_process: int = 1,
    timeout_seconds: float = TIMEOUT_TRAIN_SECONDS,
) -> dict[str, Any]:
    arch = get_architecture(architecture)
    if arch is None:
        return {
            "ok": False,
            "error": f"unknown architecture {architecture!r}. Known: {architecture_names()}",
        }
    if not arch.live:
        return {
            "ok": False,
            "error": (
                f"architecture {architecture!r} is a placeholder. "
                f"Live: {[a.name for a in live_architectures()]}"
            ),
        }
    if not arch.has_train_network():
        return {
            "ok": False,
            "error": f"architecture {architecture!r} has no train_network script",
        }
    if not os.path.isfile(dataset_config):
        return {"ok": False, "error": f"dataset_config not found: {dataset_config}"}

    # --- Per-arch required-encoder + discriminator checks ---
    if arch.name == "wan":
        if not t5:
            return {"ok": False, "error": "wan training requires `t5` (T5 UMT5-XXL)"}
        if task and arch.tasks and task not in arch.tasks:
            return {
                "ok": False,
                "error": (
                    f"invalid task {task!r} for wan. Choose one of {list(arch.tasks)}"
                ),
            }
        # CLAUDE-NOTE: --dit_high_noise without --timestep_boundary is technically
        # allowed (Musubi has defaults: 0.9 I2V, 0.875 T2V) but we surface the
        # pairing in the tool description; we don't hard-require here.
    elif arch.name == "flux_2":
        if not text_encoder:
            return {"ok": False, "error": "flux_2 training requires `text_encoder`"}
        if not model_version:
            return {
                "ok": False,
                "error": (
                    "flux_2 training requires model_version — "
                    f"one of {list(arch.model_versions or ())}"
                ),
            }
        if arch.model_versions and model_version not in arch.model_versions:
            return {
                "ok": False,
                "error": (
                    f"invalid model_version {model_version!r} for flux_2. "
                    f"Choose one of {list(arch.model_versions)}"
                ),
            }
    elif arch.name == "zimage":
        if not text_encoder:
            return {"ok": False, "error": "zimage training requires `text_encoder`"}

    # --- Build argv ---
    args: list[str] = [
        "--dataset_config", dataset_config,
        "--output_dir", output_dir,
        "--output_name", output_name,
        "--dit", dit,
    ]
    if vae:
        args += [arch.vae_arg, vae]

    # Encoders (arch-gated).
    if arch.name == "wan":
        args += ["--t5", t5]  # type: ignore[list-item]
        if clip:
            args += ["--clip", clip]
        if task:
            args += ["--task", task]
        if dit_high_noise:
            args += ["--dit_high_noise", dit_high_noise]
        if timestep_boundary is not None:
            args += ["--timestep_boundary", str(timestep_boundary)]
    elif arch.name in ("flux_2", "zimage"):
        args += ["--text_encoder", text_encoder]  # type: ignore[list-item]
        if arch.name == "flux_2":
            args += ["--model_version", model_version]  # type: ignore[list-item]

    # Network.
    module = network_module or arch.network_module
    if module:
        args += ["--network_module", module]
    if network_dim is not None:
        args += ["--network_dim", str(network_dim)]
    if network_alpha is not None:
        args += ["--network_alpha", str(network_alpha)]
    if network_args:
        args += ["--network_args", *network_args]

    # Schedule.
    if learning_rate is not None:
        args += ["--learning_rate", str(learning_rate)]
    if max_train_epochs is not None:
        args += ["--max_train_epochs", str(max_train_epochs)]
    if max_train_steps is not None:
        args += ["--max_train_steps", str(max_train_steps)]
    if save_every_n_epochs is not None:
        args += ["--save_every_n_epochs", str(save_every_n_epochs)]
    if save_every_n_steps is not None:
        args += ["--save_every_n_steps", str(save_every_n_steps)]
    if optimizer_type:
        args += ["--optimizer_type", optimizer_type]
    if optimizer_args:
        args += ["--optimizer_args", *optimizer_args]
    if lr_scheduler:
        args += ["--lr_scheduler", lr_scheduler]
    if lr_warmup_steps is not None:
        args += ["--lr_warmup_steps", str(lr_warmup_steps)]
    if seed is not None:
        args += ["--seed", str(seed)]

    # Memory-saving.
    # CLAUDE-NOTE: --mixed_precision is passed to the accelerate launcher
    # AND many training scripts read it too (they re-parse it). Passing it
    # here for the script is redundant with the launcher flag, but harmless
    # — the scripts accept it. We only pass it at the launcher level.
    if fp8_base:
        args.append("--fp8_base")
    if fp8_scaled:
        args.append("--fp8_scaled")
    if gradient_checkpointing:
        args.append("--gradient_checkpointing")
    if gradient_checkpointing_cpu_offload:
        args.append("--gradient_checkpointing_cpu_offload")
    if blocks_to_swap is not None:
        args += ["--blocks_to_swap", str(blocks_to_swap)]

    # Flow / timestep.
    if timestep_sampling:
        args += ["--timestep_sampling", timestep_sampling]
    if discrete_flow_shift is not None:
        args += ["--discrete_flow_shift", str(discrete_flow_shift)]
    if weighting_scheme:
        args += ["--weighting_scheme", weighting_scheme]

    # Logging / sampling.
    if logging_dir:
        args += ["--logging_dir", logging_dir]
    if log_with:
        args += ["--log_with", log_with]
    if sample_every_n_epochs is not None:
        args += ["--sample_every_n_epochs", str(sample_every_n_epochs)]
    if sample_every_n_steps is not None:
        args += ["--sample_every_n_steps", str(sample_every_n_steps)]
    if sample_prompts:
        args += ["--sample_prompts", sample_prompts]

    if extra_args:
        args.extend(extra_args)

    # --- Env passthroughs ---
    extra_env = {
        k: os.environ[k]
        for k in ("HF_TOKEN", "WANDB_API_KEY")
        if os.environ.get(k)
    }

    result = await run_musubi_training(
        arch.train_network_script,  # type: ignore[arg-type]
        args,
        mixed_precision=mixed_precision,
        num_cpu_threads_per_process=num_cpu_threads_per_process,
        timeout=timeout_seconds,
        extra_env=extra_env or None,
    )
    return {
        "ok": result.succeeded,
        "architecture": arch.name,
        "script": arch.train_network_script,
        "network_module": module,
        "summary": result.short_summary(),
        # CLAUDE-NOTE: Training produces many MB of log output. We return
        # only the last 8 KB of each stream — the full result is on disk
        # via --logging_dir. Truncation boundary chosen empirically: enough
        # to capture the final epoch + any error traceback, short enough
        # to stay under typical MCP response limits.
        "result": {
            **result.to_dict(),
            "stdout": result.stdout[-8000:],
            "stderr": result.stderr[-8000:],
        },
    }


# ---------------------------------------------------------------------------
# Tool: musubi_finetune
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Full fine-tuning (not LoRA) via `accelerate launch {arch}_train.py`. "
        "Only available for architectures with a full-train script. In the "
        "live set, that means Z-Image (`zimage_train.py`). Same pipeline "
        "shape as musubi_train, but there is no network_module — the DiT "
        "itself is trained. Much higher VRAM than LoRA; use `fp8_base`, "
        "`blocks_to_swap`, and `gradient_checkpointing_cpu_offload` to fit. "
        "Z-Image docs specifically recommend "
        "`--block_swap_optimizer_patch_params` when combining blocks_to_swap "
        "with certain optimizers; pass it via `extra_args` if needed."
    ),
)
async def musubi_finetune(
    architecture: str,
    dataset_config: str,
    output_dir: str,
    output_name: str,
    dit: str,
    vae: Optional[str] = None,
    text_encoder: Optional[str] = None,   # Z-Image, Qwen-Image
    t5: Optional[str] = None,             # HunyuanVideo (placeholder)
    # Schedule.
    learning_rate: Optional[float] = None,
    max_train_epochs: Optional[int] = None,
    max_train_steps: Optional[int] = None,
    save_every_n_epochs: Optional[int] = None,
    save_every_n_steps: Optional[int] = None,
    optimizer_type: Optional[str] = None,
    optimizer_args: Optional[list[str]] = None,
    lr_scheduler: Optional[str] = None,
    lr_warmup_steps: Optional[int] = None,
    seed: Optional[int] = None,
    # Memory-saving.
    mixed_precision: str = DEFAULT_MIXED_PRECISION,
    fp8_base: bool = False,
    fp8_scaled: bool = False,
    gradient_checkpointing: bool = False,
    gradient_checkpointing_cpu_offload: bool = False,
    gradient_accumulation_steps: Optional[int] = None,
    blocks_to_swap: Optional[int] = None,
    # Flow/timestep.
    timestep_sampling: Optional[str] = None,
    discrete_flow_shift: Optional[float] = None,
    weighting_scheme: Optional[str] = None,
    # Logging.
    logging_dir: Optional[str] = None,
    log_with: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
    num_cpu_threads_per_process: int = 1,
    timeout_seconds: float = TIMEOUT_TRAIN_SECONDS,
) -> dict[str, Any]:
    arch = get_architecture(architecture)
    if arch is None:
        return {
            "ok": False,
            "error": f"unknown architecture {architecture!r}. Known: {architecture_names()}",
        }
    if not arch.live:
        return {
            "ok": False,
            "error": (
                f"architecture {architecture!r} is a placeholder. "
                f"Live: {[a.name for a in live_architectures()]}"
            ),
        }
    if not arch.has_train_full():
        return {
            "ok": False,
            "error": (
                f"architecture {architecture!r} has no full-fine-tune script. "
                f"Archs that do (live or placeholder): "
                f"{[a.name for a in all_architectures() if a.has_train_full()]}"
            ),
        }
    if not os.path.isfile(dataset_config):
        return {"ok": False, "error": f"dataset_config not found: {dataset_config}"}

    # Per-arch required-encoder check. In the live set only zimage has a
    # full-train script, so we gate text_encoder here; HunyuanVideo will
    # need its own gate when it moves from placeholder to live.
    if arch.name == "zimage" and not text_encoder:
        return {"ok": False, "error": "zimage full fine-tune requires `text_encoder`"}

    args: list[str] = [
        "--dataset_config", dataset_config,
        "--output_dir", output_dir,
        "--output_name", output_name,
        "--dit", dit,
    ]
    if vae:
        args += [arch.vae_arg, vae]
    if arch.name == "zimage" and text_encoder:
        args += ["--text_encoder", text_encoder]
    if t5:
        args += ["--t5", t5]

    if learning_rate is not None:
        args += ["--learning_rate", str(learning_rate)]
    if max_train_epochs is not None:
        args += ["--max_train_epochs", str(max_train_epochs)]
    if max_train_steps is not None:
        args += ["--max_train_steps", str(max_train_steps)]
    if save_every_n_epochs is not None:
        args += ["--save_every_n_epochs", str(save_every_n_epochs)]
    if save_every_n_steps is not None:
        args += ["--save_every_n_steps", str(save_every_n_steps)]
    if optimizer_type:
        args += ["--optimizer_type", optimizer_type]
    if optimizer_args:
        args += ["--optimizer_args", *optimizer_args]
    if lr_scheduler:
        args += ["--lr_scheduler", lr_scheduler]
    if lr_warmup_steps is not None:
        args += ["--lr_warmup_steps", str(lr_warmup_steps)]
    if seed is not None:
        args += ["--seed", str(seed)]

    if fp8_base:
        args.append("--fp8_base")
    if fp8_scaled:
        args.append("--fp8_scaled")
    if gradient_checkpointing:
        args.append("--gradient_checkpointing")
    if gradient_checkpointing_cpu_offload:
        args.append("--gradient_checkpointing_cpu_offload")
    if gradient_accumulation_steps is not None:
        args += ["--gradient_accumulation_steps", str(gradient_accumulation_steps)]
    if blocks_to_swap is not None:
        args += ["--blocks_to_swap", str(blocks_to_swap)]

    if timestep_sampling:
        args += ["--timestep_sampling", timestep_sampling]
    if discrete_flow_shift is not None:
        args += ["--discrete_flow_shift", str(discrete_flow_shift)]
    if weighting_scheme:
        args += ["--weighting_scheme", weighting_scheme]

    if logging_dir:
        args += ["--logging_dir", logging_dir]
    if log_with:
        args += ["--log_with", log_with]

    if extra_args:
        args.extend(extra_args)

    extra_env = {
        k: os.environ[k]
        for k in ("HF_TOKEN", "WANDB_API_KEY")
        if os.environ.get(k)
    }

    result = await run_musubi_training(
        arch.train_full_script,  # type: ignore[arg-type]
        args,
        mixed_precision=mixed_precision,
        num_cpu_threads_per_process=num_cpu_threads_per_process,
        timeout=timeout_seconds,
        extra_env=extra_env or None,
    )
    return {
        "ok": result.succeeded,
        "architecture": arch.name,
        "script": arch.train_full_script,
        "summary": result.short_summary(),
        "result": {
            **result.to_dict(),
            "stdout": result.stdout[-8000:],
            "stderr": result.stderr[-8000:],
        },
    }


# ---------------------------------------------------------------------------
# Tool: musubi_generate
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Generate an image or video using a trained checkpoint (with or "
        "without a LoRA). Routes to {arch}_generate_image.py (image archs) "
        "or {arch}_generate_video.py (video archs). For Wan you must pass "
        "`task` (matching the checkpoint's task). For FLUX.2 pass "
        "`model_version`. `lora_weight` + `lora_multiplier` accept lists "
        "for stacking LoRAs. Wan2.2 dual-model generation uses "
        "`dit_high_noise` + `lora_weight_high_noise` + "
        "`lora_multiplier_high_noise`. Use `video_size` / `video_length` / "
        "`fps` for video archs; `image_size` for image archs."
    ),
)
async def musubi_generate(
    architecture: str,
    prompt: str,
    # Model paths.
    dit: str,
    vae: Optional[str] = None,
    text_encoder: Optional[str] = None,   # FLUX.2, Z-Image
    t5: Optional[str] = None,             # Wan
    clip: Optional[str] = None,           # Wan2.1 I2V
    # LoRA inputs (parallel lists; one multiplier per weight).
    lora_weight: Optional[list[str]] = None,
    lora_multiplier: Optional[list[float]] = None,
    # Sampling.
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    infer_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    flow_shift: Optional[float] = None,
    embedded_cfg_scale: Optional[float] = None,
    # Image archs.
    image_size_w: Optional[int] = None,
    image_size_h: Optional[int] = None,
    # Video archs.
    video_size_w: Optional[int] = None,
    video_size_h: Optional[int] = None,
    video_length: Optional[int] = None,
    fps: Optional[int] = None,
    image_path: Optional[str] = None,         # Wan I2V
    end_image_path: Optional[str] = None,     # Wan flf2v
    # Memory + attention.
    fp8: bool = False,
    fp8_scaled: bool = False,
    fp8_t5: bool = False,                     # Wan
    fp8_llm: bool = False,                    # Z-Image
    fp8_text_encoder: bool = False,           # FLUX.2
    blocks_to_swap: Optional[int] = None,
    attn_mode: Optional[str] = None,
    # Wan discriminators + dual-model.
    task: Optional[str] = None,
    dit_high_noise: Optional[str] = None,
    lora_weight_high_noise: Optional[list[str]] = None,
    lora_multiplier_high_noise: Optional[list[float]] = None,
    timestep_boundary: Optional[float] = None,
    guidance_scale_high_noise: Optional[float] = None,
    # FLUX.2 discriminator.
    model_version: Optional[str] = None,
    # Control image(s) — FLUX.2 Kontext, Wan Fun-Control.
    control_image_path: Optional[list[str]] = None,
    control_path: Optional[str] = None,
    no_resize_control: bool = False,
    # Output.
    save_path: Optional[str] = None,
    output_type: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
    timeout_seconds: float = TIMEOUT_GENERATE_SECONDS,
) -> dict[str, Any]:
    arch = get_architecture(architecture)
    if arch is None:
        return {
            "ok": False,
            "error": f"unknown architecture {architecture!r}. Known: {architecture_names()}",
        }
    if not arch.live:
        return {
            "ok": False,
            "error": (
                f"architecture {architecture!r} is a placeholder. "
                f"Live: {[a.name for a in live_architectures()]}"
            ),
        }
    if not arch.has_generate():
        return {"ok": False, "error": f"architecture {architecture!r} has no generate script"}

    if lora_weight and lora_multiplier and len(lora_weight) != len(lora_multiplier):
        return {
            "ok": False,
            "error": (
                f"lora_weight and lora_multiplier must be the same length "
                f"(got {len(lora_weight)} weights, {len(lora_multiplier)} multipliers)"
            ),
        }

    # Per-arch required-path + discriminator checks.
    if arch.name == "wan":
        if not t5:
            return {"ok": False, "error": "wan generation requires `t5` (T5 UMT5-XXL)"}
        if task and arch.tasks and task not in arch.tasks:
            return {
                "ok": False,
                "error": f"invalid task {task!r} for wan. Choose one of {list(arch.tasks)}",
            }
    elif arch.name == "flux_2":
        if not text_encoder:
            return {"ok": False, "error": "flux_2 generation requires `text_encoder`"}
        if not model_version:
            return {
                "ok": False,
                "error": (
                    "flux_2 generation requires model_version — "
                    f"one of {list(arch.model_versions or ())}"
                ),
            }
        if arch.model_versions and model_version not in arch.model_versions:
            return {
                "ok": False,
                "error": (
                    f"invalid model_version {model_version!r} for flux_2. "
                    f"Choose one of {list(arch.model_versions)}"
                ),
            }
    elif arch.name == "zimage":
        if not text_encoder:
            return {"ok": False, "error": "zimage generation requires `text_encoder`"}

    args: list[str] = ["--prompt", prompt, "--dit", dit]
    if vae:
        args += [arch.vae_arg, vae]

    # Encoders.
    if arch.name == "wan":
        args += ["--t5", t5]  # type: ignore[list-item]
        if clip:
            args += ["--clip", clip]
        if task:
            args += ["--task", task]
        if dit_high_noise:
            args += ["--dit_high_noise", dit_high_noise]
        if timestep_boundary is not None:
            args += ["--timestep_boundary", str(timestep_boundary)]
        if guidance_scale_high_noise is not None:
            args += ["--guidance_scale_high_noise", str(guidance_scale_high_noise)]
        if lora_weight_high_noise:
            args += ["--lora_weight_high_noise", *lora_weight_high_noise]
        if lora_multiplier_high_noise:
            args += [
                "--lora_multiplier_high_noise",
                *(str(m) for m in lora_multiplier_high_noise),
            ]
        if image_path:
            args += ["--image_path", image_path]
        if end_image_path:
            args += ["--end_image_path", end_image_path]
        if control_path:
            args += ["--control_path", control_path]
    elif arch.name in ("flux_2", "zimage"):
        args += ["--text_encoder", text_encoder]  # type: ignore[list-item]
        if arch.name == "flux_2":
            args += ["--model_version", model_version]  # type: ignore[list-item]

    # Common sampling.
    if negative_prompt is not None:
        args += ["--negative_prompt", negative_prompt]
    if seed is not None:
        args += ["--seed", str(seed)]
    if infer_steps is not None:
        args += ["--infer_steps", str(infer_steps)]
    if guidance_scale is not None:
        args += ["--guidance_scale", str(guidance_scale)]
    if flow_shift is not None:
        args += ["--flow_shift", str(flow_shift)]
    if embedded_cfg_scale is not None:
        args += ["--embedded_cfg_scale", str(embedded_cfg_scale)]

    # Size.
    if arch.type == "video":
        if video_size_w is not None and video_size_h is not None:
            args += ["--video_size", str(video_size_w), str(video_size_h)]
        if video_length is not None:
            args += ["--video_length", str(video_length)]
        if fps is not None:
            args += ["--fps", str(fps)]
    else:
        if image_size_w is not None and image_size_h is not None:
            args += ["--image_size", str(image_size_w), str(image_size_h)]

    # LoRA.
    if lora_weight:
        args += ["--lora_weight", *lora_weight]
    if lora_multiplier:
        args += ["--lora_multiplier", *(str(m) for m in lora_multiplier)]

    # Memory / attention.
    if fp8:
        args.append("--fp8")
    if fp8_scaled:
        args.append("--fp8_scaled")
    if arch.name == "wan" and fp8_t5:
        args.append("--fp8_t5")
    if arch.name == "zimage" and fp8_llm:
        args.append("--fp8_llm")
    if arch.name == "flux_2" and fp8_text_encoder:
        args.append("--fp8_text_encoder")
    if blocks_to_swap is not None:
        args += ["--blocks_to_swap", str(blocks_to_swap)]
    if attn_mode:
        args += ["--attn_mode", attn_mode]

    # Control.
    if control_image_path:
        args += ["--control_image_path", *control_image_path]
    if no_resize_control:
        args.append("--no_resize_control")

    if save_path:
        args += ["--save_path", save_path]
    if output_type:
        args += ["--output_type", output_type]

    if extra_args:
        args.extend(extra_args)

    result = await run_musubi(
        arch.generate_script,  # type: ignore[arg-type]
        args,
        timeout=timeout_seconds,
    )
    return {
        "ok": result.succeeded,
        "architecture": arch.name,
        "script": arch.generate_script,
        "summary": result.short_summary(),
        "save_path": save_path,
        "result": result.to_dict(),
    }


# ---------------------------------------------------------------------------
# Tool: musubi_validate_dataset
# ---------------------------------------------------------------------------

# CLAUDE-NOTE: Image + video extensions match what Musubi accepts (per
# docs/dataset_config.md — it uses PIL/av internally). Kept conservative;
# agents can rename weird-cased files before running.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}


@mcp.tool(
    description=(
        "Inspect a dataset directory on disk: count images or videos, "
        "check that each has a matching caption file (default `.txt` "
        "sidecar), report missing/empty captions, and list the sampled "
        "file extensions. This runs in-process (no subprocess) and is "
        "safe to call before caching, to catch data hygiene issues early."
    ),
)
async def musubi_validate_dataset(
    directory: str,
    caption_extension: str = ".txt",
    dataset_kind: str = "image",  # "image" or "video"
    max_missing_to_list: int = 50,
) -> dict[str, Any]:
    if dataset_kind not in ("image", "video"):
        return {"ok": False, "error": "dataset_kind must be 'image' or 'video'"}

    root = Path(directory)
    if not root.is_dir():
        return {"ok": False, "error": f"not a directory: {directory}"}

    exts = _IMAGE_EXTS if dataset_kind == "image" else _VIDEO_EXTS
    media_files: list[Path] = []
    # CLAUDE-NOTE: Walk the whole tree — many datasets are organized into
    # subfolders by class/concept. Musubi's dataset config also walks
    # recursively when `image_directory` points at a top-level folder.
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            media_files.append(p)

    missing_captions: list[str] = []
    empty_captions: list[str] = []
    for media in media_files:
        sidecar = media.with_suffix(caption_extension)
        if not sidecar.is_file():
            if len(missing_captions) < max_missing_to_list:
                missing_captions.append(str(media))
            continue
        try:
            text = sidecar.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as exc:
            missing_captions.append(f"{media} (unreadable: {exc})")
            continue
        if not text:
            if len(empty_captions) < max_missing_to_list:
                empty_captions.append(str(sidecar))

    extension_counts: dict[str, int] = {}
    for m in media_files:
        extension_counts[m.suffix.lower()] = extension_counts.get(m.suffix.lower(), 0) + 1

    missing_count = sum(
        1 for m in media_files if not m.with_suffix(caption_extension).is_file()
    )
    empty_count = len(empty_captions)
    ok = missing_count == 0 and empty_count == 0 and len(media_files) > 0

    issues: list[str] = []
    if not media_files:
        issues.append(f"no {dataset_kind} files found under {directory}")
    if missing_count:
        issues.append(f"{missing_count} {dataset_kind}(s) missing {caption_extension} caption sidecar")
    if empty_count:
        issues.append(f"{empty_count} caption files are empty")

    return {
        "ok": ok,
        "directory": str(root.resolve()),
        "dataset_kind": dataset_kind,
        "caption_extension": caption_extension,
        "counts": {
            "media_files": len(media_files),
            "missing_captions": missing_count,
            "empty_captions": empty_count,
            "by_extension": extension_counts,
        },
        "missing_captions_sample": missing_captions,
        "empty_captions_sample": empty_captions,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Tool: musubi_convert_lora
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Convert a LoRA/LoHa/LoKr file between Musubi's default format and "
        "an 'other' format (typically ComfyUI-compatible). `target='other'` "
        "converts Musubi → other; `target='default'` converts back. "
        "`diffusers_prefix` overrides the Diffusers weight prefix "
        "(defaults to 'diffusion_model' when omitted)."
    ),
)
async def musubi_convert_lora(
    input: str,
    output: str,
    target: str,  # "other" | "default"
    diffusers_prefix: Optional[str] = None,
) -> dict[str, Any]:
    if target not in ("other", "default"):
        return {"ok": False, "error": f"target must be 'other' or 'default', got {target!r}"}
    if not os.path.isfile(input):
        return {"ok": False, "error": f"input file not found: {input}"}

    args: list[str] = [
        "--input", input,
        "--output", output,
        "--target", target,
    ]
    if diffusers_prefix:
        args += ["--diffusers_prefix", diffusers_prefix]

    result = await run_musubi(
        "convert_lora.py", args, timeout=TIMEOUT_UTILITY_SECONDS,
    )
    return {
        "ok": result.succeeded,
        "output": output if result.succeeded else None,
        "summary": result.short_summary(),
        "result": result.to_dict(),
    }


# ---------------------------------------------------------------------------
# Tool: musubi_merge_lora
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Merge one or more LoRA weights into a base DiT checkpoint, "
        "producing a new merged checkpoint. Uses Musubi's merge_lora.py, "
        "which is HunyuanVideo-focused (the merged_model output follows "
        "that schema). `lora_weight` and `lora_multiplier` accept parallel "
        "lists for stacking multiple LoRAs at different strengths."
    ),
)
async def musubi_merge_lora(
    dit: str,
    save_merged_model: str,
    lora_weight: list[str],
    lora_multiplier: Optional[list[float]] = None,
    dit_in_channels: Optional[int] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    if not lora_weight:
        return {"ok": False, "error": "lora_weight must contain at least one path"}
    if lora_multiplier and len(lora_multiplier) != len(lora_weight):
        return {
            "ok": False,
            "error": (
                f"lora_multiplier length ({len(lora_multiplier)}) must match "
                f"lora_weight length ({len(lora_weight)})"
            ),
        }
    if not os.path.isfile(dit):
        return {"ok": False, "error": f"dit not found: {dit}"}
    missing = [p for p in lora_weight if not os.path.isfile(p)]
    if missing:
        return {"ok": False, "error": f"lora weight files not found: {missing}"}

    args: list[str] = [
        "--dit", dit,
        "--save_merged_model", save_merged_model,
        "--lora_weight", *lora_weight,
    ]
    if lora_multiplier:
        args += ["--lora_multiplier", *(str(m) for m in lora_multiplier)]
    if dit_in_channels is not None:
        args += ["--dit_in_channels", str(dit_in_channels)]
    if device:
        args += ["--device", device]

    result = await run_musubi(
        "merge_lora.py", args, timeout=TIMEOUT_UTILITY_SECONDS,
    )
    return {
        "ok": result.succeeded,
        "output": save_merged_model if result.succeeded else None,
        "summary": result.short_summary(),
        "result": result.to_dict(),
    }


# ---------------------------------------------------------------------------
# Tool: musubi_ema_merge
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Merge multiple LoRA checkpoints with Post-Hoc EMA. Pass the list "
        "of checkpoint paths in `checkpoint_paths` (order matters unless "
        "`no_sort=True`; by default they are sorted by modification time). "
        "`beta` is the EMA decay rate; `sigma_rel` enables Power Function "
        "EMA (defaults to linear interpolation when omitted)."
    ),
)
async def musubi_ema_merge(
    checkpoint_paths: list[str],
    output_file: str,
    beta: Optional[float] = None,
    beta2: Optional[float] = None,
    sigma_rel: Optional[float] = None,
    no_sort: bool = False,
) -> dict[str, Any]:
    if len(checkpoint_paths) < 2:
        return {"ok": False, "error": "ema_merge requires at least 2 checkpoint_paths"}
    missing = [p for p in checkpoint_paths if not os.path.isfile(p)]
    if missing:
        return {"ok": False, "error": f"checkpoint files not found: {missing}"}

    args: list[str] = ["--output_file", output_file]
    if no_sort:
        args.append("--no_sort")
    if beta is not None:
        args += ["--beta", str(beta)]
    if beta2 is not None:
        args += ["--beta2", str(beta2)]
    if sigma_rel is not None:
        args += ["--sigma_rel", str(sigma_rel)]
    # CLAUDE-NOTE: The checkpoint paths are positional args on
    # lora_post_hoc_ema.py — argparse parses them after the flagged opts.
    # We list them last so they are unambiguously positional.
    args.extend(checkpoint_paths)

    result = await run_musubi(
        "lora_post_hoc_ema.py", args, timeout=TIMEOUT_UTILITY_SECONDS,
    )
    return {
        "ok": result.succeeded,
        "output": output_file if result.succeeded else None,
        "summary": result.short_summary(),
        "result": result.to_dict(),
    }


# ---------------------------------------------------------------------------
# Tool: musubi_caption_images
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Auto-caption every image in a directory using a Qwen2.5-VL model "
        "via Musubi's caption_images_by_qwen_vl.py. `output_format='text'` "
        "writes a .txt sidecar next to each image; `output_format='jsonl'` "
        "writes a single JSONL file specified by `output_file`. Use "
        "`fp8_vl=True` to halve VRAM at a small quality cost; provide a "
        "custom `prompt` to steer the caption style (e.g. 'Describe subject, "
        "style, and composition. Use \\n for newlines.')."
    ),
)
async def musubi_caption_images(
    image_dir: str,
    model_path: str,
    output_file: Optional[str] = None,
    output_format: str = "jsonl",
    max_new_tokens: Optional[int] = None,
    prompt: Optional[str] = None,
    max_size: Optional[int] = None,
    fp8_vl: bool = False,
) -> dict[str, Any]:
    if output_format not in ("jsonl", "text"):
        return {"ok": False, "error": "output_format must be 'jsonl' or 'text'"}
    if output_format == "jsonl" and not output_file:
        return {"ok": False, "error": "output_format='jsonl' requires output_file"}
    if not os.path.isdir(image_dir):
        return {"ok": False, "error": f"image_dir not found: {image_dir}"}

    args: list[str] = [
        "--image_dir", image_dir,
        "--model_path", model_path,
        "--output_format", output_format,
    ]
    if output_file:
        args += ["--output_file", output_file]
    if max_new_tokens is not None:
        args += ["--max_new_tokens", str(max_new_tokens)]
    if prompt:
        args += ["--prompt", prompt]
    if max_size is not None:
        args += ["--max_size", str(max_size)]
    if fp8_vl:
        args.append("--fp8_vl")

    extra_env = {
        k: os.environ[k]
        for k in ("HF_TOKEN",)
        if os.environ.get(k)
    }

    result = await run_musubi(
        "caption_images_by_qwen_vl.py",
        args,
        timeout=TIMEOUT_TRAIN_SECONDS,  # Captioning a large dataset can take hours.
        extra_env=extra_env or None,
    )
    return {
        "ok": result.succeeded,
        "output_format": output_format,
        "output_file": output_file,
        "summary": result.short_summary(),
        "result": result.to_dict(),
    }


# ---------------------------------------------------------------------------
# Prompts — planning + diagnosis
# ---------------------------------------------------------------------------


@mcp.prompt(
    description=(
        "Produce a structured training plan: model files to download, "
        "dataset TOML config, caching + training commands with recommended "
        "hyperparameters, and expected VRAM/time. The caller fills in the "
        "architecture + goal + hardware, and the returned prompt asks "
        "Claude (or whichever model) to assemble the plan using "
        "musubi_list_architectures, the musubi://docs/* resources, and "
        "the architecture-specific gotchas."
    ),
)
def plan_training_run(
    architecture: str,
    training_type: str,       # lora | loha | lokr | finetune
    source_data: str,
    hardware: str,
    goal: str,
) -> str:
    return f"""\
You are about to plan a LoRA training run with Musubi Tuner. Produce a concrete
plan, not generalities. Work through the following:

1. Call `musubi_list_architectures` and locate the entry for `{architecture}`.
   Confirm:
     - the architecture is marked `live`
     - `{training_type}` is supported (LoRA/LoHa/LoKr via `train_network`,
       or full fine-tune via `train_full`)
     - required text-encoder flags, VAE arg, DiT arg

2. Read `musubi://docs/{architecture}` and `musubi://docs/dataset_config`
   for this architecture's specific requirements. Note any gotchas:
     - Wan2.2: dual DiT (low + high noise), `--timestep_boundary`, no `--clip` for I2V
     - FLUX.2: required `--model_version`, `--timestep_sampling flux2_shift`,
       `--weighting_scheme none`, `--fp8_text_encoder` unavailable for `dev`
     - Z-Image: full fine-tune supported; `--fp8_llm` (not `--fp8_text_encoder`)
     - HunyuanVideo / Wan2.1 / FramePack: video `target_frames` must satisfy 4n+1

3. List the **exact model files** the user needs to download, and where each
   one's path should go (which tool parameter / TOML field).

4. Draft the dataset TOML. For a video dataset, pick `target_frames` (4n+1 if
   required), `frame_extraction`, and `resolution` consistent with the
   architecture's memory/quality trade-off and the user's hardware.

5. Produce the three-step pipeline commands as MCP tool calls in order:
     a. `musubi_cache_latents(...)` with the right arch + VAE + batch_size
     b. `musubi_cache_text_encoder(...)` with correct encoder flags
     c. `musubi_train(...)` with recommended hyperparameters for this goal

   Pick learning_rate, network_dim/alpha, max_train_epochs, optimizer_type,
   mixed_precision, and memory-saving flags (fp8_base, fp8_scaled,
   blocks_to_swap, gradient_checkpointing) that will fit on the user's
   hardware. Justify each choice in one line.

6. Estimate **peak VRAM** and **wall-clock training time** with +/- ranges,
   citing what the choice of blocks_to_swap / fp8 / resolution implies.

7. End with a one-paragraph risk/review checklist: things the user should
   verify before kicking off the run (accelerate config, dataset captions
   present, disk space for checkpoints, etc.).

CONTEXT PROVIDED BY USER:
- Architecture: `{architecture}`
- Training type: `{training_type}`
- Source data: {source_data}
- Hardware: {hardware}
- Goal: {goal}
"""


@mcp.prompt(
    description=(
        "Diagnose a failed or stuck Musubi training run. The caller supplies "
        "the tail of the training log, the architecture, and the training "
        "type, and this prompt asks the model to identify the root cause "
        "(CUDA OOM, NaN loss, missing model file, dataset mismatch, etc.) "
        "and propose a concrete fix: which flag to change, in which "
        "direction, and why."
    ),
)
def diagnose_training_issue(
    log_output: str,
    architecture: str,
    training_type: str,
) -> str:
    return f"""\
Diagnose the following Musubi Tuner training failure. Identify the single
most likely root cause first, then any secondary contributing causes. For
each, propose a concrete fix — specific flag values, not generalities.

Common failure modes to check against:

- **CUDA out of memory**: reduce resolution, increase `blocks_to_swap`,
  enable `fp8_base` + `fp8_scaled`, enable `gradient_checkpointing`
  and/or `gradient_checkpointing_cpu_offload`, drop `batch_size` to 1,
  increase `gradient_accumulation_steps` to preserve effective batch size.
- **NaN / Inf loss**: lower `learning_rate` by 2–5x, verify
  `mixed_precision` isn't fp16 on a model that wants bf16, check
  `optimizer_type` compatibility, try `--cuda_allow_tf32` off.
- **Dataset 4n+1 violation** (Wan2.1 / HunyuanVideo / FramePack): call
  `musubi_validate_dataset_config` with the arch; adjust `target_frames`.
- **Missing model file**: stderr shows FileNotFoundError on a specific
  path — point the corresponding flag at the right checkpoint.
- **Wrong model_version** (FLUX.2): script errors out on shape mismatch
  or KeyError in the state dict — check the checkpoint matches
  `--model_version`.
- **accelerate not configured**: `accelerate config` prompt or
  `ValueError: --mixed_precision` — run `accelerate config` once.
- **sageattention import failures**: benign warning; safe to ignore unless
  `attn_mode=sageattn` is specifically requested.
- **FP8 + GPU mismatch**: fp8 requires Ada (RTX 4xxx) or newer for
  `--fp8_fast`; fall back to `--fp8_base --fp8_scaled` on Ampere.

CONTEXT:
- Architecture: `{architecture}`
- Training type: `{training_type}`

LOG TAIL:
```
{log_output}
```

OUTPUT:
1. Most-likely root cause (one sentence).
2. Exact fix: flag name + value change (e.g. "add `--blocks_to_swap 20` or
   increase existing value to 32").
3. Secondary issues, if any.
4. One sentence on what to verify before retrying.
"""


# ---------------------------------------------------------------------------
# Resources — Musubi architecture docs
# ---------------------------------------------------------------------------

# CLAUDE-NOTE: FastMCP registers resources via the @mcp.resource decorator,
# which wants a fixed URI per function. We generate one registration per
# doc name by closing over the name in a tiny factory. Using a list-comp
# to build handlers isn't enough on its own because the decorator needs to
# run at import time.

def _register_doc_resource(doc_name: str) -> None:
    uri = doc_uri(doc_name)

    @mcp.resource(uri, mime_type="text/markdown")
    def _handler() -> str:  # noqa: ANN202 — FastMCP infers the return type
        return read_doc(doc_name)

    _handler.__name__ = f"doc_{doc_name}"


for _name in DOC_FILES:
    _register_doc_resource(_name)


# ---------------------------------------------------------------------------
# Resources — community LoRA training knowledge base
# ---------------------------------------------------------------------------

# CLAUDE-NOTE: knowledge:// is the shared URI scheme across musubi-mcp,
# ltx-trainer-mcp, and the klippbok workspace scaffolder. Files flagged
# ``INSUFFICIENT DATA`` are exposed verbatim — the flag is load-bearing
# and must reach the orchestration agent so it asks the user instead of
# guessing hyperparameters. Do not filter.
@mcp.resource(
    "knowledge://index",
    mime_type="text/markdown",
    description="Index of the LoRA training knowledge base (per-arch training settings, "
                "failure modes, dataset quality, hardware profiles). Read this first "
                "to see what's available.",
)
def _knowledge_index_handler() -> str:  # noqa: ANN202
    return knowledge_index()


def _register_knowledge_resource(name: str) -> None:
    uri = knowledge_uri(name)

    @mcp.resource(uri, mime_type="text/markdown")
    def _handler() -> str:  # noqa: ANN202
        return read_knowledge(name)

    _handler.__name__ = f"knowledge_{name}"


for _name in all_knowledge_names():
    _register_knowledge_resource(_name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Console-script entry point. Runs the server over stdio."""
    # CLAUDE-NOTE: FastMCP.run() uses stdio by default and handles the
    # asyncio loop. stdio is what every desktop MCP client expects; a
    # streamable-http transport can be bolted on later via a CLI flag
    # if/when a remote deployment target needs it.
    mcp.run()


if __name__ == "__main__":
    main()
