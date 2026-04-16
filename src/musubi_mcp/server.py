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
import sys
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
    TIMEOUT_TRAIN_SECONDS,
)
from .dataset_config import (
    build_dataset_toml,
    dumps as toml_dumps,
    validate_dataset_toml_file,
    write_dataset_toml,
)
from .resources import DOC_FILES, doc_uri, read_doc
from .runner import (
    accelerate_executable,
    build_env,
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
