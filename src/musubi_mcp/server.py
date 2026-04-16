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
from .constants import TIMEOUT_CACHE_SECONDS
from .resources import DOC_FILES, doc_uri, read_doc
from .runner import (
    accelerate_executable,
    build_env,
    musubi_tuner_dir,
    python_executable,
    run_command,
    run_musubi,
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
