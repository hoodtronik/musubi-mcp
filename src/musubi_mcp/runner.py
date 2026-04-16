"""Async subprocess wrapper for Musubi Tuner invocations.

Two execution modes:

    1. Direct Python — caching, generation, and utility scripts.
       Command: ``<python> <musubi_tuner_dir>/<script>.py --arg val ...``

    2. Accelerate launch — every training script.
       Command: ``<accelerate> launch --num_cpu_threads_per_process 1
                 --mixed_precision bf16 <script>.py --arg val ...``

Both modes set cwd to ``MUSUBI_TUNER_DIR`` because Musubi's scripts use
relative imports from its package.

Design notes (mirroring the sibling ``klippbok-mcp`` runner):

- ``asyncio.create_subprocess_exec`` with explicit ``asyncio.subprocess.PIPE``
  — never ``capture_output=True``.
- ``PYTHONUTF8=1``, ``PYTHONIOENCODING=utf-8``, ``PYTHONUNBUFFERED=1`` on
  every spawned process. Musubi writes progress bars to stderr; forcing
  UTF-8 avoids decoding errors from tqdm's unicode block chars on Windows.
- Env is queried on every call so users can flip ``.env`` without
  restarting the server.
- ``CommandResult`` is the structured return — no exceptions bubble out
  of ``run_command`` / ``run_musubi`` / ``run_musubi_training`` for
  predictable subprocess failures; tool handlers decide how to surface
  them.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .constants import (
    DEFAULT_CPU_THREADS_PER_PROCESS,
    DEFAULT_MIXED_PRECISION,
    TIMEOUT_UTILITY_SECONDS,
)


# ---------------------------------------------------------------------------
# CommandResult — returned from every subprocess call
# ---------------------------------------------------------------------------


@dataclass
class CommandResult:
    """Structured outcome of a subprocess call.

    Exit code ``-1`` is reserved for failures where the process never
    started (e.g. executable not found, cwd missing) or was killed by
    timeout. Check ``timed_out`` to disambiguate.
    """

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    cwd: Optional[str] = None
    timed_out: bool = False
    mode: str = "python"  # "python" or "accelerate"

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def short_summary(self) -> str:
        status = (
            "ok"
            if self.succeeded
            else ("timeout" if self.timed_out else f"exit={self.exit_code}")
        )
        return f"[{self.mode} {status} elapsed={self.duration_seconds:.1f}s]"


# ---------------------------------------------------------------------------
# Env + executable discovery
# ---------------------------------------------------------------------------


def musubi_tuner_dir() -> Optional[str]:
    """Return the absolute path to the Musubi Tuner checkout, or None.

    The tool layer is responsible for surfacing a user-friendly error when
    this is unset — we just read the env.
    """
    raw = os.environ.get("MUSUBI_TUNER_DIR")
    if not raw:
        return None
    # CLAUDE-NOTE: expanduser so "~/musubi-tuner" works on *nix; abspath so
    # relative paths resolve against the MCP server's cwd rather than the
    # subprocess-child's cwd (which we're about to override).
    return os.path.abspath(os.path.expanduser(raw))


def python_executable() -> str:
    """Python interpreter used for Musubi invocations.

    Reads ``MUSUBI_PYTHON`` each time so updated env takes effect on the
    next tool call. Falls back to ``sys.executable`` — works only if the
    MCP server happens to be running inside Musubi's own venv.
    """
    # CLAUDE-NOTE: same pattern as klippbok-mcp's ``python_executable`` —
    # intentional duplication rather than a shared helper, because the two
    # servers will often be installed independently.
    return os.environ.get("MUSUBI_PYTHON") or sys.executable


def accelerate_executable() -> str:
    """Path to the ``accelerate`` launcher.

    Precedence:
      1. ``MUSUBI_ACCELERATE`` env var (explicit override).
      2. ``accelerate[.exe]`` next to ``MUSUBI_PYTHON`` — this is the
         common case when the user points us at Musubi's venv Python.
      3. ``accelerate`` on PATH (found via ``shutil.which``).
      4. Literal ``"accelerate"`` — lets the subprocess call fail loudly
         with FileNotFoundError, which the runner converts to exit_code -1.
    """
    override = os.environ.get("MUSUBI_ACCELERATE")
    if override:
        return override

    py = python_executable()
    py_dir = Path(py).parent
    # CLAUDE-NOTE: On Windows, uv's venv puts scripts in Scripts/ next to
    # python.exe; on Linux/macOS they sit in bin/ next to python. Checking
    # the sibling directory of the configured interpreter hits both.
    for candidate in ("accelerate.exe", "accelerate"):
        p = py_dir / candidate
        if p.exists():
            return str(p)

    which = shutil.which("accelerate")
    if which:
        return which

    return "accelerate"


def build_env(extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Assemble the env dict passed to every subprocess.

    - ``PYTHONUTF8=1``: UTF-8 mode, required on Windows for tqdm progress
      chars and for any Musubi script touching unicode filenames.
    - ``PYTHONIOENCODING=utf-8``: belt-and-suspenders for older code paths.
    - ``PYTHONUNBUFFERED=1``: flush stdout/stderr immediately so output
      arrives promptly (useful once we add live streaming).
    - ``extra``: caller-supplied keys (API tokens, HF_HOME, ...). Empty
      values are skipped so an un-set key doesn't clobber an inherited one.
    """
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        for key, value in extra.items():
            if value:
                env[key] = value
    return env


# ---------------------------------------------------------------------------
# Low-level subprocess runner
# ---------------------------------------------------------------------------


async def _run_exec(
    cmd: list[str],
    *,
    timeout: float,
    cwd: Optional[str],
    extra_env: Optional[dict[str, str]],
    mode: str,
) -> CommandResult:
    """Spawn, wait, decode, wrap."""
    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=build_env(extra_env),
            cwd=cwd,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            command=cmd,
            exit_code=-1,
            stdout="",
            stderr=f"executable not found: {exc}",
            duration_seconds=0.0,
            cwd=cwd,
            mode=mode,
        )
    except NotADirectoryError as exc:
        return CommandResult(
            command=cmd,
            exit_code=-1,
            stdout="",
            stderr=f"cwd is not a directory: {exc}",
            duration_seconds=0.0,
            cwd=cwd,
            mode=mode,
        )

    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        stdout_bytes, stderr_bytes = await proc.communicate()
        timed_out = True

    elapsed = time.monotonic() - start
    return CommandResult(
        command=cmd,
        exit_code=proc.returncode if proc.returncode is not None else -1,
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
        duration_seconds=elapsed,
        cwd=cwd,
        timed_out=timed_out,
        mode=mode,
    )


async def run_command(
    cmd: list[str],
    *,
    timeout: float = TIMEOUT_UTILITY_SECONDS,
    cwd: Optional[str] = None,
    extra_env: Optional[dict[str, str]] = None,
    mode: str = "python",
) -> CommandResult:
    """Run an arbitrary command list asynchronously.

    Used for raw probes (e.g. ``python --version``, ``nvidia-smi``) that
    aren't Musubi scripts. The Musubi-specific helpers below are built on
    this.
    """
    return await _run_exec(
        cmd, timeout=timeout, cwd=cwd, extra_env=extra_env, mode=mode
    )


# ---------------------------------------------------------------------------
# Musubi-aware helpers
# ---------------------------------------------------------------------------


def _resolve_script(script_name: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(script_path, cwd)`` for a Musubi script name.

    Returns ``(None, None)`` if ``MUSUBI_TUNER_DIR`` is unset or the script
    doesn't exist in that checkout. Callers surface the error.
    """
    root = musubi_tuner_dir()
    if not root:
        return None, None
    script_path = os.path.join(root, script_name)
    if not os.path.isfile(script_path):
        return None, root
    return script_path, root


async def run_musubi(
    script_name: str,
    args: Iterable[str] = (),
    *,
    timeout: float = TIMEOUT_UTILITY_SECONDS,
    extra_env: Optional[dict[str, str]] = None,
) -> CommandResult:
    """Run ``<python> <musubi>/<script> <args>`` from inside Musubi's repo root.

    Used for caching (``{arch}_cache_latents.py``,
    ``{arch}_cache_text_encoder_outputs.py``), generation
    (``{arch}_generate_*.py``), and all root-level utility scripts
    (``convert_lora.py``, ``merge_lora.py``, etc.).
    """
    script_path, cwd = _resolve_script(script_name)
    if script_path is None:
        msg = (
            f"MUSUBI_TUNER_DIR is unset — cannot resolve {script_name!r}."
            if cwd is None
            else f"script not found: {script_name!r} in {cwd}"
        )
        return CommandResult(
            command=[python_executable(), script_name, *args],
            exit_code=-1,
            stdout="",
            stderr=msg,
            duration_seconds=0.0,
            cwd=cwd,
            mode="python",
        )

    cmd = [python_executable(), script_path, *args]
    return await run_command(
        cmd, timeout=timeout, cwd=cwd, extra_env=extra_env, mode="python"
    )


async def run_musubi_training(
    script_name: str,
    args: Iterable[str] = (),
    *,
    mixed_precision: str = DEFAULT_MIXED_PRECISION,
    num_cpu_threads_per_process: int = DEFAULT_CPU_THREADS_PER_PROCESS,
    timeout: float,
    extra_env: Optional[dict[str, str]] = None,
) -> CommandResult:
    """Run a training script via ``accelerate launch``.

    Used for every ``{arch}_train_network.py`` and ``{arch}_train.py``.
    The ``mixed_precision`` arg is passed to the accelerate launcher, not
    to the training script; callers may still set ``--mixed_precision`` in
    ``args`` if they need asymmetric control (training scripts read it too).
    """
    script_path, cwd = _resolve_script(script_name)
    if script_path is None:
        msg = (
            f"MUSUBI_TUNER_DIR is unset — cannot resolve {script_name!r}."
            if cwd is None
            else f"script not found: {script_name!r} in {cwd}"
        )
        return CommandResult(
            command=[accelerate_executable(), "launch", script_name, *args],
            exit_code=-1,
            stdout="",
            stderr=msg,
            duration_seconds=0.0,
            cwd=cwd,
            mode="accelerate",
        )

    # CLAUDE-NOTE: We pass the full path to the training script (not just
    # the basename) so accelerate's own cwd-resolution doesn't matter; cwd
    # is still set so Musubi's relative imports resolve.
    cmd = [
        accelerate_executable(),
        "launch",
        "--num_cpu_threads_per_process",
        str(num_cpu_threads_per_process),
        "--mixed_precision",
        mixed_precision,
        script_path,
        *args,
    ]
    return await run_command(
        cmd, timeout=timeout, cwd=cwd, extra_env=extra_env, mode="accelerate"
    )
