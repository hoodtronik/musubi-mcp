"""Tests for the async subprocess wrapper."""
from __future__ import annotations

import os
import sys

import pytest

from musubi_mcp.runner import (
    CommandResult,
    accelerate_executable,
    build_env,
    musubi_tuner_dir,
    python_executable,
    run_command,
)


def test_python_executable_reads_env_var(monkeypatch):
    monkeypatch.setenv("MUSUBI_PYTHON", "/custom/python")
    assert python_executable() == "/custom/python"


def test_python_executable_falls_back_to_sys_executable(monkeypatch):
    monkeypatch.delenv("MUSUBI_PYTHON", raising=False)
    assert python_executable() == sys.executable


def test_musubi_tuner_dir_unset(monkeypatch):
    monkeypatch.delenv("MUSUBI_TUNER_DIR", raising=False)
    assert musubi_tuner_dir() is None


def test_musubi_tuner_dir_expands_user(monkeypatch):
    monkeypatch.setenv("MUSUBI_TUNER_DIR", "~/musubi")
    resolved = musubi_tuner_dir()
    assert resolved is not None
    assert "~" not in resolved


def test_build_env_sets_utf8_flags():
    env = build_env()
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_build_env_merges_extra_and_skips_empty():
    env = build_env({"FOO": "bar", "EMPTY": ""})
    assert env["FOO"] == "bar"
    assert "EMPTY" not in env


def test_accelerate_executable_respects_override(monkeypatch):
    monkeypatch.setenv("MUSUBI_ACCELERATE", "/weird/accelerate")
    assert accelerate_executable() == "/weird/accelerate"


def test_command_result_succeeded_property():
    r = CommandResult(
        command=["x"], exit_code=0, stdout="", stderr="", duration_seconds=0.1,
    )
    assert r.succeeded is True

    r2 = CommandResult(
        command=["x"], exit_code=1, stdout="", stderr="", duration_seconds=0.1,
    )
    assert r2.succeeded is False

    r3 = CommandResult(
        command=["x"], exit_code=0, stdout="", stderr="", duration_seconds=0.1,
        timed_out=True,
    )
    assert r3.succeeded is False


def test_command_result_to_dict_is_json_shapable():
    r = CommandResult(
        command=["x", "y"], exit_code=0, stdout="out", stderr="",
        duration_seconds=1.23, cwd="/tmp", mode="python",
    )
    d = r.to_dict()
    assert d["command"] == ["x", "y"]
    assert d["exit_code"] == 0
    assert d["mode"] == "python"


async def test_run_command_executable_not_found():
    result = await run_command(["definitely_not_a_binary_12345"], timeout=5.0)
    assert result.exit_code == -1
    assert not result.succeeded
    assert "not found" in result.stderr.lower()


async def test_run_command_real_process_succeeds():
    # Running `python -c "print('hi')"` via the host interpreter proves the
    # happy path end to end without needing Musubi Tuner.
    result = await run_command(
        [sys.executable, "-c", "print('hi')"], timeout=30.0,
    )
    assert result.succeeded, result.stderr
    assert "hi" in result.stdout
