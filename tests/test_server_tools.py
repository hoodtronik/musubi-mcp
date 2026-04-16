"""Tests for the tool layer — arg building, validation gates, routing.

These tests never launch real training. Two strategies:

- Unknown / placeholder archs: the tool returns an ``{"ok": False, "error": ...}``
  before any subprocess is spawned. We assert the error text.
- Live archs: we monkeypatch the subprocess runner (``run_musubi`` /
  ``run_musubi_training``) to capture the argv the tool would have called
  and return a fake successful ``CommandResult``. We assert the argv
  contains the right flags in the right positions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from musubi_mcp import server
from musubi_mcp.runner import CommandResult


@pytest.fixture
def fake_dataset_toml(tmp_path: Path) -> str:
    # CLAUDE-NOTE: Tools that take a dataset_config path call os.path.isfile
    # on it before spawning anything. A real (if empty) file satisfies that
    # guard without needing to build a valid TOML for these routing tests —
    # the validity tests live in test_dataset_config.py.
    p = tmp_path / "dataset.toml"
    p.write_text("# placeholder\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def capture_run(monkeypatch):
    """Patch ``run_musubi`` and ``run_musubi_training`` to record calls.

    Returns the dict that will be populated with ``script_name``, ``args``,
    ``mixed_precision``, etc. after the tool under test calls the runner.
    """
    captured: dict[str, Any] = {}

    async def fake_run_musubi(script_name, args=(), *, timeout, extra_env=None):
        captured["mode"] = "python"
        captured["script"] = script_name
        captured["args"] = list(args)
        captured["timeout"] = timeout
        captured["extra_env"] = extra_env
        return CommandResult(
            command=["<fake>", script_name, *args],
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.01,
            cwd=None,
            mode="python",
        )

    async def fake_run_musubi_training(
        script_name,
        args=(),
        *,
        mixed_precision,
        num_cpu_threads_per_process,
        timeout,
        extra_env=None,
    ):
        captured["mode"] = "accelerate"
        captured["script"] = script_name
        captured["args"] = list(args)
        captured["mixed_precision"] = mixed_precision
        captured["num_cpu_threads_per_process"] = num_cpu_threads_per_process
        captured["timeout"] = timeout
        captured["extra_env"] = extra_env
        return CommandResult(
            command=["<fake>", "launch", script_name, *args],
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.01,
            cwd=None,
            mode="accelerate",
        )

    monkeypatch.setattr(server, "run_musubi", fake_run_musubi)
    monkeypatch.setattr(server, "run_musubi_training", fake_run_musubi_training)
    return captured


# ---------------------------------------------------------------------------
# musubi_list_architectures
# ---------------------------------------------------------------------------


async def test_list_architectures_returns_nine_with_three_live():
    result = await server.musubi_list_architectures()
    assert result["count"] == 9
    assert result["live_count"] == 3
    live_names = {a["name"] for a in result["architectures"] if a["live"]}
    assert live_names == {"wan", "flux_2", "zimage"}


# ---------------------------------------------------------------------------
# musubi_cache_latents
# ---------------------------------------------------------------------------


async def test_cache_latents_unknown_arch(fake_dataset_toml):
    result = await server.musubi_cache_latents(
        architecture="not_a_real", dataset_config=fake_dataset_toml,
    )
    assert result["ok"] is False
    assert "unknown" in result["error"].lower()


async def test_cache_latents_placeholder_arch_refused(fake_dataset_toml):
    result = await server.musubi_cache_latents(
        architecture="hv", dataset_config=fake_dataset_toml,
    )
    assert result["ok"] is False
    assert "placeholder" in result["error"].lower()


async def test_cache_latents_rejects_missing_dataset_config():
    result = await server.musubi_cache_latents(
        architecture="wan", dataset_config="/does/not/exist.toml",
    )
    assert result["ok"] is False
    assert "dataset_config not found" in result["error"]


async def test_cache_latents_flux2_requires_model_version(fake_dataset_toml):
    result = await server.musubi_cache_latents(
        architecture="flux_2", dataset_config=fake_dataset_toml,
    )
    assert result["ok"] is False
    assert "model_version" in result["error"]


async def test_cache_latents_wan_builds_expected_argv(
    fake_dataset_toml, capture_run,
):
    result = await server.musubi_cache_latents(
        architecture="wan",
        dataset_config=fake_dataset_toml,
        vae="/models/wan_vae.safetensors",
        batch_size=4,
        skip_existing=True,
        i2v=True,
        clip="/models/wan_clip.safetensors",
    )
    assert result["ok"] is True
    assert capture_run["script"] == "wan_cache_latents.py"
    argv = capture_run["args"]
    assert "--dataset_config" in argv
    assert "--vae" in argv
    assert "/models/wan_vae.safetensors" in argv
    assert "--batch_size" in argv and "4" in argv
    assert "--skip_existing" in argv
    assert "--i2v" in argv
    assert "--clip" in argv
    assert "/models/wan_clip.safetensors" in argv


async def test_cache_latents_flux2_passes_model_version(
    fake_dataset_toml, capture_run,
):
    result = await server.musubi_cache_latents(
        architecture="flux_2",
        dataset_config=fake_dataset_toml,
        model_version="klein-4b",
        vae="/models/flux_vae.safetensors",
    )
    assert result["ok"] is True
    argv = capture_run["args"]
    assert capture_run["script"] == "flux_2_cache_latents.py"
    idx = argv.index("--model_version")
    assert argv[idx + 1] == "klein-4b"


async def test_cache_latents_flux2_rejects_invalid_model_version(fake_dataset_toml):
    result = await server.musubi_cache_latents(
        architecture="flux_2",
        dataset_config=fake_dataset_toml,
        model_version="nonsense-99z",
    )
    assert result["ok"] is False
    assert "invalid model_version" in result["error"]


# ---------------------------------------------------------------------------
# musubi_cache_text_encoder
# ---------------------------------------------------------------------------


async def test_cache_text_encoder_wan_requires_t5(fake_dataset_toml):
    result = await server.musubi_cache_text_encoder(
        architecture="wan", dataset_config=fake_dataset_toml,
    )
    assert result["ok"] is False
    assert "t5" in result["error"].lower()


async def test_cache_text_encoder_flux2_dev_rejects_fp8_text_encoder(
    fake_dataset_toml,
):
    result = await server.musubi_cache_text_encoder(
        architecture="flux_2",
        dataset_config=fake_dataset_toml,
        text_encoder="/models/mistral3.safetensors",
        model_version="dev",
        fp8_text_encoder=True,
    )
    assert result["ok"] is False
    assert "dev" in result["error"].lower()
    assert "fp8_text_encoder" in result["error"].lower()


async def test_cache_text_encoder_wan_builds_argv(
    fake_dataset_toml, capture_run,
):
    result = await server.musubi_cache_text_encoder(
        architecture="wan",
        dataset_config=fake_dataset_toml,
        t5="/models/t5.pth",
        clip="/models/clip.pth",
        fp8_t5=True,
    )
    assert result["ok"] is True
    assert capture_run["script"] == "wan_cache_text_encoder_outputs.py"
    argv = capture_run["args"]
    assert "--t5" in argv and "/models/t5.pth" in argv
    assert "--clip" in argv and "/models/clip.pth" in argv
    assert "--fp8_t5" in argv


async def test_cache_text_encoder_zimage_uses_fp8_llm(
    fake_dataset_toml, capture_run,
):
    result = await server.musubi_cache_text_encoder(
        architecture="zimage",
        dataset_config=fake_dataset_toml,
        text_encoder="/models/qwen3.safetensors",
        fp8_llm=True,
    )
    assert result["ok"] is True
    argv = capture_run["args"]
    assert "--fp8_llm" in argv
    assert "--fp8_text_encoder" not in argv, "Z-Image uses fp8_llm, not fp8_text_encoder"


# ---------------------------------------------------------------------------
# musubi_create_dataset_config + musubi_validate_dataset_config
# ---------------------------------------------------------------------------


async def test_create_dataset_config_writes_image_toml(tmp_path):
    out = tmp_path / "ds.toml"
    result = await server.musubi_create_dataset_config(
        output_path=str(out),
        image_directory=str(tmp_path / "images"),
        cache_directory=str(tmp_path / "cache"),
    )
    assert result["ok"] is True
    assert out.exists()
    assert "image_directory" in result["toml_text"]
    assert result["dataset_type"] == "image"


async def test_create_dataset_config_video_round_trip_and_validates(tmp_path):
    out = tmp_path / "video_ds.toml"
    result = await server.musubi_create_dataset_config(
        output_path=str(out),
        video_directory=str(tmp_path / "videos"),
        cache_directory=str(tmp_path / "cache"),
        resolution_w=512,
        resolution_h=512,
        target_frames=[1, 25, 45],
        frame_extraction="head",
    )
    assert result["ok"] is True
    validation = await server.musubi_validate_dataset_config(
        path=str(out), architecture="wan",
    )
    assert validation["ok"] is True
    assert validation["enforced_4n_plus_1"] is True


async def test_validate_dataset_config_flags_4n_plus_1_violation(tmp_path):
    out = tmp_path / "bad.toml"
    await server.musubi_create_dataset_config(
        output_path=str(out),
        video_directory=str(tmp_path / "videos"),
        cache_directory=str(tmp_path / "cache"),
        target_frames=[1, 24, 45],  # 24 is not 4n+1
        frame_extraction="head",
    )
    validation = await server.musubi_validate_dataset_config(
        path=str(out), architecture="wan",
    )
    assert validation["ok"] is False
    assert any("4+1" in e or "4n+1" in e.replace("*", "n").lower() or "N*4+1" in e for e in validation["errors"])


async def test_validate_dataset_config_missing_file():
    result = await server.musubi_validate_dataset_config(path="/nope.toml")
    assert result["ok"] is False


async def test_create_dataset_config_rejects_multiple_sources(tmp_path):
    result = await server.musubi_create_dataset_config(
        output_path=str(tmp_path / "ds.toml"),
        image_directory=str(tmp_path / "i"),
        video_directory=str(tmp_path / "v"),
        cache_directory=str(tmp_path / "c"),
    )
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# musubi_train
# ---------------------------------------------------------------------------


async def test_train_refuses_placeholder(fake_dataset_toml):
    result = await server.musubi_train(
        architecture="hv",
        dataset_config=fake_dataset_toml,
        output_dir="/out",
        output_name="test",
        dit="/models/dit.safetensors",
    )
    assert result["ok"] is False
    assert "placeholder" in result["error"].lower()


async def test_train_wan_requires_t5(fake_dataset_toml):
    result = await server.musubi_train(
        architecture="wan",
        dataset_config=fake_dataset_toml,
        output_dir="/out",
        output_name="test",
        dit="/models/dit.safetensors",
    )
    assert result["ok"] is False
    assert "t5" in result["error"].lower()


async def test_train_wan_rejects_invalid_task(fake_dataset_toml):
    result = await server.musubi_train(
        architecture="wan",
        dataset_config=fake_dataset_toml,
        output_dir="/out",
        output_name="test",
        dit="/models/dit.safetensors",
        t5="/models/t5.pth",
        task="not-a-task",
    )
    assert result["ok"] is False
    assert "invalid task" in result["error"].lower()


async def test_train_wan_full_argv(fake_dataset_toml, capture_run):
    result = await server.musubi_train(
        architecture="wan",
        dataset_config=fake_dataset_toml,
        output_dir="/out",
        output_name="character_lora",
        dit="/models/dit_low.safetensors",
        vae="/models/wan_vae.safetensors",
        t5="/models/t5.pth",
        clip="/models/clip.pth",
        task="i2v-A14B",
        network_dim=32,
        network_alpha=16.0,
        learning_rate=1e-4,
        max_train_epochs=10,
        save_every_n_epochs=2,
        optimizer_type="AdamW8bit",
        fp8_base=True,
        gradient_checkpointing=True,
        blocks_to_swap=20,
        timestep_sampling="shift",
        discrete_flow_shift=3.0,
        dit_high_noise="/models/dit_high.safetensors",
        timestep_boundary=0.875,
    )
    assert result["ok"] is True, result.get("error")
    assert capture_run["mode"] == "accelerate"
    assert capture_run["script"] == "wan_train_network.py"
    argv = capture_run["args"]
    # Core plumbing.
    assert "--dataset_config" in argv
    assert "--output_dir" in argv and "/out" in argv
    assert "--output_name" in argv and "character_lora" in argv
    assert "--dit" in argv and "/models/dit_low.safetensors" in argv
    assert "--vae" in argv and "/models/wan_vae.safetensors" in argv
    assert "--t5" in argv and "/models/t5.pth" in argv
    assert "--clip" in argv and "/models/clip.pth" in argv
    # Network defaults via registry.
    idx = argv.index("--network_module")
    assert argv[idx + 1] == "networks.lora_wan"
    # Task + Wan2.2 dual-model.
    assert "--task" in argv and "i2v-A14B" in argv
    assert "--dit_high_noise" in argv and "/models/dit_high.safetensors" in argv
    assert "--timestep_boundary" in argv and "0.875" in argv
    # Memory + optimizer.
    assert "--fp8_base" in argv
    assert "--gradient_checkpointing" in argv
    assert "--blocks_to_swap" in argv and "20" in argv
    assert "--optimizer_type" in argv and "AdamW8bit" in argv


async def test_train_flux2_requires_model_version(fake_dataset_toml):
    result = await server.musubi_train(
        architecture="flux_2",
        dataset_config=fake_dataset_toml,
        output_dir="/out",
        output_name="test",
        dit="/models/flux_dit.safetensors",
        text_encoder="/models/mistral.safetensors",
    )
    assert result["ok"] is False
    assert "model_version" in result["error"]


async def test_train_zimage_lora_routing(fake_dataset_toml, capture_run):
    result = await server.musubi_train(
        architecture="zimage",
        dataset_config=fake_dataset_toml,
        output_dir="/out",
        output_name="z_lora",
        dit="/models/z_dit.safetensors",
        text_encoder="/models/qwen3.safetensors",
        vae="/models/z_vae.safetensors",
        network_dim=16,
    )
    assert result["ok"] is True
    assert capture_run["script"] == "zimage_train_network.py"
    argv = capture_run["args"]
    idx = argv.index("--network_module")
    assert argv[idx + 1] == "networks.lora_zimage"
    assert "--text_encoder" in argv
    assert "/models/qwen3.safetensors" in argv


async def test_train_override_network_module_for_loha(fake_dataset_toml, capture_run):
    result = await server.musubi_train(
        architecture="zimage",
        dataset_config=fake_dataset_toml,
        output_dir="/out",
        output_name="z_loha",
        dit="/models/z_dit.safetensors",
        text_encoder="/models/qwen3.safetensors",
        network_module="networks.loha_zimage",  # user override
    )
    assert result["ok"] is True
    argv = capture_run["args"]
    idx = argv.index("--network_module")
    assert argv[idx + 1] == "networks.loha_zimage"
