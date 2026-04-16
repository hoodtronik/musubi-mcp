"""Tests for the second batch of tools + prompts.

Covers: musubi_finetune, musubi_generate, musubi_validate_dataset,
musubi_convert_lora, musubi_merge_lora, musubi_ema_merge,
musubi_caption_images, plan_training_run, diagnose_training_issue.

All subprocess-launching tools are tested with the same monkeypatch
strategy used in test_server_tools.py: patch ``run_musubi`` /
``run_musubi_training`` on the server module to capture argv without
actually spawning anything.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from musubi_mcp import server
from musubi_mcp.runner import CommandResult


@pytest.fixture
def fake_dataset_toml(tmp_path: Path) -> str:
    p = tmp_path / "dataset.toml"
    p.write_text("# placeholder\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def capture_run(monkeypatch):
    captured: dict[str, Any] = {}

    async def fake_run_musubi(script_name, args=(), *, timeout, extra_env=None):
        captured["mode"] = "python"
        captured["script"] = script_name
        captured["args"] = list(args)
        captured["timeout"] = timeout
        captured["extra_env"] = extra_env
        return CommandResult(
            command=["<fake>", script_name, *args],
            exit_code=0, stdout="ok", stderr="",
            duration_seconds=0.01, cwd=None, mode="python",
        )

    async def fake_run_musubi_training(
        script_name, args=(), *, mixed_precision,
        num_cpu_threads_per_process, timeout, extra_env=None,
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
            exit_code=0, stdout="ok", stderr="",
            duration_seconds=0.01, cwd=None, mode="accelerate",
        )

    monkeypatch.setattr(server, "run_musubi", fake_run_musubi)
    monkeypatch.setattr(server, "run_musubi_training", fake_run_musubi_training)
    return captured


# ---------------------------------------------------------------------------
# musubi_finetune
# ---------------------------------------------------------------------------


async def test_finetune_refuses_arch_without_full_train_script(fake_dataset_toml):
    # wan has no train_full_script → tool must refuse.
    result = await server.musubi_finetune(
        architecture="wan",
        dataset_config=fake_dataset_toml,
        output_dir="/out",
        output_name="test",
        dit="/d.safetensors",
    )
    assert result["ok"] is False
    assert "full-fine-tune" in result["error"]


async def test_finetune_zimage_requires_text_encoder(fake_dataset_toml):
    result = await server.musubi_finetune(
        architecture="zimage",
        dataset_config=fake_dataset_toml,
        output_dir="/out",
        output_name="test",
        dit="/d.safetensors",
    )
    assert result["ok"] is False
    assert "text_encoder" in result["error"]


async def test_finetune_zimage_argv(fake_dataset_toml, capture_run):
    result = await server.musubi_finetune(
        architecture="zimage",
        dataset_config=fake_dataset_toml,
        output_dir="/out",
        output_name="zimage_ft",
        dit="/models/z.safetensors",
        vae="/models/z_vae.safetensors",
        text_encoder="/models/qwen3.safetensors",
        learning_rate=5e-5,
        max_train_steps=2000,
        fp8_base=True,
        blocks_to_swap=10,
        gradient_accumulation_steps=4,
        extra_args=["--block_swap_optimizer_patch_params"],
    )
    assert result["ok"] is True
    assert capture_run["mode"] == "accelerate"
    assert capture_run["script"] == "zimage_train.py"
    argv = capture_run["args"]
    assert "--dit" in argv and "/models/z.safetensors" in argv
    assert "--text_encoder" in argv and "/models/qwen3.safetensors" in argv
    assert "--fp8_base" in argv
    assert "--blocks_to_swap" in argv and "10" in argv
    assert "--gradient_accumulation_steps" in argv and "4" in argv
    assert "--block_swap_optimizer_patch_params" in argv
    # Confirm no network_module flag present — finetune is not a LoRA tool.
    assert "--network_module" not in argv


# ---------------------------------------------------------------------------
# musubi_generate
# ---------------------------------------------------------------------------


async def test_generate_refuses_placeholder_arch():
    result = await server.musubi_generate(
        architecture="hv", prompt="hi", dit="/d.safetensors",
    )
    assert result["ok"] is False


async def test_generate_wan_requires_t5():
    result = await server.musubi_generate(
        architecture="wan", prompt="hi", dit="/d.safetensors",
    )
    assert result["ok"] is False
    assert "t5" in result["error"].lower()


async def test_generate_lora_weight_and_multiplier_length_mismatch():
    result = await server.musubi_generate(
        architecture="wan",
        prompt="hi",
        dit="/d.safetensors",
        t5="/t5.pth",
        lora_weight=["/a.safetensors", "/b.safetensors"],
        lora_multiplier=[1.0],  # length mismatch
    )
    assert result["ok"] is False
    assert "same length" in result["error"]


async def test_generate_wan_i2v_argv_with_dual_model(capture_run):
    result = await server.musubi_generate(
        architecture="wan",
        prompt="a dog running",
        negative_prompt="blurry",
        dit="/dit_low.safetensors",
        vae="/vae.safetensors",
        t5="/t5.pth",
        clip="/clip.pth",
        task="i2v-A14B",
        image_path="/input.png",
        dit_high_noise="/dit_high.safetensors",
        lora_weight=["/lora.safetensors"],
        lora_multiplier=[0.8],
        lora_weight_high_noise=["/lora_hi.safetensors"],
        lora_multiplier_high_noise=[1.0],
        timestep_boundary=0.9,
        guidance_scale=5.0,
        guidance_scale_high_noise=4.0,
        video_size_w=832,
        video_size_h=480,
        video_length=81,
        fps=16,
        infer_steps=20,
        seed=42,
        fp8=True,
        fp8_scaled=True,
        fp8_t5=True,
        blocks_to_swap=20,
        attn_mode="sdpa",
        save_path="/out.mp4",
    )
    assert result["ok"] is True
    assert capture_run["script"] == "wan_generate_video.py"
    argv = capture_run["args"]
    assert "--prompt" in argv and "a dog running" in argv
    assert "--negative_prompt" in argv
    assert "--task" in argv and "i2v-A14B" in argv
    assert "--image_path" in argv and "/input.png" in argv
    assert "--dit_high_noise" in argv and "/dit_high.safetensors" in argv
    assert "--lora_weight_high_noise" in argv and "/lora_hi.safetensors" in argv
    assert "--lora_multiplier_high_noise" in argv and "1.0" in argv
    assert "--timestep_boundary" in argv and "0.9" in argv
    assert "--guidance_scale_high_noise" in argv and "4.0" in argv
    assert "--video_size" in argv and "832" in argv and "480" in argv
    assert "--video_length" in argv and "81" in argv
    assert "--fps" in argv and "16" in argv
    assert "--fp8" in argv
    assert "--fp8_scaled" in argv
    assert "--fp8_t5" in argv
    assert "--attn_mode" in argv and "sdpa" in argv
    assert "--save_path" in argv and "/out.mp4" in argv


async def test_generate_flux2_image_argv(capture_run):
    result = await server.musubi_generate(
        architecture="flux_2",
        prompt="a cat",
        dit="/dit.safetensors",
        vae="/ae.safetensors",
        text_encoder="/mistral.safetensors",
        model_version="klein-4b",
        image_size_w=1024,
        image_size_h=1024,
        infer_steps=28,
        guidance_scale=3.5,
        embedded_cfg_scale=2.5,
        fp8_text_encoder=True,
        save_path="/out.png",
    )
    assert result["ok"] is True
    assert capture_run["script"] == "flux_2_generate_image.py"
    argv = capture_run["args"]
    assert "--model_version" in argv and "klein-4b" in argv
    assert "--image_size" in argv and "1024" in argv
    assert "--embedded_cfg_scale" in argv and "2.5" in argv
    assert "--fp8_text_encoder" in argv
    # Image archs should not see video-only flags.
    assert "--video_size" not in argv
    assert "--video_length" not in argv


async def test_generate_zimage_uses_fp8_llm(capture_run):
    result = await server.musubi_generate(
        architecture="zimage",
        prompt="portrait",
        dit="/dit.safetensors",
        vae="/vae.safetensors",
        text_encoder="/qwen3.safetensors",
        fp8_llm=True,
    )
    assert result["ok"] is True
    argv = capture_run["args"]
    assert "--fp8_llm" in argv
    assert "--fp8_text_encoder" not in argv


# ---------------------------------------------------------------------------
# musubi_validate_dataset (filesystem)
# ---------------------------------------------------------------------------


async def test_validate_dataset_missing_directory():
    result = await server.musubi_validate_dataset(directory="/totally/missing")
    assert result["ok"] is False
    assert "not a directory" in result["error"]


async def test_validate_dataset_image_happy_path(tmp_path):
    # Create 3 jpgs with matching .txt sidecars.
    for i in range(3):
        (tmp_path / f"img_{i}.jpg").write_bytes(b"")
        (tmp_path / f"img_{i}.txt").write_text(f"caption {i}", encoding="utf-8")

    result = await server.musubi_validate_dataset(
        directory=str(tmp_path), dataset_kind="image",
    )
    assert result["ok"] is True
    assert result["counts"]["media_files"] == 3
    assert result["counts"]["missing_captions"] == 0
    assert result["counts"]["empty_captions"] == 0
    assert result["counts"]["by_extension"][".jpg"] == 3


async def test_validate_dataset_reports_missing_and_empty_captions(tmp_path):
    # 2 images with captions, 1 missing, 1 empty.
    (tmp_path / "a.png").write_bytes(b"")
    (tmp_path / "a.txt").write_text("good", encoding="utf-8")
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "b.txt").write_text("   \n", encoding="utf-8")  # empty after strip
    (tmp_path / "c.png").write_bytes(b"")  # no sidecar

    result = await server.musubi_validate_dataset(
        directory=str(tmp_path), dataset_kind="image",
    )
    assert result["ok"] is False
    assert result["counts"]["missing_captions"] == 1
    assert result["counts"]["empty_captions"] == 1
    assert any("c.png" in m for m in result["missing_captions_sample"])


async def test_validate_dataset_video_kind_uses_video_exts(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"")
    (tmp_path / "clip.txt").write_text("caption", encoding="utf-8")
    (tmp_path / "ignored.png").write_bytes(b"")  # not a video

    result = await server.musubi_validate_dataset(
        directory=str(tmp_path), dataset_kind="video",
    )
    assert result["ok"] is True
    assert result["counts"]["media_files"] == 1


async def test_validate_dataset_recurses_into_subfolders(tmp_path):
    sub = tmp_path / "concept1"
    sub.mkdir()
    (sub / "a.png").write_bytes(b"")
    (sub / "a.txt").write_text("x", encoding="utf-8")
    result = await server.musubi_validate_dataset(
        directory=str(tmp_path), dataset_kind="image",
    )
    assert result["counts"]["media_files"] == 1


async def test_validate_dataset_rejects_bad_kind():
    result = await server.musubi_validate_dataset(
        directory="/", dataset_kind="audio",
    )
    assert result["ok"] is False
    assert "dataset_kind" in result["error"]


# ---------------------------------------------------------------------------
# musubi_convert_lora
# ---------------------------------------------------------------------------


async def test_convert_lora_rejects_bad_target(tmp_path):
    inp = tmp_path / "lora.safetensors"
    inp.write_bytes(b"")
    result = await server.musubi_convert_lora(
        input=str(inp), output=str(tmp_path / "out.safetensors"), target="sideways",
    )
    assert result["ok"] is False
    assert "target" in result["error"]


async def test_convert_lora_requires_input_exists(tmp_path):
    result = await server.musubi_convert_lora(
        input="/no/such/file.safetensors",
        output=str(tmp_path / "out.safetensors"),
        target="other",
    )
    assert result["ok"] is False
    assert "not found" in result["error"]


async def test_convert_lora_argv(tmp_path, capture_run):
    inp = tmp_path / "lora.safetensors"
    inp.write_bytes(b"")
    result = await server.musubi_convert_lora(
        input=str(inp),
        output=str(tmp_path / "out.safetensors"),
        target="other",
        diffusers_prefix="custom_prefix",
    )
    assert result["ok"] is True
    assert capture_run["script"] == "convert_lora.py"
    argv = capture_run["args"]
    assert "--input" in argv
    assert "--target" in argv and "other" in argv
    assert "--diffusers_prefix" in argv and "custom_prefix" in argv


# ---------------------------------------------------------------------------
# musubi_merge_lora
# ---------------------------------------------------------------------------


async def test_merge_lora_requires_at_least_one_lora(tmp_path, capture_run):
    dit = tmp_path / "dit.safetensors"
    dit.write_bytes(b"")
    result = await server.musubi_merge_lora(
        dit=str(dit),
        save_merged_model=str(tmp_path / "merged.safetensors"),
        lora_weight=[],
    )
    assert result["ok"] is False
    assert "lora_weight" in result["error"]


async def test_merge_lora_rejects_length_mismatch(tmp_path):
    dit = tmp_path / "dit.safetensors"
    dit.write_bytes(b"")
    l1 = tmp_path / "l1.safetensors"
    l2 = tmp_path / "l2.safetensors"
    l1.write_bytes(b""); l2.write_bytes(b"")
    result = await server.musubi_merge_lora(
        dit=str(dit),
        save_merged_model=str(tmp_path / "out.safetensors"),
        lora_weight=[str(l1), str(l2)],
        lora_multiplier=[0.8],
    )
    assert result["ok"] is False
    assert "must match" in result["error"]


async def test_merge_lora_argv(tmp_path, capture_run):
    dit = tmp_path / "dit.safetensors"
    dit.write_bytes(b"")
    l1 = tmp_path / "l1.safetensors"
    l2 = tmp_path / "l2.safetensors"
    l1.write_bytes(b""); l2.write_bytes(b"")

    result = await server.musubi_merge_lora(
        dit=str(dit),
        save_merged_model=str(tmp_path / "merged.safetensors"),
        lora_weight=[str(l1), str(l2)],
        lora_multiplier=[0.8, 1.2],
        dit_in_channels=32,
    )
    assert result["ok"] is True
    assert capture_run["script"] == "merge_lora.py"
    argv = capture_run["args"]
    assert "--dit" in argv
    assert "--save_merged_model" in argv
    assert argv.count("--lora_weight") == 1
    # Both weight paths should follow the single --lora_weight flag.
    idx = argv.index("--lora_weight")
    assert str(l1) in argv[idx + 1 : idx + 3]
    assert str(l2) in argv[idx + 1 : idx + 3]
    assert "--dit_in_channels" in argv and "32" in argv


# ---------------------------------------------------------------------------
# musubi_ema_merge
# ---------------------------------------------------------------------------


async def test_ema_merge_requires_two_checkpoints(tmp_path):
    c = tmp_path / "a.safetensors"
    c.write_bytes(b"")
    result = await server.musubi_ema_merge(
        checkpoint_paths=[str(c)],
        output_file=str(tmp_path / "out.safetensors"),
    )
    assert result["ok"] is False


async def test_ema_merge_argv_positional_order(tmp_path, capture_run):
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    c = tmp_path / "c.safetensors"
    for p in (a, b, c):
        p.write_bytes(b"")

    result = await server.musubi_ema_merge(
        checkpoint_paths=[str(a), str(b), str(c)],
        output_file=str(tmp_path / "ema.safetensors"),
        beta=0.95,
        sigma_rel=0.3,
        no_sort=True,
    )
    assert result["ok"] is True
    argv = capture_run["args"]
    assert capture_run["script"] == "lora_post_hoc_ema.py"
    # Positional paths come last.
    assert argv[-3:] == [str(a), str(b), str(c)]
    assert "--no_sort" in argv
    assert "--beta" in argv and "0.95" in argv
    assert "--sigma_rel" in argv and "0.3" in argv
    assert "--output_file" in argv


# ---------------------------------------------------------------------------
# musubi_caption_images
# ---------------------------------------------------------------------------


async def test_caption_images_jsonl_requires_output_file(tmp_path):
    result = await server.musubi_caption_images(
        image_dir=str(tmp_path),
        model_path="/m",
        output_format="jsonl",
    )
    assert result["ok"] is False
    assert "output_file" in result["error"]


async def test_caption_images_rejects_bad_format(tmp_path):
    result = await server.musubi_caption_images(
        image_dir=str(tmp_path),
        model_path="/m",
        output_format="xml",
    )
    assert result["ok"] is False


async def test_caption_images_text_mode_argv(tmp_path, capture_run):
    result = await server.musubi_caption_images(
        image_dir=str(tmp_path),
        model_path="/models/qwen_vl",
        output_format="text",
        max_new_tokens=512,
        prompt="describe",
        max_size=1024,
        fp8_vl=True,
    )
    assert result["ok"] is True
    assert capture_run["script"] == "caption_images_by_qwen_vl.py"
    argv = capture_run["args"]
    assert "--image_dir" in argv
    assert "--model_path" in argv
    assert "--output_format" in argv and "text" in argv
    assert "--max_new_tokens" in argv and "512" in argv
    assert "--prompt" in argv and "describe" in argv
    assert "--max_size" in argv and "1024" in argv
    assert "--fp8_vl" in argv


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_plan_training_run_substitutes_context():
    rendered = server.plan_training_run(
        architecture="wan",
        training_type="lora",
        source_data="50 clips of a character",
        hardware="RTX 4090 24GB",
        goal="character LoRA for Wan2.2 I2V",
    )
    assert "wan" in rendered
    assert "character LoRA for Wan2.2 I2V" in rendered
    assert "musubi_cache_latents" in rendered
    assert "musubi_cache_text_encoder" in rendered
    assert "musubi_train" in rendered
    assert "4n+1" in rendered


def test_diagnose_training_issue_includes_log():
    log = "CUDA out of memory. Tried to allocate 1.23 GiB"
    rendered = server.diagnose_training_issue(
        log_output=log,
        architecture="flux_2",
        training_type="lora",
    )
    assert log in rendered
    assert "flux_2" in rendered
    assert "blocks_to_swap" in rendered  # OOM hint surface
    assert "fp8_base" in rendered
