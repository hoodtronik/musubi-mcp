"""Tests for dataset TOML generation and validation."""
from __future__ import annotations

import pytest

from musubi_mcp.dataset_config import (
    build_dataset_toml,
    dumps,
    loads,
    validate_dataset_config,
)


def test_build_image_dataset_toml_minimal():
    config = build_dataset_toml(
        image_directory="/data/images",
        cache_directory="/data/cache",
    )
    assert config["general"]["resolution"] == [960, 544]
    ds = config["datasets"][0]
    assert ds["image_directory"] == "/data/images"
    assert ds["cache_directory"] == "/data/cache"
    # Round-trip through TOML.
    parsed = loads(dumps(config))
    assert parsed == config


def test_build_video_dataset_requires_target_frames():
    with pytest.raises(ValueError, match="target_frames"):
        build_dataset_toml(
            video_directory="/data/videos",
            cache_directory="/data/cache",
            resolution=(512, 512),
        )


def test_build_video_dataset_with_target_frames():
    config = build_dataset_toml(
        video_directory="/data/videos",
        cache_directory="/data/cache",
        resolution=(512, 512),
        target_frames=[1, 25, 45],
        frame_extraction="head",
        source_fps=30.0,
    )
    ds = config["datasets"][0]
    assert ds["target_frames"] == [1, 25, 45]
    assert ds["frame_extraction"] == "head"
    assert ds["source_fps"] == 30.0


def test_build_rejects_multiple_sources():
    with pytest.raises(ValueError, match="exactly ONE"):
        build_dataset_toml(
            image_directory="/data/images",
            video_directory="/data/videos",
            cache_directory="/data/cache",
        )


def test_build_rejects_no_source():
    with pytest.raises(ValueError, match="provide one of"):
        build_dataset_toml(cache_directory="/data/cache")


def test_build_rejects_invalid_frame_extraction():
    with pytest.raises(ValueError, match="frame_extraction"):
        build_dataset_toml(
            video_directory="/v",
            cache_directory="/c",
            target_frames=[1],
            frame_extraction="banana",
        )


def test_validate_flags_4n_plus_1_violation():
    config = build_dataset_toml(
        video_directory="/v",
        cache_directory="/c",
        target_frames=[1, 24, 45],  # 24 is NOT 4n+1
        frame_extraction="head",
    )
    result = validate_dataset_config(config, enforce_4n_plus_1=True)
    assert result.ok is False
    assert any("N*4+1" in e for e in result.errors)


def test_validate_accepts_4n_plus_1():
    config = build_dataset_toml(
        video_directory="/v",
        cache_directory="/c",
        target_frames=[1, 5, 9, 13, 81],
        frame_extraction="head",
    )
    result = validate_dataset_config(config, enforce_4n_plus_1=True)
    assert result.ok is True, result.errors


def test_validate_missing_datasets_block():
    result = validate_dataset_config({"general": {}})
    assert result.ok is False
    assert any("[[datasets]]" in e for e in result.errors)


def test_validate_empty_datasets():
    result = validate_dataset_config({"general": {}, "datasets": []})
    assert result.ok is False


def test_validate_dataset_with_no_source():
    result = validate_dataset_config({
        "general": {},
        "datasets": [{"cache_directory": "/c"}],
    })
    assert result.ok is False
    assert any("needs one of" in e for e in result.errors)


def test_control_image_fields_round_trip():
    config = build_dataset_toml(
        image_directory="/img",
        cache_directory="/cache",
        control_directory="/control",
        control_resolution=(2024, 2024),
        no_resize_control=True,
    )
    ds = config["datasets"][0]
    assert ds["control_directory"] == "/control"
    assert ds["control_resolution"] == [2024, 2024]
    assert ds["no_resize_control"] is True
