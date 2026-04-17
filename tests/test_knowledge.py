"""Tests for the knowledge:// resource surface."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from musubi_mcp import knowledge


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    # Isolate each test from whatever the dev machine has exported.
    monkeypatch.delenv("KNOWLEDGE_BASE_DIR", raising=False)


def test_bundled_files_are_shipped():
    # The package's bundled folder should contain every reviewed .md file.
    src = knowledge._BUNDLED_DIR
    assert src.is_dir(), f"bundled knowledge dir missing: {src}"
    files = list(src.glob("*.md"))
    assert len(files) >= 10, (
        f"expected at least 10 bundled knowledge files, got {len(files)}. "
        "Did the hatchling include rule drop them on build?"
    )


def test_all_knowledge_names_sorted_and_no_md_suffix():
    names = knowledge.all_knowledge_names()
    assert names == sorted(names)
    for n in names:
        assert not n.endswith(".md"), n
    # Spot-check the usual suspects.
    for core in ("wan22_training", "flux2_klein_training", "ltx23_training",
                "hardware_profiles", "dataset_quality", "common_failures"):
        assert core in names, f"expected {core!r} in bundled knowledge"


def test_read_known_file_returns_full_content():
    content = knowledge.read_knowledge("wan22_training")
    # Real file starts with a header; error markers start with "[musubi-mcp]".
    assert not content.startswith("[musubi-mcp]"), content
    assert "Wan 2.2" in content
    assert "Confidence" in content  # each reviewed file has this label


def test_read_unknown_returns_error_marker_not_exception():
    content = knowledge.read_knowledge("does_not_exist_xyz")
    assert content.startswith("[musubi-mcp]")
    assert "unknown knowledge resource" in content
    assert "Available:" in content  # lists what IS available


def test_read_survives_path_with_separators():
    # Don't let a malicious name walk out of the source dir. We expect
    # the concat-then-glob approach to just not find a matching file,
    # returning the error marker rather than serving arbitrary content.
    content = knowledge.read_knowledge("../../etc/passwd")
    assert content.startswith("[musubi-mcp]")
    assert "unknown knowledge resource" in content


def test_knowledge_index_mentions_insufficient_data_rule():
    idx = knowledge.knowledge_index()
    assert "INSUFFICIENT DATA" in idx, (
        "index must explicitly tell the agent to ask the user when this "
        "flag appears — that behaviour is load-bearing, not optional"
    )
    assert "confidence" in idx.lower()
    # Index should list every bundled resource.
    for name in knowledge.all_knowledge_names():
        assert f"`knowledge://{name}`" in idx, f"{name} missing from index"


def test_env_var_override_lives_syncs(tmp_path: Path, monkeypatch):
    # Stand up a fresh knowledge dir, point the env var at it, and
    # confirm the module reads from there without any reload.
    (tmp_path / "my_override.md").write_text("# custom\n", encoding="utf-8")
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path))
    assert knowledge.all_knowledge_names() == ["my_override"]
    assert knowledge.read_knowledge("my_override").startswith("# custom")


def test_env_var_override_ignored_when_path_missing(tmp_path: Path, monkeypatch):
    # If the env var points at something that doesn't exist, we fall
    # back to the bundled copy — don't crash, don't serve an empty set.
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "nonexistent"))
    names = knowledge.all_knowledge_names()
    assert len(names) >= 10, "should have fallen back to bundled copy"


def test_insufficient_data_files_are_not_filtered():
    # Every reviewed file (including REVIEW_NOTES.md with its full INSUFFICIENT
    # DATA tracking) must be exposed. The flag is the point — filtering would
    # hide it from the orchestration agent, which is the opposite of what we want.
    names = knowledge.all_knowledge_names()
    assert "REVIEW_NOTES" in names, (
        "REVIEW_NOTES must be exposed verbatim so agents see the "
        "INSUFFICIENT DATA flag inventory"
    )
    content = knowledge.read_knowledge("REVIEW_NOTES")
    # Content should be the raw file — no redaction.
    assert "REVIEW NOTES" in content


def test_resources_registered_on_server():
    # Importing server.py runs the registration loop at module load time.
    from musubi_mcp import server

    rm = server.mcp._resource_manager
    uris = list(rm._resources.keys()) if hasattr(rm, "_resources") else []
    knowledge_uris = [u for u in uris if str(u).startswith("knowledge://")]
    # All files + the index resource.
    expected = len(knowledge.all_knowledge_names()) + 1
    assert len(knowledge_uris) >= expected, (
        f"expected at least {expected} knowledge:// resources on the server, "
        f"got {len(knowledge_uris)}. URIs: {knowledge_uris[:20]}"
    )
    # Index specifically.
    assert any("knowledge://index" in str(u) for u in knowledge_uris), (
        "knowledge://index missing from the registered resource list"
    )
