"""Tests for the architecture registry."""
from __future__ import annotations

from musubi_mcp.architectures import (
    all_architectures,
    architecture_names,
    get_architecture,
    live_architectures,
)


def test_registry_is_non_empty():
    archs = all_architectures()
    assert len(archs) >= 9, "expected at least 9 architectures registered"


def test_live_architectures_are_the_three_priorities():
    live_names = {a.name for a in live_architectures()}
    assert live_names == {"wan", "flux_2", "zimage"}


def test_wan_has_t5_required_and_clip_optional():
    wan = get_architecture("wan")
    assert wan is not None
    assert wan.live is True
    assert wan.type == "video"
    flags = {te.flag: te for te in wan.text_encoders}
    assert flags["--t5"].required is True
    assert flags["--clip"].required is False
    assert wan.network_module == "networks.lora_wan"
    assert wan.cache_latents_script == "wan_cache_latents.py"
    assert wan.train_network_script == "wan_train_network.py"
    assert wan.generate_script == "wan_generate_video.py"
    assert "t2v-A14B" in (wan.tasks or ())
    assert "i2v-A14B" in (wan.tasks or ())


def test_flux_2_requires_model_version_enum():
    flux = get_architecture("flux_2")
    assert flux is not None
    assert flux.live is True
    assert flux.model_versions is not None
    assert set(flux.model_versions) >= {"dev", "klein-4b", "klein-9b"}
    # CLAUDE-NOTE: the FLUX.2 docs refer to its VAE as AE, but the flag
    # is --vae in Musubi's CLI. Regression guard for someone "fixing" it.
    assert flux.vae_arg == "--vae"
    assert flux.network_module == "networks.lora_flux_2"
    assert flux.train_full_script is None, "FLUX.2 has no full fine-tune script"


def test_zimage_supports_full_fine_tune():
    z = get_architecture("zimage")
    assert z is not None
    assert z.live is True
    assert z.train_full_script == "zimage_train.py"
    assert z.network_module == "networks.lora_zimage"


def test_get_unknown_architecture_returns_none():
    assert get_architecture("not_a_real_arch") is None


def test_placeholder_architectures_are_registered_but_not_live():
    for name in ("hv", "hv_1_5", "fpack", "flux_kontext", "qwen_image", "kandinsky5"):
        arch = get_architecture(name)
        assert arch is not None, f"{name} should be registered"
        assert arch.live is False, f"{name} should still be a placeholder"


def test_architecture_names_contains_all_known():
    names = set(architecture_names())
    expected = {
        "wan", "flux_2", "zimage",
        "hv", "hv_1_5", "fpack", "flux_kontext", "qwen_image", "kandinsky5",
    }
    assert names >= expected


def test_to_public_dict_shape():
    wan = get_architecture("wan")
    public = wan.to_public_dict()
    assert public["name"] == "wan"
    assert public["capabilities"]["cache_latents"] is True
    assert public["model_args"]["vae"] == "--vae"
    assert any(te["flag"] == "--t5" for te in public["model_args"]["text_encoders"])
