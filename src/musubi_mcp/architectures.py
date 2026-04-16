"""Architecture registry for Musubi Tuner.

Maps architecture names (``wan``, ``flux_2``, ``zimage``, ...) to the set of
scripts, required model arg flags, and supported tasks/model_versions that
Musubi Tuner actually exposes. The tool layer consults this registry to:

- Route a generic ``musubi_cache_latents(architecture="wan", ...)`` call to
  ``wan_cache_latents.py``.
- Validate that the caller provided the right encoder args for the chosen
  architecture (e.g. ``--t5`` + optional ``--clip`` for Wan2.1; only
  ``--t5`` for Wan2.2; ``--text_encoder`` for FLUX.2 and Z-Image).
- Expose capability discovery via ``musubi_list_architectures``.

Source of truth for arg names: ``docs/cli_help.txt`` (output of ``--help``
on every Musubi script) and the architecture doc pages under
``musubi-tuner/docs/``. Flags here were cross-referenced against those
sources; if Musubi upstream renames a flag, update this file and re-dump.

Three architectures are **live** (fully wired, tested in the MCP server):
``wan``, ``flux_2``, ``zimage``. The rest are **placeholders** — registered
so ``musubi_list_architectures`` reports them, but tool calls that target
them return a "not yet implemented" error. Activating a placeholder is a
matter of filling in its flag specifics and flipping ``live=True``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .constants import ArchType


# ---------------------------------------------------------------------------
# TextEncoderSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextEncoderSpec:
    """One text-encoder input on a Musubi script.

    ``flag`` is the CLI flag name (e.g. ``--t5``). ``role`` is a
    human-readable label used in tool descriptions and error messages.
    ``required`` distinguishes mandatory encoders (Wan's ``--t5``) from
    conditionally-required ones (Wan2.1's ``--clip`` for I2V tasks).
    """

    flag: str
    role: str
    required: bool = True


# ---------------------------------------------------------------------------
# ArchitectureConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchitectureConfig:
    """Full spec for one architecture family.

    Scripts that don't exist for a given architecture are reported as
    ``None`` on the corresponding ``*_script`` field — the tool layer
    refuses to route to them. This handles asymmetries like FLUX.2 having
    no full fine-tune script, or HunyuanVideo reusing the generic
    ``cache_latents.py`` at the repo root.
    """

    name: str
    display_name: str
    type: ArchType
    live: bool = False

    # Script names (relative to MUSUBI_TUNER_DIR). ``None`` means the script
    # doesn't exist for this architecture.
    cache_latents_script: Optional[str] = None
    cache_text_encoder_script: Optional[str] = None
    train_network_script: Optional[str] = None
    train_full_script: Optional[str] = None  # full fine-tune (most archs lack this)
    generate_script: Optional[str] = None

    # Required model arg flags.
    vae_arg: str = "--vae"
    dit_arg: str = "--dit"
    text_encoders: tuple[TextEncoderSpec, ...] = field(default_factory=tuple)

    # Network module for LoRA/LoHa/LoKr — passed as ``--network_module``.
    network_module: Optional[str] = None

    # Enumerations, if the architecture has a discriminator flag.
    tasks: Optional[tuple[str, ...]] = None            # ``--task <value>``
    model_versions: Optional[tuple[str, ...]] = None   # ``--model_version <value>``

    # Free-form notes shown to the agent via ``musubi_list_architectures``.
    notes: str = ""

    # ------------------------------------------------------------------
    # Capability helpers
    # ------------------------------------------------------------------

    def has_cache_latents(self) -> bool:
        return self.cache_latents_script is not None

    def has_cache_text_encoder(self) -> bool:
        return self.cache_text_encoder_script is not None

    def has_train_network(self) -> bool:
        return self.train_network_script is not None

    def has_train_full(self) -> bool:
        return self.train_full_script is not None

    def has_generate(self) -> bool:
        return self.generate_script is not None

    def required_encoder_flags(self) -> tuple[str, ...]:
        return tuple(t.flag for t in self.text_encoders if t.required)

    def optional_encoder_flags(self) -> tuple[str, ...]:
        return tuple(t.flag for t in self.text_encoders if not t.required)

    def to_public_dict(self) -> dict:
        """Shape suitable for ``musubi_list_architectures`` JSON output."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "type": self.type,
            "live": self.live,
            "capabilities": {
                "cache_latents": self.has_cache_latents(),
                "cache_text_encoder": self.has_cache_text_encoder(),
                "train_network": self.has_train_network(),
                "train_full": self.has_train_full(),
                "generate": self.has_generate(),
            },
            "model_args": {
                "vae": self.vae_arg,
                "dit": self.dit_arg,
                "text_encoders": [
                    {"flag": t.flag, "role": t.role, "required": t.required}
                    for t in self.text_encoders
                ],
            },
            "network_module": self.network_module,
            "tasks": list(self.tasks) if self.tasks else None,
            "model_versions": list(self.model_versions) if self.model_versions else None,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Registry — populated from docs/cli_help.txt + Musubi's architecture docs
# ---------------------------------------------------------------------------

# CLAUDE-NOTE: Wan2.1 I2V historically needs --clip; Wan2.2 I2V does NOT
# (documented in musubi-tuner/docs/wan.md). We mark --clip optional here
# and let the plan_training_run prompt sort out which task needs it.
_WAN = ArchitectureConfig(
    name="wan",
    display_name="Wan2.1 / Wan2.2",
    type="video",
    live=True,
    cache_latents_script="wan_cache_latents.py",
    cache_text_encoder_script="wan_cache_text_encoder_outputs.py",
    train_network_script="wan_train_network.py",
    train_full_script=None,
    generate_script="wan_generate_video.py",
    vae_arg="--vae",
    dit_arg="--dit",
    text_encoders=(
        TextEncoderSpec(flag="--t5", role="T5 UMT5-XXL", required=True),
        TextEncoderSpec(flag="--clip", role="CLIP (Wan2.1 I2V only)", required=False),
    ),
    network_module="networks.lora_wan",
    tasks=(
        # Wan2.1 official
        "t2v-1.3B", "t2v-14B", "i2v-14B", "t2i-14B",
        # Wan2.1 Fun Control
        "t2v-1.3B-FC", "t2v-14B-FC", "i2v-14B-FC",
        # Wan2.2
        "t2v-A14B", "i2v-A14B",
    ),
    model_versions=None,
    notes=(
        "Wan2.2 dual-model training: pass --dit (low-noise) AND --dit_high_noise. "
        "Default --timestep_boundary: 0.9 for I2V, 0.875 for T2V. "
        "Wan2.2 I2V does NOT use --clip. VAE is shared between Wan2.1 and 2.2."
    ),
)

_FLUX_2 = ArchitectureConfig(
    name="flux_2",
    display_name="FLUX.2 (dev / klein)",
    type="image",
    live=True,
    cache_latents_script="flux_2_cache_latents.py",
    cache_text_encoder_script="flux_2_cache_text_encoder_outputs.py",
    train_network_script="flux_2_train_network.py",
    train_full_script=None,
    generate_script="flux_2_generate_image.py",
    vae_arg="--vae",  # CLAUDE-NOTE: FLUX.2 uses --vae despite the model being called "AE"; confirmed in docs/flux_2.md.
    dit_arg="--dit",
    text_encoders=(
        TextEncoderSpec(flag="--text_encoder", role="Mistral3 (dev) / Qwen3 (klein)", required=True),
    ),
    network_module="networks.lora_flux_2",
    tasks=None,
    model_versions=("dev", "klein-4b", "klein-base-4b", "klein-9b", "klein-base-9b"),
    notes=(
        "--model_version is required. Recommended for training: --timestep_sampling flux2_shift, "
        "--weighting_scheme none, --fp8_base --fp8_scaled. Klein 9B trains in ~9.6GB VRAM with "
        "fp8 + torch.compile. --fp8_text_encoder NOT available for dev (Mistral3)."
    ),
)

_ZIMAGE = ArchitectureConfig(
    name="zimage",
    display_name="Z-Image (Base / Turbo)",
    type="image",
    live=True,
    cache_latents_script="zimage_cache_latents.py",
    cache_text_encoder_script="zimage_cache_text_encoder_outputs.py",
    train_network_script="zimage_train_network.py",
    train_full_script="zimage_train.py",  # Z-Image supports full fine-tune
    generate_script="zimage_generate_image.py",
    vae_arg="--vae",
    dit_arg="--dit",
    text_encoders=(
        TextEncoderSpec(flag="--text_encoder", role="Qwen3", required=True),
    ),
    network_module="networks.lora_zimage",
    tasks=None,
    model_versions=None,
    notes=(
        "Z-Image supports full fine-tuning via zimage_train.py as well as LoRA. "
        "Text encoder fp8: use --fp8_llm (not --fp8_text_encoder). "
        "For full fine-tuning with blocks_to_swap + certain optimizers, "
        "pass --block_swap_optimizer_patch_params."
    ),
)

# ---------------------------------------------------------------------------
# Placeholder architectures — registered for discovery but not wired up yet.
# Filling these in is a matter of verifying their flags against cli_help.txt
# and flipping `live=True`.
# ---------------------------------------------------------------------------

_HV = ArchitectureConfig(
    name="hv",
    display_name="HunyuanVideo",
    type="video",
    live=False,
    # CLAUDE-NOTE: HunyuanVideo historically uses the generic cache_latents.py
    # / cache_text_encoder_outputs.py at repo root (no hv_ prefix). Kept as
    # the script names so the placeholder carries correct info for the audit.
    cache_latents_script="cache_latents.py",
    cache_text_encoder_script="cache_text_encoder_outputs.py",
    train_network_script="hv_train_network.py",
    train_full_script="hv_train.py",
    generate_script="hv_generate_video.py",
    notes="Placeholder — arg flags not yet audited against cli_help.txt.",
)

_HV_1_5 = ArchitectureConfig(
    name="hv_1_5",
    display_name="HunyuanVideo 1.5",
    type="video",
    live=False,
    cache_latents_script="hv_1_5_cache_latents.py",
    cache_text_encoder_script="hv_1_5_cache_text_encoder_outputs.py",
    train_network_script="hv_1_5_train_network.py",
    generate_script="hv_1_5_generate_video.py",
    notes="Placeholder — arg flags not yet audited against cli_help.txt.",
)

_FPACK = ArchitectureConfig(
    name="fpack",
    display_name="FramePack",
    type="video",
    live=False,
    cache_latents_script="fpack_cache_latents.py",
    cache_text_encoder_script="fpack_cache_text_encoder_outputs.py",
    train_network_script="fpack_train_network.py",
    generate_script="fpack_generate_video.py",
    notes="Placeholder — arg flags not yet audited against cli_help.txt.",
)

_FLUX_KONTEXT = ArchitectureConfig(
    name="flux_kontext",
    display_name="FLUX.1 Kontext",
    type="image",
    live=False,
    cache_latents_script="flux_kontext_cache_latents.py",
    cache_text_encoder_script="flux_kontext_cache_text_encoder_outputs.py",
    train_network_script="flux_kontext_train_network.py",
    generate_script="flux_kontext_generate_image.py",
    notes="Placeholder — arg flags not yet audited against cli_help.txt.",
)

_QWEN_IMAGE = ArchitectureConfig(
    name="qwen_image",
    display_name="Qwen-Image / -Edit / -Layered",
    type="image",
    live=False,
    cache_latents_script="qwen_image_cache_latents.py",
    cache_text_encoder_script="qwen_image_cache_text_encoder_outputs.py",
    train_network_script="qwen_image_train_network.py",
    train_full_script="qwen_image_train.py",
    generate_script="qwen_image_generate_image.py",
    notes="Placeholder — arg flags not yet audited against cli_help.txt.",
)

_KANDINSKY5 = ArchitectureConfig(
    name="kandinsky5",
    display_name="Kandinsky 5",
    type="video",
    live=False,
    cache_latents_script="kandinsky5_cache_latents.py",
    cache_text_encoder_script="kandinsky5_cache_text_encoder_outputs.py",
    train_network_script="kandinsky5_train_network.py",
    generate_script="kandinsky5_generate_video.py",
    notes="Placeholder — arg flags not yet audited against cli_help.txt.",
)


_REGISTRY: dict[str, ArchitectureConfig] = {
    arch.name: arch
    for arch in (
        _WAN,
        _FLUX_2,
        _ZIMAGE,
        _HV,
        _HV_1_5,
        _FPACK,
        _FLUX_KONTEXT,
        _QWEN_IMAGE,
        _KANDINSKY5,
    )
}


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def all_architectures() -> list[ArchitectureConfig]:
    """Return every registered architecture in stable insertion order."""
    return list(_REGISTRY.values())


def get_architecture(name: str) -> Optional[ArchitectureConfig]:
    """Look up an architecture by canonical name (e.g. ``"wan"``)."""
    return _REGISTRY.get(name)


def live_architectures() -> list[ArchitectureConfig]:
    """Architectures that are fully wired up and testable."""
    return [a for a in _REGISTRY.values() if a.live]


def architecture_names() -> list[str]:
    """Canonical names, used for Literal typing in tool signatures."""
    return list(_REGISTRY.keys())
