"""
Named (BackboneConfig, TrainConfig-overrides) presets spanning "the tiny
diagnostic scale every FAST_DEV_RUN in this repo already uses" up through
"large enough that you'll want gradient checkpointing and accumulation on a
free-tier GPU". These are starting points, not guarantees.

Param counts below are computed directly from this repo's own
`SequenceBackbone.num_params()` (see the docstring on each preset) --
real numbers from this codebase, not estimates. GPU memory/throughput
guidance is NOT measured the same way: this repo was built in a
CPU-only, single-core, ~4GB-RAM sandbox with no GPU access at all, so
every memory/fit claim below is a rule-of-thumb estimate from standard
transformer/SSM memory scaling, explicitly flagged as such. Profile on
your actual Kaggle/Colab GPU before trusting these past "small" -- see
docs/scaling_up.md for how.

One number IS a directly-reproduced fact rather than an estimate: this
sandbox's own build measurably OOM-killed a plain forward+backward at
d_model=64/n_layers=3/d_state=32/seq_len=128/batch=32 (~4GB RAM, no GPU),
and gradient checkpointing measurably fixed it (see
tests/test_grad_checkpointing.py and docs/scaling_up.md) -- so "checkpointing
matters past tiny scale" is demonstrated, not just claimed, even though the
exact GPU-memory numbers below are estimates.
"""
from __future__ import annotations

from dataclasses import dataclass

from ssm_lab.models.backbone import BackboneConfig


@dataclass
class ScalePreset:
    name: str
    backbone_kwargs: dict
    train_overrides: dict
    approx_params_at_vocab_256: str  # computed once via num_params(), see module docstring
    guidance: str


PRESETS: dict[str, ScalePreset] = {
    "tiny": ScalePreset(
        name="tiny",
        backbone_kwargs=dict(d_model=64, n_layers=2, mixer_kwargs=dict(d_state=16), use_gradient_checkpointing=False),
        train_overrides=dict(grad_accum_steps=1),
        approx_params_at_vocab_256="~112K",
        guidance=(
            "The diagnostic/FAST_DEV_RUN scale already used throughout this repo's "
            "notebooks and synthetic benchmarks. Runs on CPU in seconds. No "
            "checkpointing or accumulation needed."
        ),
    ),
    "small": ScalePreset(
        name="small",
        backbone_kwargs=dict(d_model=256, n_layers=6, mixer_kwargs=dict(d_state=64), use_gradient_checkpointing=False),
        train_overrides=dict(grad_accum_steps=1),
        approx_params_at_vocab_256="~4.6M",
        guidance=(
            "Comfortable on a free Kaggle/Colab GPU (T4 or P100, 16GB) at seq_len "
            "up to roughly 512-1024 and batch sizes in the tens, with room to "
            "spare. Should not need gradient checkpointing or accumulation -- "
            "if you hit OOM here, seq_len is the more likely culprit than params."
        ),
    ),
    "medium": ScalePreset(
        name="medium",
        backbone_kwargs=dict(d_model=512, n_layers=12, mixer_kwargs=dict(d_state=64), use_gradient_checkpointing=True),
        train_overrides=dict(grad_accum_steps=2),
        approx_params_at_vocab_256="~34M",
        guidance=(
            "Likely needs gradient checkpointing on a free 16GB T4/P100 once "
            "seq_len passes a few hundred tokens; grad_accum_steps=2-4 recovers "
            "effective batch size lost to whatever micro-batch fits. Comfortable "
            "without either on a Colab Pro A100/L4."
        ),
    ),
    "large": ScalePreset(
        name="large",
        backbone_kwargs=dict(d_model=768, n_layers=16, mixer_kwargs=dict(d_state=96), use_gradient_checkpointing=True),
        train_overrides=dict(grad_accum_steps=4),
        approx_params_at_vocab_256="~102M",
        guidance=(
            "Expect to need both gradient checkpointing and accumulation on any "
            "free-tier 16GB GPU, plus checkpoint/resume across sessions for "
            "anything beyond a short run (Kaggle's 12h session cap in "
            "particular). This is the tier where profiling your actual "
            "memory/step-time on the real hardware before committing to a long "
            "run stops being optional."
        ),
    ),
}


def get_preset(name: str, vocab_size: int, **backbone_overrides) -> tuple[BackboneConfig, dict]:
    """Returns (BackboneConfig, train_config_overrides_dict) for a named
    preset. `vocab_size` isn't baked into the presets above since it's a
    property of your tokenizer/corpus, not the model scale. Any keyword
    override replaces the preset's default for that BackboneConfig field
    (e.g. `get_preset("medium", vocab_size=256, use_gradient_checkpointing=False)`
    to try medium-scale without checkpointing first).

    Usage:
        cfg, train_overrides = get_preset("medium", vocab_size=data.vocab_size)
        model = SequenceBackbone(S4DLayer, cfg)
        tcfg = TrainConfig(max_steps=5000, **train_overrides)
    """
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {sorted(PRESETS)}")
    preset = PRESETS[name]
    kwargs = dict(preset.backbone_kwargs)
    kwargs.update(backbone_overrides)
    cfg = BackboneConfig(vocab_size=vocab_size, **kwargs)
    return cfg, dict(preset.train_overrides)
