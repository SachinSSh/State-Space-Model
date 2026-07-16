"""
Single source of truth for "what models exist in this repo right now".

Every notebook selects a mixer by name through `get_mixer`, never by
importing a layer class directly -- that's what lets `20_benchmark_suite.ipynb`
loop over every implemented model without a giant if/elif ladder, and it's
the exact list docs/how_to_add_a_model.md tells you to append to.
"""
from __future__ import annotations

from ssm_lab.layers.s4d import S4DLayer
from ssm_lab.layers.s5 import S5Layer

MIXER_REGISTRY: dict[str, dict] = {
    "s4d": dict(
        cls=S4DLayer,
        default_kwargs=dict(d_state=64, init="lin"),
        paper="Gu, Gupta, Goel & Re, 2022 - On the Parameterization and Initialization of Diagonal SSMs",
        year=2022,
        family="foundation",
    ),
    "s5": dict(
        cls=S5Layer,
        default_kwargs=dict(d_state=128, init="lin"),
        paper="Smith, Warrington & Linderman, 2023 - Simplified State Space Layers for Sequence Modeling",
        year=2023,
        family="foundation",
    ),
    # -- appended here as each subsequent model in the 14-model lineage lands:
    # "mamba":         Mamba / S6 (selective scan)                       -- mamba lineage
    # "mamba2":        Mamba-2 / SSD                                       -- mamba lineage
    # "mamba3":        Mamba-3 (complex state, trapezoidal disc.)         -- mamba lineage
    # "retnet":        RetNet (fixed multi-scale decay)                   -- linear-attn family
    # "gla":           Gated Linear Attention                              -- linear-attn family
    # "deltanet":      DeltaNet / Gated DeltaNet                           -- linear-attn family
    # "rwkv7":         RWKV-7 "Goose"                                      -- linear-attn family
    # "xlstm":         xLSTM (mLSTM block)                                 -- linear-attn family
    # "jamba_block":   Jamba-style Mamba2 + attention + MoE hybrid         -- hybrid
    # "griffin":       Griffin (RG-LRU + local attention)                  -- hybrid
    # "vision_mamba":  bidirectional-scan Mamba for images                 -- applied
    # "titans_mini":   Titans-style surprise-gated test-time memory        -- memory frontier
}


def get_mixer(name: str) -> dict:
    if name not in MIXER_REGISTRY:
        raise KeyError(f"Unknown mixer '{name}'. Available: {sorted(MIXER_REGISTRY)}")
    return MIXER_REGISTRY[name]


def list_mixers() -> list[str]:
    return sorted(MIXER_REGISTRY)
