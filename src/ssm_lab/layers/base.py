"""
Common interface every sequence-mixing layer ("mixer") in this repo implements.

The point of this repo is a *fair* comparison: 14+ architectures spanning
S4 (2022) through Mamba-3 (2026) are each a drop-in replacement for `mixer`
inside the exact same residual backbone (`ssm_lab.models.backbone.SequenceBackbone`),
trained by the exact same `ssm_lab.training.trainer.Trainer`, on the exact
same benchmarks. Only what happens inside `forward` differs between models.
That is the entire methodology -- so this interface is the most
important contract in the codebase.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SequenceMixer(nn.Module):
    """
    Contract:
      - forward(x) takes  x: (batch, seq_len, d_model)
                  returns y: (batch, seq_len, d_model)   -- same shape in and out
      - causal by default: y_t must not depend on x_{t+1}, x_{t+2}, ...
        (Vision Mamba is the one deliberate non-causal exception in this repo,
        and it says so explicitly in its own module docstring)
      - __init__ always accepts `d_model` and `layer_idx` as keyword args,
        even if a given mixer ignores `layer_idx` -- the backbone passes it
        to every layer uniformly so architectures that DO use it
        (e.g. Jamba-style hybrids, which only put attention on specific
        layer indices) don't need special-cased wiring.

    Optional, but implemented wherever the underlying math supports it:
      - step(x_t, state) -> (y_t, new_state):  O(1)-memory, O(1)-compute
        autoregressive decoding, one token at a time. This is the whole
        practical selling point of SSMs/linear attention over softmax
        attention (no growing KV cache) -- so a mixer that *can't* implement
        this cheaply (plain softmax attention) says so loudly rather than
        silently degrading to an O(L) cache, because that gap is itself
        part of what this repo measures.
      - init_state(batch_size, device, dtype): returns the zero/initial
        recurrent state in whatever shape this mixer's `step` expects.
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def step(self, x_t: torch.Tensor, state):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement O(1) single-step "
            f"decoding. If that's expected for this architecture (e.g. plain "
            f"softmax attention needs a growing KV cache instead), that's a "
            f"real, measured limitation -- not a TODO."
        )

    def init_state(self, batch_size: int, device=None, dtype=None):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement init_state / step."
        )
