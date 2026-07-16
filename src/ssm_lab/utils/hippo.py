"""
Diagonal approximation to the HiPPO-LegS eigenvalue spectrum.

Originally implemented inline in `ssm_lab.layers.s4d` (S4D was the first
model in this repo to need it); pulled out here once S5 needed the exact
same spectrum for a differently-shaped state. Full HiPPO-LegS (Gu et al.,
NeurIPS 2020) has a Normal-Plus-Low-Rank structure; both S4D and S5
approximate it with a purely diagonal matrix instead, at two different
levels of fidelity to the true asymptotic spectrum -- "lin" (S4D-Lin) and
"inv" (S4D-Inv), both from Gu, Gupta, Goel & Ré (S4D, NeurIPS 2022).

This function returns only the state-dimension spectrum, `(d_state,)` --
what a model does with it differs by architecture:
  - S4D repeats it across `d_model` channels, since S4D gives every
    channel its own independent state (see `S4DLayer.__init__`).
  - S5 uses it exactly as returned, unrepeated, since S5's state is
    shared across all channels rather than per-channel (see
    `S5Layer.__init__` and `docs/02_s5.md`).
"""
from __future__ import annotations

import math

import torch


def hippo_diag_spectrum(d_state: int, mode: str = "lin") -> torch.Tensor:
    """Returns the (d_state,) imaginary-part spectrum used to initialize a
    diagonal approximation of A = -1/2 + i * spectrum.

    mode="lin" (S4D-Lin): imaginary parts grow linearly in the state index,
    matching the leading-order asymptotic behaviour of the true HiPPO-LegS
    eigenvalues -- simple, and what most from-scratch diagonal-SSM
    implementations use by default.

    mode="inv" (S4D-Inv): closer to HiPPO-LegS's true inverse-law falloff;
    tends to help most on longer sequences at the same state size.
    """
    n = torch.arange(d_state, dtype=torch.float32)
    if mode == "lin":
        return math.pi * n
    if mode == "inv":
        return (d_state / math.pi) * (d_state / (2 * n + 1) - 1)
    raise ValueError(f"unknown init mode '{mode}', expected 'lin' or 'inv'")
