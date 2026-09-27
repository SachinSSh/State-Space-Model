"""

Author: SachinSSh

S4D -- diagonal state space layer.

Reference: Gu, Gupta, Goel & Re, "On the Parameterization and Initialization
of Diagonal State Space Models" (NeurIPS 2022), building on Gu, Goel & Re,
"Efficiently Modeling Long Sequences with Structured State Spaces" (S4,
ICLR 2022) and the HiPPO theory of online function approximation
(Gu et al., NeurIPS 2020) that motivates the initialization below.

Honest scope note (read this before assuming a bug): full S4 parameterizes
its state matrix A as Normal-Plus-Low-Rank (NPLR) and evaluates the implied
convolution kernel through a Cauchy-kernel / Woodbury-identity trick,
specifically so a *non-diagonal* A stays subquadratic to evaluate. That
machinery is intricate enough that most from-scratch "build S4" writeups --
this repo included -- implement S4D instead: make A diagonal from the start,
which sidesteps the Cauchy-kernel derivation entirely. We approximate the
HiPPO-LegS eigenvalue spectrum with a diagonal initialization (S4D-Lin or
S4D-Inv, both below) rather than deriving the exact NPLR decomposition, and
we evaluate the recurrence with the parallel associative scan in
`ssm_lab.utils.scan` rather than the FFT convolution path the S4 paper uses.
That scan is not a shortcut specific to this layer -- it's the same primitive
S5, Mamba's selective scan, and (with a data-dependent a_t) the GLA/DeltaNet/
RWKV-7 family in this repo all reduce to. What genuinely sets S4D apart from
Mamba two models later in this same repo is that a_t = exp(dt * A) here is a
learned *constant*, not a function of the input -- S4D cannot "select" what
to remember based on content, only on position. That single difference is
most of the empirical gap between this layer and Mamba on the associative
recall benchmark.

We also keep the full N-dimensional complex state per channel rather than
exploiting conjugate-pair symmetry to track only N/2 and double the real
part (a standard S4/S4D compute optimization). That costs roughly 2x the
compute/memory of a tuned implementation for the same N, in exchange for a
forward pass with no "why is there a factor of 2 here" step. Noted as a
first optimization target in docs/01_s4.md, not hidden.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ssm_lab.layers.base import SequenceMixer
from ssm_lab.utils.hippo import hippo_diag_spectrum
from ssm_lab.utils.scan import parallel_scan_log


def _hippo_diag_init(d_model: int, d_state: int, mode: str = "lin") -> torch.Tensor:
    """S4D gives every channel its own independent state, so the shared
    (d_state,) spectrum from `ssm_lab.utils.hippo` gets repeated identically
    across all d_model channels here. Contrast with S5 (`s5.py`), which uses
    `hippo_diag_spectrum` directly, unrepeated, since its state is shared
    across channels rather than per-channel."""
    return hippo_diag_spectrum(d_state, mode=mode).unsqueeze(0).repeat(d_model, 1)


class S4DLayer(SequenceMixer):
    """
    One S4D layer: d_model independent single-input-single-output (SISO)
    diagonal SSMs, one per channel, each with its own d_state-dimensional
    complex hidden state, followed by a GELU-gated output projection.

    Args:
        d_model:  number of channels (= residual stream width).
        d_state:  per-channel complex state dimension N. 64 is the standard
                  S4/S4D default; this is a quality/compute knob, not a
                  correctness one -- the scan cost is O(L log L * d_model * d_state).
        dt_min/dt_max: per-channel step size Delta is drawn log-uniformly from
                  this range at init and then learned; the range controls what
                  timescales of memory the layer can represent (this is the
                  main S4/S4D hyperparameter that actually matters in practice).
        init:     'lin' or 'inv', see `_hippo_diag_init`.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        init: str = "lin",
        layer_idx: int = 0,
        **_ignored,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.layer_idx = layer_idx

        # A = -exp(log_A_real) + i * A_imag  -- the -exp(.) reparameterization
        # guarantees Re(A) < 0 (a stable, decaying mode) for *any* real value
        # of log_A_real, so training can never accidentally push the system
        # into an exploding regime; this is the standard S4/S4D trick.
        log_A_real = torch.log(0.5 * torch.ones(d_model, d_state))
        A_imag = _hippo_diag_init(d_model, d_state, mode=init)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

        # B initialized to all-ones (standard S4D choice -- with a diagonal A,
        # B's magnitude is redundant with C's, so there's nothing to gain from
        # a fancier init here).
        self.B_real = nn.Parameter(torch.ones(d_model, d_state))
        self.B_imag = nn.Parameter(torch.zeros(d_model, d_state))

        # C is the only randomly-initialized SSM parameter.
        self.C_real = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.C_imag = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))

        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        self.D = nn.Parameter(torch.ones(d_model))  # feedthrough / skip term

        self.out_proj = nn.Linear(d_model, 2 * d_model)
        self.act = nn.GELU()

    def _discretize(self):
        """Zero-order-hold discretization, elementwise since A is diagonal."""
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (d_model, d_state)
        dt = torch.exp(self.log_dt).unsqueeze(-1)  # (d_model, 1)
        dtA = dt * A
        A_bar = torch.exp(dtA)
        B = self.B_real + 1j * self.B_imag
        # ZOH for B: Bbar = A^{-1}(Abar - I) B, guarded against the A ~ 0
        # singularity (never hit with this init, but cheap to guard).
        safe = A.abs() > 1e-6
        B_bar_safe = (A_bar - 1) / torch.where(safe, A, torch.ones_like(A)) * B
        B_bar = torch.where(safe, B_bar_safe, dt * B)
        return A_bar, B_bar

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: (batch, seq_len, d_model), real
        batch, seq_len, d_model = x.shape
        A_bar, B_bar = self._discretize()  # (d_model, d_state) complex each

        u = x.unsqueeze(-1).to(A_bar.dtype)  # (batch, seq_len, d_model, 1)
        a = A_bar.view(1, 1, d_model, self.d_state).expand(batch, seq_len, d_model, self.d_state)
        b = B_bar.view(1, 1, d_model, self.d_state) * u  # (batch, seq_len, d_model, d_state)

        h = parallel_scan_log(a, b)  # (batch, seq_len, d_model, d_state), complex

        C = self.C_real + 1j * self.C_imag
        y = torch.einsum("bldn,dn->bld", h, C).real  # back to real, (batch, seq_len, d_model)
        y = y + self.D * x

        y_proj, gate = self.out_proj(y).chunk(2, dim=-1)
        return y_proj * self.act(gate)

    # ---- O(1) autoregressive decoding -------------------------------------
    def init_state(self, batch_size: int, device=None, dtype=None):
        return torch.zeros(batch_size, self.d_model, self.d_state, dtype=torch.cfloat, device=device)

    def step(self, x_t: torch.Tensor, state: torch.Tensor):
        # x_t: (batch, d_model) real; state: (batch, d_model, d_state) complex
        A_bar, B_bar = self._discretize()
        new_state = A_bar.unsqueeze(0) * state + B_bar.unsqueeze(0) * x_t.unsqueeze(-1).to(state.dtype)
        C = self.C_real + 1j * self.C_imag
        y = torch.einsum("bdn,dn->bd", new_state, C).real + self.D * x_t
        y_proj, gate = self.out_proj(y).chunk(2, dim=-1)
        return y_proj * self.act(gate), new_state
