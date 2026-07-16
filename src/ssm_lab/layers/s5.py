"""
S5 -- one shared MIMO (multi-input, multi-output) diagonal state space,
mixing across all channels directly inside the recurrence.

Reference: Smith, Warrington & Linderman, "Simplified State Space Layers
for Sequence Modeling" (ICLR 2023), building on the same HiPPO theory
(Gu et al., NeurIPS 2020) as S4/S4D.

What changed relative to Model 1 (S4D): S4D gives every one of its
`d_model` channels an *independent* SISO (single-input-single-output)
SSM -- each channel has its own N-dimensional state, and channels never
interact until the surrounding FFN mixes them. S5 replaces that with
*one* shared, `d_state`-dimensional MIMO SSM: a single recurrent state
that every input channel writes into (via a `(d_state, d_model)` matrix
B) and every output channel reads from (via a `(d_model, d_state)`
matrix C). Concretely, S4D's `B̄ * u_t` is an elementwise product (no
mixing); S5's `B̄ @ u_t` is a real matrix-vector product (every state
component is a learned weighted combination of *all* input channels).
Same for C on the output side. Cross-channel mixing happens inside the
recurrence itself now, not only around it.

That's also the only structural difference the rest of this file's math
needs: A is still diagonal, still discretized via zero-order hold, still
evaluated with the exact same `ssm_lab.utils.scan.parallel_scan_log`
primitive S4D uses -- once `B̄ @ u_t` is precomputed for every timestep
(a single batched matmul, done once, before the scan), what's left is the
identical elementwise affine recurrence `h_t = Ā*h_{t-1} + b_t` with
`b_t` now equal to that precomputed `B̄ @ u_t` instead of S4D's
elementwise `B̄ * u_t`. No new scan primitive needed for this model --
exactly the reason S5 was next on the roadmap after S4D.

Honest scope notes:
- A is initialized from the same diagonal HiPPO-LegS approximation as
  S4D (`ssm_lab.utils.hippo.hippo_diag_spectrum`), unrepeated across
  channels since this state is shared, not per-channel. The original S5
  paper derives its init from an eigendecomposition of a normal
  approximation to HiPPO-LegS, which is closely related to but not
  identically derived as the S4D-Lin/Inv approximation reused here --
  both are diagonal approximations to the same underlying spectrum, and
  using one reference implementation for both models in this repo was
  judged more valuable than reproducing S5's exact original derivation.
- Same full-complex-state simplification as S4D (no conjugate-pair
  halving) -- see `docs/01_s4.md` for why, unchanged here.
- B and C are both randomly initialized (scaled by fan-in), unlike
  S4D's B (fixed at all-ones, since S4D's *scalar* per-channel B is
  redundant with C). That shortcut doesn't apply here: S5's B is a real
  `(d_state, d_model)` routing matrix, and a fixed B would force every
  state to read an identical fixed combination of input channels --
  removing exactly the cross-channel mixing capacity that's the point
  of this model. Learned, randomly-initialized B and C, both scaled by
  their fan-in, are the standard choice instead.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ssm_lab.layers.base import SequenceMixer
from ssm_lab.utils.hippo import hippo_diag_spectrum
from ssm_lab.utils.scan import parallel_scan_log


class S5Layer(SequenceMixer):
    """
    One S5 layer: a single `d_state`-dimensional complex diagonal SSM,
    shared across all `d_model` channels, followed by a GELU-gated output
    projection (same gating pattern as S4D, for consistency across this
    repo's models -- and the same pattern the original S5 paper's own
    reference block uses).

    Args:
        d_model:  number of channels (= residual stream width).
        d_state:  the *shared* complex state dimension (called P in the
                  paper). Not per-channel the way S4D's d_state is --
                  see the module docstring. Scan cost is
                  O(L log L * d_state), independent of d_model (unlike
                  S4D, where it's O(L log L * d_model * d_state)).
        dt_min/dt_max: per-state-dimension step size Delta, log-uniform
                  at init, then learned -- same role as in S4D.
        init:     'lin' or 'inv', see `ssm_lab.utils.hippo`.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
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

        # A: diagonal, complex, shared across channels -- (d_state,), not
        # (d_model, d_state) like S4D. Same -exp(.) reparameterization
        # trick guaranteeing Re(A) < 0 for any real log_A_real.
        log_A_real = torch.log(0.5 * torch.ones(d_state))
        A_imag = hippo_diag_spectrum(d_state, mode=init)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

        # B, C: both learned and randomly initialized (see module
        # docstring for why S4D's "fix B at ones" shortcut doesn't
        # transfer here), scaled by 1/sqrt(fan_in) each.
        self.B_real = nn.Parameter(torch.randn(d_state, d_model) / math.sqrt(d_model))
        self.B_imag = nn.Parameter(torch.randn(d_state, d_model) / math.sqrt(d_model))
        self.C_real = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.C_imag = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))

        log_dt = torch.rand(d_state) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        self.D = nn.Parameter(torch.ones(d_model))  # feedthrough / skip term, per-channel like S4D

        self.out_proj = nn.Linear(d_model, 2 * d_model)
        self.act = nn.GELU()

    def _discretize(self):
        """Zero-order-hold discretization. A, dt live in (d_state,); B
        lives in (d_state, d_model), so the ZOH ratio broadcasts over the
        channel dimension rather than matching B's shape elementwise the
        way it does in S4D."""
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (d_state,)
        dt = torch.exp(self.log_dt)  # (d_state,)
        A_bar = torch.exp(dt * A)  # (d_state,)
        B = self.B_real + 1j * self.B_imag  # (d_state, d_model)

        safe = A.abs() > 1e-6  # (d_state,)
        ratio_safe = (A_bar - 1) / torch.where(safe, A, torch.ones_like(A))
        ratio = torch.where(safe, ratio_safe, dt.to(ratio_safe.dtype))  # (d_state,)
        B_bar = ratio.unsqueeze(-1) * B  # (d_state, 1) * (d_state, d_model) -> (d_state, d_model)
        return A_bar, B_bar

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: (batch, seq_len, d_model), real
        batch, seq_len, d_model = x.shape
        A_bar, B_bar = self._discretize()  # (d_state,), (d_state, d_model)

        Bu = torch.einsum("ph,blh->blp", B_bar, x.to(B_bar.dtype))  # (batch, seq_len, d_state)
        a = A_bar.view(1, 1, self.d_state).expand(batch, seq_len, self.d_state)

        h = parallel_scan_log(a, Bu)  # (batch, seq_len, d_state), complex -- same scan as S4D

        C = self.C_real + 1j * self.C_imag  # (d_model, d_state)
        y = torch.einsum("hp,blp->blh", C, h).real + self.D * x  # (batch, seq_len, d_model)

        y_proj, gate = self.out_proj(y).chunk(2, dim=-1)
        return y_proj * self.act(gate)

    # ---- O(1) autoregressive decoding -------------------------------------
    def init_state(self, batch_size: int, device=None, dtype=None):
        return torch.zeros(batch_size, self.d_state, dtype=torch.cfloat, device=device)

    def step(self, x_t: torch.Tensor, state: torch.Tensor):
        # x_t: (batch, d_model) real; state: (batch, d_state) complex
        A_bar, B_bar = self._discretize()
        Bu_t = torch.einsum("ph,bh->bp", B_bar, x_t.to(state.dtype))  # (batch, d_state)
        new_state = A_bar.unsqueeze(0) * state + Bu_t
        C = self.C_real + 1j * self.C_imag
        y = torch.einsum("hp,bp->bh", C, new_state).real + self.D * x_t
        y_proj, gate = self.out_proj(y).chunk(2, dim=-1)
        return y_proj * self.act(gate), new_state
