"""
Shared linear-recurrence scan.

Almost every model in this repo is, at its core, a diagonal affine recurrence

    h_t = a_t * h_{t-1} + b_t          (elementwise; h_0 = 0 unless h0 given)

with different definitions of a_t and b_t:

  - S4 / S4D:      a_t = Abar          (constant across t -- NOT input-dependent)
                    b_t = Bbar * u_t
  - Mamba (S6):    a_t = Abar(u_t)     (input-dependent -- "selective")
                    b_t = Bbar(u_t) * u_t
  - GLA / RWKV-7:  a_t = sigmoid(...)  (a learned data-dependent forget gate)
                    b_t = k_t (x) v_t  (an outer-product / matrix-valued state,
                                        handled by reshaping the trailing dims)

`sequential_scan` is the O(L) ground truth used only for unit tests and for
tiny debugging runs. `parallel_scan_log` is the O(log L)-sequential-step
Hillis-Steele associative scan that every notebook in this repo actually
trains with -- it is fully vectorized across batch/channels/state and only
loops in Python over log2(L) doubling rounds, so it is bandwidth-bound on a
GPU rather than launch-overhead-bound.

Both functions work for real- or complex-dtype (a, b), which is what lets
S4D (complex diagonal state) and Mamba (real diagonal state) share one
implementation.
"""
from __future__ import annotations

import torch


def sequential_scan(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
    """
    Ground-truth O(L) recurrence, looped in Python. Only for tests / tiny debug
    runs -- do not use this inside a training loop, it will be extremely slow
    on a GPU because of the L sequential kernel launches.

    a, b: (batch, seq_len, *dims), same shape, real or complex.
    h0:   (batch, *dims) or None (treated as zeros).
    returns h: (batch, seq_len, *dims), h[:, t] = h_t.
    """
    batch, seq_len = a.shape[0], a.shape[1]
    if h0 is None:
        h = torch.zeros(batch, *a.shape[2:], dtype=a.dtype, device=a.device)
    else:
        h = h0.to(a.dtype)
    out = []
    for t in range(seq_len):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


def parallel_scan_log(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
    """
    Hillis-Steele parallel scan. Computes the same recurrence as
    `sequential_scan` in ceil(log2(seq_len)) sequential rounds instead of
    seq_len sequential steps.

    The trick: at each round we combine every position t with the position
    d steps behind it (d doubling each round: 1, 2, 4, 8, ...) using the
    *composition* of two affine maps, not their sum:

        combine(new=(a2,b2), old=(a1,b1)) = (a2*a1, a2*b1 + b2)

    which represents "apply the affine map (a1,b1), then apply (a2,b2)".
    Positions with no partner d steps behind them (t < d) are padded with the
    identity element of this operator, (a=1, b=0) -- composing with the
    identity is a no-op, so padding this way means we can update *every*
    position uniformly, with no masking / branching.

    a, b: (batch, seq_len, *dims), same shape, real or complex.
    h0:   (batch, *dims) or None (treated as zeros).
    returns h: (batch, seq_len, *dims), h[:, t] = h_t.
    """
    a = a.clone()
    b = b.clone()
    seq_len = a.shape[1]

    d = 1
    while d < seq_len:
        a_id = torch.ones(a.shape[0], d, *a.shape[2:], dtype=a.dtype, device=a.device)
        b_id = torch.zeros(b.shape[0], d, *b.shape[2:], dtype=b.dtype, device=b.device)
        a_shifted = torch.cat([a_id, a[:, :-d]], dim=1)
        b_shifted = torch.cat([b_id, b[:, :-d]], dim=1)

        b = a * b_shifted + b
        a = a * a_shifted
        d *= 2

    h = b
    if h0 is not None:
        h = h + a * h0.unsqueeze(1)
    return h
