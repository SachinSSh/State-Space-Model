"""
State-tracking benchmarks: parity and S5 permutation composition.

Reference: Merrill, Petty & Sabharwal, "The Illusion of State in
State-Space Models" (2024) formalizes why this matters: SSMs/linear
attention with a *diagonal*, decay-only state transition are restricted to
computations in TC^0, the same complexity class as plain attention -- they
provably cannot track state for problems outside it. S5 permutation
composition is NC^1-complete and is the standard separator used to show
this in practice; parity (in TC^0, but empirically still hard for
pure-decay diagonal SSMs at scale) is a gentler sanity check to run first.
This is precisely the gap DeltaNet's delta-rule state update, RWKV-7's
generalized delta rule (proven to recognize all regular languages), and
Mamba-3's complex-valued state-tracking mechanism are each, in their own
way, designed to close -- and precisely what a plain diagonal-decay S4D
layer is expected to fail at in this repo's results table. That expected
failure is a result, not a bug, once it shows up.

Both tasks are dense sequence-labelling (the label at position t is a
function of the *entire prefix* x_1..x_t, produced from the model's
representation at that same position t) rather than next-token-shifted --
loss_mask is all-True; there is no "unpredictable" position to mask out,
every prefix has a well-defined running answer.
"""
from __future__ import annotations

import itertools

import torch

# --- S5 permutation composition -------------------------------------------

_S5_ELEMENTS: list[tuple[int, ...]] = list(itertools.permutations(range(5)))  # 120 elements
_S5_INDEX: dict[tuple[int, ...], int] = {p: i for i, p in enumerate(_S5_ELEMENTS)}
_IDENTITY_IDX: int = _S5_INDEX[tuple(range(5))]
S5_VOCAB_SIZE = len(_S5_ELEMENTS)  # 120


def _compose(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    """(p o q)(x) = p(q(x)) -- apply q first, then p."""
    return tuple(p[q[i]] for i in range(5))


def _build_mult_table() -> torch.Tensor:
    table = torch.zeros(S5_VOCAB_SIZE, S5_VOCAB_SIZE, dtype=torch.long)
    for i, p in enumerate(_S5_ELEMENTS):
        for j, q in enumerate(_S5_ELEMENTS):
            table[i, j] = _S5_INDEX[_compose(p, q)]
    return table


_MULT_TABLE = _build_mult_table()  # (120, 120): table[new, old] = compose(new, old)


def generate_s5_state_tracking_batch(
    batch_size: int, seq_len: int, device: str = "cpu", generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x: a stream of S5 elements (tokens 0..119), sampled i.i.d. uniformly.
    y_t: the group product g_t o g_{t-1} o ... o g_1 (newest applied last),
         i.e. the cumulative running composition, also a token 0..119.
    """
    x = torch.randint(0, S5_VOCAB_SIZE, (batch_size, seq_len), generator=generator)
    targets = torch.empty_like(x)
    state = torch.full((batch_size,), _IDENTITY_IDX, dtype=torch.long)
    for t in range(seq_len):
        state = _MULT_TABLE[x[:, t], state]
        targets[:, t] = state
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return x.to(device), targets.to(device), loss_mask.to(device)


def make_s5_tracking_loader(batch_size: int, seq_len: int, device: str = "cpu", seed: int | None = None):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def _batch_fn():
        return generate_s5_state_tracking_batch(batch_size, seq_len, device, generator)

    return _batch_fn


# --- parity -----------------------------------------------------------------

PARITY_VOCAB_SIZE = 2


def generate_parity_batch(
    batch_size: int, seq_len: int, device: str = "cpu", generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x: i.i.d. random bits. y_t: running parity (cumulative XOR) of x_1..x_t."""
    bits = torch.randint(0, 2, (batch_size, seq_len), generator=generator)
    targets = torch.remainder(torch.cumsum(bits, dim=1), 2)
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return bits.to(device), targets.to(device), loss_mask.to(device)


def make_parity_loader(batch_size: int, seq_len: int, device: str = "cpu", seed: int | None = None):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def _batch_fn():
        return generate_parity_batch(batch_size, seq_len, device, generator)

    return _batch_fn
