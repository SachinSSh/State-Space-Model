"""
Copying task.

Reference: Jelassi, Brandfonbrener, Kakade & Malach, "Repeat After Me:
Transformers are Better than State Space Models at Copying" (2024). Their
theoretical result: a model whose state is a *fixed size* independent of
sequence length (any SSM, any linear-attention/RNN variant) cannot copy an
arbitrary string once the string is longer than what that fixed state can
represent, while attention -- whose "state" is the entire growing KV cache
-- can. This benchmark makes that gap directly observable: fix the model's
state size, then sweep the length of the string to be copied and watch
accuracy fall off a cliff at the length that saturates that state.

    [random tokens, length k] [DELIM] [same random tokens, length k again]

The model reads the first copy causally, and once it reaches DELIM must
reproduce the sequence again token-for-token; loss_mask is set on the
positions of the second copy (predicting the *next* token in that copy),
never on the first presentation (unpredictable) or the delimiter itself.
"""
from __future__ import annotations

import torch

DELIM_ID = 0  # reserved; never sampled as part of the random content


def generate_copy_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    copy_len: int,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seq_len < 2 * copy_len + 1:
        raise ValueError(f"seq_len={seq_len} too short to copy {copy_len} tokens twice plus a delimiter")

    to_copy = torch.randint(1, vocab_size, (batch_size, copy_len), generator=generator)
    x = torch.full((batch_size, seq_len), DELIM_ID, dtype=torch.long)
    x[:, :copy_len] = to_copy
    # x[:, copy_len] stays DELIM_ID
    x[:, copy_len + 1 : 2 * copy_len + 1] = to_copy
    if seq_len > 2 * copy_len + 1:
        x[:, 2 * copy_len + 1 :] = torch.randint(1, vocab_size, (batch_size, seq_len - (2 * copy_len + 1)), generator=generator)

    targets = torch.roll(x, shifts=-1, dims=1)
    targets[:, -1] = -100

    loss_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    # position (copy_len + i) predicts x[copy_len + i + 1] == to_copy[:, i], for i in [0, copy_len-1)
    loss_mask[:, copy_len : 2 * copy_len - 1] = True

    return x.to(device), targets.to(device), loss_mask.to(device)


def make_copy_loader(
    batch_size: int, seq_len: int, vocab_size: int, copy_len: int,
    device: str = "cpu", seed: int | None = None,
):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def _batch_fn():
        return generate_copy_batch(batch_size, seq_len, vocab_size, copy_len, device, generator)

    return _batch_fn
