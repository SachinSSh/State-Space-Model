"""
Induction-head probe.

Reference: Olsson et al., "In-context Learning and Induction Heads"
(Anthropic, 2022); adopted as an architecture probe in the H3, Hyena, and
Mamba papers specifically because it is the minimal template for the
"complete the pattern using what came right after this token, last time it
appeared" circuit that (in Transformers) is implemented by a composition of
a previous-token head and a matching head across two attention layers.

A bigram [A][B] is planted early in an otherwise-random sequence; the
*second* occurrence of A is planted later and must be immediately followed
by B again. loss_mask is set only at that second occurrence of A (predicting
the token right after it); nothing else in the sequence is scored, since
everything except that one bigram is unpredictable filler by construction.
"""
from __future__ import annotations

import torch


def generate_induction_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_triggers: int = 1,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seq_len < 4 * num_triggers + 2:
        raise ValueError(f"seq_len={seq_len} too short for num_triggers={num_triggers}")

    x = torch.randint(1, vocab_size, (batch_size, seq_len), generator=generator)
    loss_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    # Python double loop (batch x num_triggers): num_triggers is small
    # (typically 1-3) so this is cheap; see the note in mqar.py on why the
    # batch loop itself is left unvectorized for a from-scratch reference impl.
    for b in range(batch_size):
        for _ in range(num_triggers):
            a_pos = int(torch.randint(0, seq_len - 3, (1,), generator=generator).item())
            b_pos = a_pos + 1
            repeat_pos = int(torch.randint(b_pos + 1, seq_len - 1, (1,), generator=generator).item())
            A_tok, B_tok = int(x[b, a_pos].item()), int(x[b, b_pos].item())
            x[b, repeat_pos] = A_tok
            x[b, repeat_pos + 1] = B_tok
            loss_mask[b, repeat_pos] = True

    targets = torch.roll(x, shifts=-1, dims=1)
    targets[:, -1] = -100
    loss_mask[:, -1] = False

    return x.to(device), targets.to(device), loss_mask.to(device)


def make_induction_loader(
    batch_size: int, seq_len: int, vocab_size: int, num_triggers: int = 1,
    device: str = "cpu", seed: int | None = None,
):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def _batch_fn():
        return generate_induction_batch(batch_size, seq_len, vocab_size, num_triggers, device, generator)

    return _batch_fn
