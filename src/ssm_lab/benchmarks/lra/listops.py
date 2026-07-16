"""
ListOps, from the Long Range Arena suite.

Reference: Nangia & Bowman, "ListOps: A Diagnostic Dataset for Latent Tree
Learning" (2018); adopted as one of the six Long Range Arena tasks (Tay et
al., 2020). See `src/ssm_lab/benchmarks/lra/README.md` and
`docs/00_research_lineage.md` §7 for why this repo treats LRA as a legacy
comparison point rather than the headline benchmark.

Fully synthetic (unlike Text/Retrieval/Image/Pathfinder in this same
suite, which need real external datasets and are staged rather than
implemented -- see the README above) -- a nested-bracket expression over
single-digit operands and four operators, e.g.

    [ MAX 2 9 [ MIN 4 7 ] 0 ]  ->  9

evaluated bottom-up: MAX/MIN are the obvious reductions, MED is the
median, SM is the sum of its arguments mod 10. The label is always a
single digit 0-9. Unlike this repo's other benchmarks, ListOps is *sequence
classification*: the whole expression must be read before the answer is
knowable, so loss_mask is True only at the last real (non-padding)
position of each example -- there is nothing to score before the closing
bracket of the outermost expression.
"""
from __future__ import annotations

import random

import torch

OPS = ("MAX", "MIN", "MED", "SM")
_VOCAB = ["<pad>"] + [str(d) for d in range(10)] + list(OPS) + ["[", "]"]
TOKEN_TO_ID = {tok: i for i, tok in enumerate(_VOCAB)}
PAD_ID = TOKEN_TO_ID["<pad>"]
LISTOPS_VOCAB_SIZE = len(_VOCAB)
NUM_CLASSES = 10  # the label is always a single digit


def _make_expr(max_depth: int, max_args: int, rng: random.Random) -> tuple[list[str], int]:
    if max_depth <= 0 or rng.random() < 0.35:
        d = rng.randint(0, 9)
        return [str(d)], d

    op = rng.choice(OPS)
    n_args = rng.randint(2, max_args)
    tokens = ["[", op]
    values = []
    for _ in range(n_args):
        sub_tokens, sub_val = _make_expr(max_depth - 1, max_args, rng)
        tokens.extend(sub_tokens)
        values.append(sub_val)
    tokens.append("]")

    if op == "MAX":
        value = max(values)
    elif op == "MIN":
        value = min(values)
    elif op == "MED":
        s = sorted(values)
        value = s[len(s) // 2]
    else:  # "SM" - sum mod 10
        value = sum(values) % 10
    return tokens, value


def generate_listops_batch(
    batch_size: int,
    seq_len: int,
    max_depth: int = 4,
    max_args: int = 4,
    device: str = "cpu",
    seed: int | None = None,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rng is None:
        rng = random.Random(seed)

    x = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    loss_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    b, attempts, max_attempts = 0, 0, batch_size * 50
    while b < batch_size:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate {batch_size} ListOps examples that fit in "
                f"seq_len={seq_len} after {max_attempts} attempts. Increase "
                f"seq_len or reduce max_depth/max_args."
            )
        tokens, value = _make_expr(max_depth, max_args, rng)
        if len(tokens) > seq_len:
            continue
        ids = torch.tensor([TOKEN_TO_ID[t] for t in tokens], dtype=torch.long)
        x[b, : len(ids)] = ids
        last_pos = len(ids) - 1
        targets[b, last_pos] = value
        loss_mask[b, last_pos] = True
        b += 1

    return x.to(device), targets.to(device), loss_mask.to(device)


def make_listops_loader(
    batch_size: int, seq_len: int, max_depth: int = 4, max_args: int = 4,
    device: str = "cpu", seed: int | None = None,
):
    """Unlike the tensor-`torch.Generator`-based loaders elsewhere in this repo,
    ListOps generation is recursive Python, so this holds a plain `random.Random`
    instance across calls -- same idea (one persistent, seeded source of
    randomness per loader), different mechanism, because the generation
    itself is host-side recursion, not a batched tensor op."""
    rng = random.Random(seed)

    def _batch_fn():
        return generate_listops_batch(batch_size, seq_len, max_depth, max_args, device=device, rng=rng)

    return _batch_fn
