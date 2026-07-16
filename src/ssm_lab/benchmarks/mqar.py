"""
Multi-Query Associative Recall (MQAR).

Reference: Arora, Eyuboglu, Zhang, Timalsina, Alberti, Zinsley, Zou, Rudra
& Re, "Zoology: Measuring and Improving Recall in Efficient Language
Models" (2024) -- the paper that established MQAR performance as strongly
predictive of real downstream language-modeling quality for sub-quadratic
architectures, more so than perplexity alone. It is the reason "how good is
this new linear-time architecture at recall" is now normally answered with
an MQAR sweep before anyone trains a real language model.

Task: a sequence presents `num_kv_pairs` (key, value) token pairs once each
(the "presentation" phase), then repeatedly re-queries previously-seen keys
in random order with random filler gaps between queries (the "query"
phase). The next-token target immediately after a query key is that key's
associated value. Loss is only computed at query-key positions -- both the
presentation phase and the filler gaps are, by construction, unpredictable
from context (nothing has been "seen" yet, or it's random noise), so scoring
them would just measure how well a model learns the marginal token
distribution, not recall.

    presentation:  k1 v1 k2 v2 k3 v3 ...           (loss_mask = 0 throughout)
    query:         k2 v2 <gap> k1 v1 <gap> k3 v3    (loss_mask = 1 at each k_i)

Difficulty knobs, in order of how much they matter empirically (per Zoology):
    1. num_kv_pairs  (more pairs to hold in a fixed-size state = harder)
    2. seq_len       (longer gap between presentation and query = harder)
    3. vocab_size    (larger vocab = keys/values harder to compress into state)
"""
from __future__ import annotations

import torch


def generate_mqar_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_kv_pairs: int,
    gap_range: tuple[int, int] = (0, 4),
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seq_len <= 2 * num_kv_pairs + 2:
        raise ValueError(f"seq_len={seq_len} too short for num_kv_pairs={num_kv_pairs}")
    if vocab_size <= num_kv_pairs + 2:
        raise ValueError(f"vocab_size={vocab_size} too small for num_kv_pairs={num_kv_pairs}")

    x = torch.zeros(batch_size, seq_len, dtype=torch.long)
    loss_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    # Not vectorized across the batch (Python loop per example): batch sizes
    # here are small (tens to low hundreds), this is integer bookkeeping only
    # (no tensor math), and it runs on CPU while the GPU is busy with the
    # previous training step -- clarity over throughput was the right trade
    # for a from-scratch reference implementation. Revisit if this ever shows
    # up as the actual bottleneck in a profile.
    for b in range(batch_size):
        keys = torch.randperm(vocab_size - 1, generator=generator)[:num_kv_pairs] + 1
        values = torch.randint(1, vocab_size, (num_kv_pairs,), generator=generator)

        order = torch.randperm(num_kv_pairs, generator=generator).tolist()
        pos = 0
        for i in order:
            x[b, pos], x[b, pos + 1] = keys[i], values[i]
            pos += 2

        while pos < seq_len - 1:
            i = int(torch.randint(0, num_kv_pairs, (1,), generator=generator).item())
            x[b, pos] = keys[i]
            loss_mask[b, pos] = True  # target at this position (== x[pos+1]) must equal values[i]
            x[b, pos + 1] = values[i]
            pos += 2
            gap = int(torch.randint(gap_range[0], gap_range[1] + 1, (1,), generator=generator).item())
            gap = min(gap, seq_len - 1 - pos)
            if gap > 0:
                x[b, pos : pos + gap] = torch.randint(1, vocab_size, (gap,), generator=generator)
                pos += gap
        if pos < seq_len:
            x[b, pos:] = torch.randint(1, vocab_size, (seq_len - pos,), generator=generator)

    targets = torch.roll(x, shifts=-1, dims=1)
    targets[:, -1] = -100
    loss_mask[:, -1] = False  # no next token to score at the final position

    return x.to(device), targets.to(device), loss_mask.to(device)


def make_mqar_loader(
    batch_size: int, seq_len: int, vocab_size: int, num_kv_pairs: int,
    gap_range: tuple[int, int] = (0, 4), device: str = "cpu", seed: int | None = None,
):
    """Returns a zero-arg callable that yields a fresh random MQAR batch each call."""
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def _batch_fn():
        return generate_mqar_batch(batch_size, seq_len, vocab_size, num_kv_pairs, gap_range, device, generator)

    return _batch_fn
