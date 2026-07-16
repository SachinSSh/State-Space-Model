import torch

from ssm_lab.benchmarks.copying import generate_copy_batch
from ssm_lab.benchmarks.induction_heads import generate_induction_batch
from ssm_lab.benchmarks.mqar import generate_mqar_batch
from ssm_lab.benchmarks.state_tracking import (
    S5_VOCAB_SIZE,
    generate_parity_batch,
    generate_s5_state_tracking_batch,
)


def _check_shapes(x, y, mask, batch_size, seq_len):
    assert x.shape == (batch_size, seq_len)
    assert y.shape == (batch_size, seq_len)
    assert mask.shape == (batch_size, seq_len)
    assert x.dtype == torch.long
    assert mask.dtype == torch.bool


def test_mqar_shapes_and_recall_correctness():
    torch.manual_seed(0)
    num_kv_pairs = 4
    seq_len = 30
    x, y, mask = generate_mqar_batch(batch_size=4, seq_len=seq_len, vocab_size=20, num_kv_pairs=num_kv_pairs)
    _check_shapes(x, y, mask, 4, seq_len)
    assert mask.any(), "MQAR should have at least one scored query position"
    assert not mask[:, : 2 * num_kv_pairs].any(), "presentation phase must never be scored"

    # every masked (query) position's ground-truth target really is the value
    # paired with that key during the presentation phase -- if this weren't
    # true, high accuracy on this benchmark would mean nothing.
    for b in range(4):
        pairs = {}
        for pos in range(0, 2 * num_kv_pairs, 2):
            pairs[int(x[b, pos].item())] = int(x[b, pos + 1].item())
        for t in range(seq_len):
            if mask[b, t]:
                key = int(x[b, t].item())
                assert key in pairs, "queried key must have appeared in the presentation phase"
                assert int(y[b, t].item()) == pairs[key]


def test_copy_task_targets_are_correct():
    torch.manual_seed(0)
    x, y, mask = generate_copy_batch(batch_size=4, seq_len=20, vocab_size=15, copy_len=5)
    _check_shapes(x, y, mask, 4, 20)
    assert mask.sum(dim=1).unique().item() == 4  # copy_len - 1 scored positions per example
    original = x[:, :5]
    second_copy = x[:, 6:11]
    assert torch.equal(original, second_copy)


def test_induction_targets_are_correct():
    torch.manual_seed(0)
    x, y, mask = generate_induction_batch(batch_size=4, seq_len=20, vocab_size=10, num_triggers=2)
    _check_shapes(x, y, mask, 4, 20)
    for b in range(4):
        for t in range(20):
            if mask[b, t]:
                assert int(y[b, t].item()) == int(x[b, t + 1].item())


def test_parity_is_cumulative_xor():
    torch.manual_seed(0)
    x, y, mask = generate_parity_batch(batch_size=4, seq_len=15)
    assert mask.all()
    expected = torch.remainder(torch.cumsum(x, dim=1), 2)
    assert torch.equal(y, expected)


def test_s5_tracking_vocab_and_identity():
    torch.manual_seed(0)
    x, y, mask = generate_s5_state_tracking_batch(batch_size=4, seq_len=8)
    assert mask.all()
    assert x.max().item() < S5_VOCAB_SIZE
    assert y.max().item() < S5_VOCAB_SIZE
    assert S5_VOCAB_SIZE == 120


def test_mqar_rejects_impossible_configs():
    import pytest

    with pytest.raises(ValueError):
        generate_mqar_batch(batch_size=2, seq_len=4, vocab_size=20, num_kv_pairs=10)
