import torch

from ssm_lab.benchmarks.lra.listops import (
    LISTOPS_VOCAB_SIZE,
    PAD_ID,
    TOKEN_TO_ID,
    generate_listops_batch,
    make_listops_loader,
)

_ID_TO_TOKEN = {v: k for k, v in TOKEN_TO_ID.items()}


def test_shapes_and_padding():
    x, y, mask = generate_listops_batch(batch_size=6, seq_len=40, seed=0)
    assert x.shape == (6, 40)
    assert y.shape == (6, 40)
    assert mask.shape == (6, 40)
    assert mask.sum(dim=1).eq(1).all(), "exactly one scored (answer) position per example"


def test_labels_are_valid_digits():
    x, y, mask = generate_listops_batch(batch_size=8, seq_len=48, seed=1)
    for b, t in mask.nonzero(as_tuple=False).tolist():
        assert 0 <= y[b, t].item() <= 9


def test_padding_uses_pad_id_after_content():
    x, y, mask = generate_listops_batch(batch_size=4, seq_len=48, seed=2)
    for b in range(4):
        answer_pos = mask[b].nonzero(as_tuple=False).item()
        assert (x[b, answer_pos + 1 :] == PAD_ID).all(), "everything after the answer must be padding"
        assert x[b, answer_pos] != PAD_ID, "the answer position itself is real content"


def _parse_and_eval(tokens: list[str], pos: int = 0) -> tuple[int, int]:
    """Independent re-implementation of the bottom-up evaluator, used only to
    check the generator's own labels against a from-scratch recomputation."""
    tok = tokens[pos]
    if tok.isdigit():
        return int(tok), pos + 1
    assert tok == "["
    op = tokens[pos + 1]
    pos += 2
    values = []
    while tokens[pos] != "]":
        v, pos = _parse_and_eval(tokens, pos)
        values.append(v)
    pos += 1  # consume ']'
    if op == "MAX":
        return max(values), pos
    if op == "MIN":
        return min(values), pos
    if op == "MED":
        s = sorted(values)
        return s[len(s) // 2], pos
    return sum(values) % 10, pos  # "SM"


def test_generated_label_matches_independent_recomputation():
    x, y, mask = generate_listops_batch(batch_size=10, seq_len=64, max_depth=3, max_args=3, seed=3)
    for b in range(10):
        answer_pos = mask[b].nonzero(as_tuple=False).item()
        tokens = [_ID_TO_TOKEN[i.item()] for i in x[b, : answer_pos + 1]]
        recomputed, consumed = _parse_and_eval(tokens)
        assert consumed == len(tokens), "the expression should exactly fill the non-padded prefix"
        assert recomputed == y[b, answer_pos].item()


def test_loader_produces_varying_batches():
    loader = make_listops_loader(batch_size=4, seq_len=48, seed=0)
    x1, _, _ = loader()
    x2, _, _ = loader()
    assert not torch.equal(x1, x2), "successive loader calls should draw fresh examples"


def test_vocab_size_consistent():
    assert LISTOPS_VOCAB_SIZE == 1 + 10 + 4 + 2  # pad + digits 0-9 + 4 ops + 2 brackets
