import numpy as np
import torch

from ssm_lab.data.tokenizer import ByteTokenizer
from ssm_lab.data.text_lm import TextLMData, prepare_corpus


def test_byte_tokenizer_roundtrips_arbitrary_text():
    tok = ByteTokenizer()
    samples = [
        "Hello, world!",
        "To be, or not to be, that is the question.",
        "unicode: café — 你好 😀",
        "",
        "\n\t special \x00 chars \\ ' \" ",
    ]
    for s in samples:
        assert tok.decode(tok.encode(s)) == s


def test_byte_tokenizer_vocab_size():
    tok = ByteTokenizer()
    ids = tok.encode("the quick brown fox")
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert tok.vocab_size == 256


def test_prepare_corpus_and_load(tmp_path):
    text = "abcdefghij " * 500  # small deterministic corpus, > any seq_len used below
    tok = ByteTokenizer()
    meta = prepare_corpus(text, tok, out_dir=str(tmp_path), val_fraction=0.2)

    assert meta["vocab_size"] == 256
    assert meta["n_train_tokens"] + meta["n_val_tokens"] == len(text.encode("utf-8"))
    assert (tmp_path / "train.bin").exists()
    assert (tmp_path / "val.bin").exists()
    assert (tmp_path / "meta.json").exists()

    data = TextLMData.from_bin_dir(str(tmp_path))
    assert data.vocab_size == 256
    assert len(data.train_ids) == meta["n_train_tokens"]
    assert len(data.val_ids) == meta["n_val_tokens"]
    assert isinstance(data.train_ids, np.memmap)


def test_get_batch_fn_shapes_and_shift_relationship(tmp_path):
    text = "the quick brown fox jumps over the lazy dog. " * 200
    tok = ByteTokenizer()
    prepare_corpus(text, tok, out_dir=str(tmp_path), val_fraction=0.1)
    data = TextLMData.from_bin_dir(str(tmp_path))

    batch_fn = data.get_batch_fn(batch_size=4, seq_len=16, split="train", seed=0)
    x, y, mask = batch_fn()
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert mask.shape == (4, 16)
    assert mask.all(), "every position in a real-text LM batch is a real prediction target"
    assert x.dtype == torch.long and y.dtype == torch.long
    assert x.max().item() < data.vocab_size

    # within a sampled window, y is exactly x shifted by one token
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_successive_batches_differ(tmp_path):
    text = "lorem ipsum dolor sit amet " * 300
    tok = ByteTokenizer()
    prepare_corpus(text, tok, out_dir=str(tmp_path), val_fraction=0.1)
    data = TextLMData.from_bin_dir(str(tmp_path))
    batch_fn = data.get_batch_fn(batch_size=4, seq_len=16, seed=0)
    x1, _, _ = batch_fn()
    x2, _, _ = batch_fn()
    assert not torch.equal(x1, x2)


def test_decoded_batch_is_real_text(tmp_path):
    text = "the state space model processes sequences efficiently. " * 50
    tok = ByteTokenizer()
    prepare_corpus(text, tok, out_dir=str(tmp_path), val_fraction=0.1)
    data = TextLMData.from_bin_dir(str(tmp_path))
    batch_fn = data.get_batch_fn(batch_size=2, seq_len=20, seed=0)
    x, _, _ = batch_fn()
    decoded = tok.decode(x[0].tolist())
    # every decoded window must be a literal substring of the source text
    assert decoded in text


def test_seq_len_too_long_raises(tmp_path):
    import pytest

    text = "short"
    tok = ByteTokenizer()
    prepare_corpus(text, tok, out_dir=str(tmp_path), val_fraction=0.2)
    data = TextLMData.from_bin_dir(str(tmp_path))
    with pytest.raises(ValueError):
        data.get_batch_fn(batch_size=2, seq_len=10_000, split="train")()
