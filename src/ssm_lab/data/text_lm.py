"""
Real-text language modeling: corpus prep + memory-efficient loading.

Every other benchmark in this repo (`ssm_lab.benchmarks`) is a synthetic
algorithmic probe -- deliberately so, since isolating *why* one architecture
beats another needs a controlled task. But "wins at MQAR" and "works as a
language model" are different claims, and this repo had nothing testing the
second one until this module existed. This is what actually grounds the
comparison in real text.

Design, mirroring the standard (nanoGPT-style) pattern for exactly this
reason -- it's the well-tested way to decouple corpus size from RAM usage:

    prepare_corpus(text, tokenizer, out_dir)   # tokenize once, write to disk
        -> out_dir/train.bin, out_dir/val.bin  # raw token ids, uint16/uint32
    TextLMData.from_bin_dir(out_dir)           # np.memmap, NOT loaded into RAM
        .get_batch_fn(batch_size, seq_len)     # samples random windows,
                                                # only the sampled bytes are
                                                # ever paged in

This is what makes the pipeline "convertible to larger models": a 1MB
tiny-shakespeare run and a multi-GB WikiText-103/FineWeb-Edu run go through
the exact same code path, and RAM usage tracks batch_size * seq_len, not
corpus size, so which corpus you point this at is a config choice, not a
rewrite. See docs/scaling_up.md for how to point this at WikiText-103 (a
one-click Kaggle Dataset attach) or a FineWeb-Edu shard instead of
tiny-shakespeare.

Returns the same (x, y, loss_mask) triple as every benchmark in
`ssm_lab.benchmarks`, so it plugs into the exact same `Trainer` unchanged --
loss_mask is all-True here (every position is real, predictable-in-principle
next-token prediction, unlike the synthetic benchmarks' sparse masks).
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import torch

from ssm_lab.data.tokenizer import Tokenizer

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(dest_path: str = "data/tinyshakespeare.txt") -> str:
    """Fetches Karpathy's ~1.1MB tiny-shakespeare corpus -- the standard
    zero-setup sanity-check text for exactly this kind of pipeline. Returns
    the path it was saved to; no-ops (just returns the path) if already
    present, so this is safe to call at the top of every notebook run."""
    dest = Path(dest_path)
    if dest.exists():
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, dest)
    return str(dest)


def _dtype_for_vocab(vocab_size: int) -> np.dtype:
    return np.uint16 if vocab_size <= 65535 else np.uint32


def prepare_corpus(
    text: str,
    tokenizer: Tokenizer,
    out_dir: str,
    val_fraction: float = 0.1,
) -> dict:
    """Tokenizes `text` once and writes `train.bin` / `val.bin` (raw token
    ids, memmap-friendly dtype) into `out_dir`. Returns a small metadata
    dict (also written as `out_dir/meta.json`) recording vocab_size and
    token counts, so `TextLMData.from_bin_dir` doesn't need the tokenizer
    passed again."""
    ids = tokenizer.encode(text)
    dtype = _dtype_for_vocab(tokenizer.vocab_size)
    arr = np.array(ids, dtype=dtype)

    split = int(len(arr) * (1 - val_fraction))
    train_ids, val_ids = arr[:split], arr[split:]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_ids.tofile(out / "train.bin")
    val_ids.tofile(out / "val.bin")

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "dtype": np.dtype(dtype).name,
        "n_train_tokens": int(len(train_ids)),
        "n_val_tokens": int(len(val_ids)),
    }
    import json

    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


class TextLMData:
    """Holds `train.bin` / `val.bin` as read-only `np.memmap`s (not loaded
    into RAM) plus the metadata needed to interpret them."""

    def __init__(self, train_ids: np.memmap, val_ids: np.memmap, vocab_size: int):
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.vocab_size = vocab_size

    @classmethod
    def from_bin_dir(cls, out_dir: str) -> "TextLMData":
        import json

        out = Path(out_dir)
        with open(out / "meta.json") as f:
            meta = json.load(f)
        dtype = np.dtype(meta["dtype"])
        train_ids = np.memmap(out / "train.bin", dtype=dtype, mode="r")
        val_ids = np.memmap(out / "val.bin", dtype=dtype, mode="r")
        return cls(train_ids, val_ids, meta["vocab_size"])

    def get_batch_fn(
        self,
        batch_size: int,
        seq_len: int,
        split: str = "train",
        device: str = "cpu",
        seed: int | None = None,
    ):
        """Returns a zero-arg callable yielding a fresh (x, y, loss_mask)
        batch each call -- same calling convention as every loader in
        `ssm_lab.benchmarks`. Only the sampled (batch_size, seq_len+1)
        windows are ever materialized as torch tensors; the memmap backing
        array is never fully read into RAM regardless of corpus size."""
        data = self.train_ids if split == "train" else self.val_ids
        if len(data) <= seq_len + 1:
            raise ValueError(
                f"corpus split '{split}' has only {len(data)} tokens, too short for seq_len={seq_len}"
            )
        rng = np.random.default_rng(seed)

        def _batch_fn():
            ix = rng.integers(0, len(data) - seq_len - 1, size=batch_size)
            x = torch.stack([torch.from_numpy(data[i : i + seq_len].astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy(data[i + 1 : i + seq_len + 1].astype(np.int64)) for i in ix])
            loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
            return x.to(device), y.to(device), loss_mask.to(device)

        return _batch_fn
