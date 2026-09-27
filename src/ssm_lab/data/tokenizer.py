"""
Tokenizers for real-text language modeling.

Every synthetic benchmark in `ssm_lab.benchmarks` works over small,
task-specific vocabularies the generator controls directly. Real text needs
an actual tokenizer, and this repo deliberately keeps two tiers:

- `ByteTokenizer` (default): vocab=256, zero dependencies, encode/decode is
  a total function on *any* input (every byte string round-trips), and it
  is exactly the scheme nanoGPT's char-rnn-style "tiny shakespeare" demo and
  GPT-2's byte-level BPE both build on -- just without the BPE merge step.
  Works identically offline, in this sandbox, and on Kaggle/Colab.
- `GPT2Tokenizer` (optional): real GPT-2 byte-pair encoding via `tiktoken`
  (`pip install tiktoken`), vocab=50257. Meaningfully-comparable-to-published
  perplexity numbers, at the cost of an external dependency that needs to
  download its merge table from `openaipublic.blob.core.windows.net` on
  first use. Note on scope: that download endpoint isn't reachable from the
  sandbox this repo was built in (network-restricted), so this path is
  implemented and shape-tested but its happy path is NOT verified end-to-end
  in this repo's own CI -- only the byte-level path carries that guarantee.
  It is a well-established library; expect it to work fine on Kaggle/Colab,
  which have unrestricted internet access, but verify on first use there.
"""
from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids) -> str: ...


class ByteTokenizer:
    """UTF-8 byte-level tokenizer. vocab_size=256. Always round-trips."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


class GPT2Tokenizer:
    """Real GPT-2 BPE via `tiktoken`. Requires `pip install tiktoken` and,
    on first use, network access to fetch its merge table."""

    def __init__(self):
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "GPT2Tokenizer needs `pip install tiktoken` (not a core "
                "requirement of this repo -- see requirements.txt)."
            ) from e
        self._enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def decode(self, ids) -> str:
        return self._enc.decode(list(ids))


def get_tokenizer(name: str = "byte") -> Tokenizer:
    if name == "byte":
        return ByteTokenizer()
    if name == "gpt2":
        return GPT2Tokenizer()
    raise ValueError(f"unknown tokenizer '{name}', expected 'byte' or 'gpt2'")
