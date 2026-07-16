# Scaling up: from diagnostic probes to a real, larger training run

Every benchmark this repo started with (`ssm_lab.benchmarks`) is a
synthetic algorithmic probe, deliberately small, specifically so a handful
of steps on a free CPU could isolate *why* one architecture beats another.
That's still the right tool for that question. It is not, by itself,
evidence a model works as a language model on real text at a scale anyone
would call "larger." This doc is the added infrastructure for that
separate, legitimate question, and how to actually drive it.

## 1. Real text, not just synthetic probes

`ssm_lab.data` is a new subpackage, separate from `ssm_lab.benchmarks`,
because it does something categorically different: prepares and loads a
*real* corpus instead of generating a synthetic one.

```python
from ssm_lab.data.tokenizer import ByteTokenizer
from ssm_lab.data.text_lm import download_tiny_shakespeare, prepare_corpus, TextLMData

path = download_tiny_shakespeare()          # ~1.1MB, zero setup, for a quick sanity check
text = open(path).read()
tok = ByteTokenizer()                       # vocab=256, zero deps, always round-trips
prepare_corpus(text, tok, out_dir="data/tinyshakespeare_bin")
data = TextLMData.from_bin_dir("data/tinyshakespeare_bin")

train_fn = data.get_batch_fn(batch_size=32, seq_len=128, split="train", device=DEVICE, seed=1)
val_fn   = data.get_batch_fn(batch_size=32, seq_len=128, split="val",   device=DEVICE, seed=2)
# from here it's the exact same Trainer/SequenceBackbone as every synthetic benchmark
```

Verified end-to-end in this repo's own build: a 69K-param S4D model
trained 200 steps on real tiny-shakespeare took validation perplexity from
chance (256, byte-level vocab) down to ~11.4 (`tests/test_text_lm.py`
covers the pipeline; the training run itself is reproduced in
`notebooks/01_s4.ipynb`'s real-LM section).

**Tokenizer choice.** `ByteTokenizer` (vocab=256) is the default because
it's dependency-free and provably round-trips *any* input -- what this
repo's own tests actually verify. `GPT2Tokenizer` (real GPT-2 BPE,
vocab=50257, via `pip install tiktoken`) is available for
comparable-to-published perplexity numbers, but its first-use download
(`openaipublic.blob.core.windows.net`) wasn't reachable from the
network-restricted sandbox this repo was built in, so that path is
implemented and shape-tested, not verified end-to-end here. It's a
standard, widely-used library -- expect it to work on Kaggle/Colab, which
have normal internet access, but confirm on first use there rather than
taking this repo's word for that specific path.

**Bigger than tiny-shakespeare.** Point `prepare_corpus` at anything --
it just needs a Python string. For an actual "does this scale" run instead
of a sanity check:
- **Kaggle**: search Datasets for "WikiText-103" or "FineWeb-Edu", attach
  with one click, point `prepare_corpus` at the file(s) under
  `/kaggle/input/...`.
- **Colab**: `pip install datasets` and stream a slice --
  `load_dataset("HuggingFaceFW/fineweb-edu", streaming=True)` -- concatenate
  a few thousand documents into one string, same `prepare_corpus` call.

RAM usage tracks `batch_size * seq_len`, not corpus size, regardless of
which of these you use -- `TextLMData` opens `train.bin`/`val.bin` as
`np.memmap`, so a multi-GB corpus is never loaded into RAM wholesale, only
the sampled windows are ever paged in. That's what makes "which corpus"
a config choice instead of a rewrite.

## 2. Fitting a larger model in memory: gradient checkpointing

`BackboneConfig(..., use_gradient_checkpointing=True)` recomputes each
layer's activations during backward instead of storing them from the
forward pass -- one extra forward pass per layer, in exchange for not
holding onto every layer's intermediates simultaneously. Verified
numerically transparent (identical loss and gradients with it on vs. off,
`tests/test_grad_checkpointing.py`) -- it only changes memory cost, never
the actual computation.

This isn't a hypothetical trade-off in this repo's own build: a plain
forward+backward at `d_model=64, n_layers=3, d_state=32, seq_len=128,
batch=32` **OOM-killed the process** in the ~4GB-RAM, no-GPU, single-core
sandbox this repo was built and tested in. The identical config, with only
`use_gradient_checkpointing=True` added, completed successfully
(reproduced directly in `tests/test_grad_checkpointing.py`, so this is a
claim you can rerun, not just read). Separately, a moderate config
(`d_model=48, n_layers=4, d_state=24, seq_len=96, batch=16`) measured 2151MB
peak RSS without checkpointing vs. 1283MB with it -- a ~40% reduction, and
that gap widens with depth, since checkpointing's savings scale with
`n_layers`.

```python
cfg = BackboneConfig(vocab_size=data.vocab_size, d_model=512, n_layers=12,
                      mixer_kwargs=dict(d_state=64), use_gradient_checkpointing=True)
```

## 3. Fitting a larger *effective batch size*: gradient accumulation

`TrainConfig(grad_accum_steps=k)` runs `k` micro-batches through forward
+ backward (each scaled by `1/k`) before a single optimizer step --
mathematically equivalent to one backward pass over a batch `k` times
larger (verified to ~1e-8 numerical agreement in
`tests/test_grad_accum.py`), at the peak memory cost of only one
micro-batch at a time. `global_step` (and therefore the LR schedule and
eval cadence) advances once per *optimizer* step, not once per
micro-batch -- accumulation changes what happens inside a step, not how
many steps `fit()` takes.

```python
tcfg = TrainConfig(max_steps=5000, grad_accum_steps=4, ...)  # effective batch = 4x the micro-batch
```

## 4. Scaling presets

`ssm_lab.models.presets` bundles a `BackboneConfig` + recommended
`TrainConfig` overrides per named scale tier. Param counts are computed
directly from this repo's own `num_params()` (real numbers, not
estimates); GPU memory/fit guidance is a rule-of-thumb estimate, flagged
as such -- this repo was built with zero GPU access, so "will this fit on
a T4" is informed by standard scaling behavior, not measured here.

| preset | params (vocab=256) | checkpointing | grad_accum | rough fit |
|---|---|---|---|---|
| `tiny` | ~112K | off | 1 | CPU, seconds -- this repo's own FAST_DEV_RUN scale |
| `small` | ~4.6M | off | 1 | free T4/P100, seq_len ~512-1024, comfortably |
| `medium` | ~34M | on | 2 | free T4/P100 with checkpointing; comfortable on A100/L4 |
| `large` | ~102M | on | 4 | needs checkpointing + accumulation on any free-tier GPU |

```python
from ssm_lab.models.presets import get_preset

cfg, train_overrides = get_preset("medium", vocab_size=data.vocab_size)
model = SequenceBackbone(S4DLayer, cfg)
tcfg = TrainConfig(max_steps=5000, **train_overrides)
```

Override any field: `get_preset("medium", vocab_size=256, n_layers=10)`.

## 5. Training across more than one session: checkpoint and resume

Kaggle caps a session at 12 hours; Colab free tier disconnects on its own
schedule. Past `small`, plan for training to span sessions from the start
rather than treating an interruption as a failure.

```python
# session 1
tcfg = TrainConfig(max_steps=20_000, checkpoint_dir="checkpoints", checkpoint_every=1000, ...)
trainer = Trainer(model, tcfg, run_name="medium_run")
trainer.fit(train_fn, val_fn)     # auto-saves checkpoints/medium_run_step{N}.pt periodically

# session 2 (fresh runtime, same cfg -- max_steps must match if you want the
# original LR schedule to keep making sense; see load_checkpoint's docstring)
trainer = Trainer(SequenceBackbone(S4DLayer, cfg), tcfg, run_name="medium_run")
trainer.load_checkpoint("checkpoints/medium_run_step10000.pt")
trainer.fit(train_fn, val_fn)     # continues from global_step=10000, not from 0
```

What's guaranteed exactly (verified in `tests/test_checkpoint.py`): reloading
reproduces bit-identical eval loss on a fixed batch, restores optimizer
state (AdamW's per-parameter moment estimates, not just the weights), and
restores the global RNG streams (torch/python/numpy-legacy) so a draw
right after reload matches whatever draw would have happened with no
interruption at all. What's *not* guaranteed: each benchmark loader
(`make_mqar_loader`, `TextLMData.get_batch_fn`, etc.) holds its own
independent generator for its own reproducibility, and that generator's
position isn't part of the checkpoint -- resumed training sees a fresh,
still-valid, still-unbiased stream of new batches, not a bit-exact replay
of the pre-checkpoint sequence. That's a real, documented limitation, not
an oversight: exact replay of the data stream isn't something a resumed
training run actually needs.

## 6. Multi-GPU: Kaggle's "T4 x2" and beyond

Kaggle explicitly offers 2x T4 as a GPU option. This repo's `Trainer` is
single-device by design (matching the "runs on a free single-GPU session"
scope everything else here was built and tested against) and multi-GPU
training was **not implemented or tested** in this pass -- there's no
multi-GPU hardware in the sandbox this repo was built in to test it
against, and shipping untested distributed-training code would be worse
than not shipping it. The standard path, for whenever that's the next
piece of work: wrap `model` in `torch.nn.parallel.DistributedDataParallel`,
launch with `torchrun --nproc_per_node=2`, and shard each loader's batch
across ranks. Flagged here as the documented next step, not implemented.

## What doesn't change

Same backbone, same benchmark-suite philosophy, same "one mixer swapped
per model" comparison this whole repo is built around --
`use_gradient_checkpointing`, `grad_accum_steps`, and a real corpus are
config choices layered on the exact same `SequenceBackbone` /`Trainer`
every synthetic benchmark already uses, not a separate scaled-up
codepath to maintain in parallel.
