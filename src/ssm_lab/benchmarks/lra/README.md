# Long Range Arena (LRA)

Reference: Tay et al., *Long Range Arena: A Benchmark for Efficient
Transformers* (2020) — the benchmark suite S4, S5, and early Mamba all
built their published reputations on. See
`docs/00_research_lineage.md` §7 for why this repo treats LRA as a legacy
comparison point rather than the headline metric (short version: a 2025
re-analysis found most LRA tasks solvable by convolutions with under a
30-token receptive field — a lot of published "long-range" gains were
locality shortcuts).

## Status

- **ListOps** — **implemented** (`listops.py`). Fully synthetic (nested
  MAX/MIN/MEDIAN/SUM_MOD operations over bracketed digit sequences,
  single-digit classification target), matching the spirit of the
  original Nangia & Bowman (2018) generator LRA's ListOps task is built
  on — no external dataset download needed, so it fits this repo's "runs
  anywhere, no data wrangling" constraint the same way the other four
  benchmarks do. Sequence *classification* rather than next-token
  prediction: `loss_mask` is True only at the last real (non-padding)
  token of each example, since the answer isn't knowable until the whole
  expression has been read.
- **Text** (IMDB sentiment, byte-level) — needs the aclImdb dataset.
  Staged; not implemented yet.
- **Retrieval** (byte-level document matching) — needs paired
  long-document data. Staged; not implemented yet.
- **Image** (sequential CIFAR-10, grayscale) — needs CIFAR-10. Staged;
  not implemented yet.
- **Pathfinder** — needs the synthetic Pathfinder image generator (its
  own separate codebase upstream). Staged; not implemented yet.

## Why staged rather than stubbed with fake data

Text/Retrieval/Image/Pathfinder each need a real external dataset to mean
anything — a stub that generates random bytes and calls it "LRA-Text"
would produce numbers that look like a benchmark result but aren't one.
Better to say clearly what's not built yet than to ship something that
quietly measures nothing. When these land, each gets a `<task>.py` with
the same `generate_<task>_batch(...) -> (x, y, loss_mask)` signature as
every other benchmark in this repo, plus a one-time `download_<task>.sh`
or equivalent.
