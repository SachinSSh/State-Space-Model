# Roadmap

Full rationale for this exact list: `docs/00_research_lineage.md`. Process
for adding each one: `docs/how_to_add_a_model.md`.

## Shared infrastructure

- [x] `SequenceMixer` interface (`src/ssm_lab/layers/base.py`)
- [x] Parallel scan, verified against sequential ground truth, real + complex (`src/ssm_lab/utils/scan.py`)
- [x] Shared residual backbone (`src/ssm_lab/models/backbone.py`)
- [x] Shared trainer: masked cross-entropy, cosine schedule, mixed precision (`src/ssm_lab/training/trainer.py`)
- [x] Benchmarks: MQAR, copying, state-tracking (parity + S5), induction heads
- [x] Benchmarks: LRA ListOps (fully synthetic, no external data)
- [ ] Benchmarks: LRA Text / Retrieval / Image / Pathfinder (each needs a real external dataset; see `src/ssm_lab/benchmarks/lra/README.md`)
- [x] Model registry + `00_setup_and_smoke_test.ipynb`
- [x] Real-text LM data pipeline (`ssm_lab.data`: tokenizer + memmap-backed corpus loading)
- [x] Scaling infrastructure: gradient checkpointing, gradient accumulation, checkpoint/resume, named scaling presets (`docs/scaling_up.md`)
- [x] Export-for-inference handoff (`scripts/export_for_inference.py`) + companion serving project `ssm-infer` ("Project B") with a real vendored example model
- [ ] `20_benchmark_suite.ipynb` (cross-model comparison table + plots) — add once 3+ models exist

## Models

| # | Model | Family | Status | Notebook | Docs |
|---|-------|--------|--------|----------|------|
| 1 | S4D | foundation | **done** | `01_s4.ipynb` | `01_s4.md` |
| 2 | S5 | foundation | **done** | `02_s5.ipynb` | `02_s5.md` |
| 3 | Mamba / S6 | mamba lineage | planned | | |
| 4 | Mamba-2 / SSD | mamba lineage | planned | | |
| 5 | Mamba-3 | mamba lineage | planned | | |
| 6 | RetNet | linear-attn family | planned | | |
| 7 | GLA | linear-attn family | planned | | |
| 8 | DeltaNet / Gated DeltaNet | linear-attn family | planned | | |
| 9 | RWKV-7 "Goose" | linear-attn family | planned | | |
| 10 | xLSTM (mLSTM) | linear-attn family | planned | | |
| 11 | Jamba-style block | hybrid | planned | | |
| 12 | Griffin | hybrid | planned | | |
| 13 | Vision Mamba | applied | planned | | |
| 14 | Titans (mini) | memory frontier | planned | | |

**Stretch**, once the 14 are solid: Kimi Delta Attention / Gated DeltaNet-2
(2026's decoupled erase/write refinement), HyenaDNA/Caduceus (genomics
track), full LRA (Text/Retrieval/Image/Pathfinder).

## Suggested build order for the next session

Mamba (model 3) next. It's the first model in the roadmap that needs
`ssm_lab.utils.scan` extended to accept an input-dependent `a_t` — both
S4D and S5 use the scan with a *constant* `a_t`, so Mamba's selective
scan is genuinely new ground, not a variation on what's already tested.
Expect this to be the model that finally moves the MQAR/induction-heads
numbers (see `docs/02_s5.md`'s results table for the baseline it needs
to beat).
