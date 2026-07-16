# Research lineage: S4 (2022) → Mamba-3 (2026)

This is the detailed version of the map in `README.md` — what each model
actually changed relative to the one before it, and why that change
mattered enough to be worth its own paper. Model numbers refer to
`ROADMAP.md` / `src/ssm_lab/layers/registry.py`.

## 0. Before S4: why this was hard

Gu et al.'s HiPPO framework (NeurIPS 2020) showed that a linear ODE with a
specific state matrix (HiPPO-LegS) gives a provably optimal online
compression of a function's history into a fixed-size state — the
mathematical basis every model in this repo builds on. LSSL (2021)
plugged HiPPO into a deep sequence model and it worked, but training it
required materializing and differentiating through the state matrix
directly, which was too slow and memory-hungry to scale.

## 1. S4 / S4D — Foundations

**S4** (Gu, Goel & Ré, ICLR 2022) made HiPPO practical: decompose the state
matrix as Normal Plus Low-Rank (NPLR), which admits an efficient
convolution-kernel computation via a Cauchy-kernel / Woodbury-identity
trick — the whole point is a *non-diagonal* A stays cheap to evaluate over
long sequences.

**S4D** (Gu, Gupta, Goel & Ré, NeurIPS 2022) — **implemented in this
repo** — asks: what if we skip the NPLR machinery and just make A
diagonal from the start? A diagonal complex matrix's powers are trivial
elementwise powers, so the kernel falls out directly. Two initializations
approximate the true (non-diagonal) HiPPO-LegS spectrum: S4D-Lin
(imaginary parts linear in the state index) and S4D-Inv (closer to
HiPPO-LegS's true inverse-law falloff, usually better on longer sequences).
This repo implements S4D-Lin by default, selectable to S4D-Inv.

Both S4 and S4D share the defining property that makes this whole later
lineage interesting: **A, B, C are learned constants, not functions of the
input.** The model can choose *how fast* to forget (via the per-channel
step size), but not *what* to remember based on content.

**S5** (Smith, Warrington & Linderman, ICLR 2023) — planned as model 2 —
changes the *shape* of the computation rather than the transition itself:
instead of `d_model` independent single-input-single-output (SISO) SSMs,
one per channel, S5 runs a single multi-input-multi-output (MIMO) SSM
across all channels, evaluated with a parallel associative scan instead
of an FFT convolution. That scan is the direct ancestor of
`ssm_lab.utils.scan.parallel_scan_log` — the same primitive this repo's
S4D implementation already uses, and the same one Mamba's selective scan
generalizes next.

## 2. The Mamba lineage

**Mamba / S6** (Gu & Dao, 2023) — planned as model 3 — makes the single
change that defines this whole era: **A, B, C (or at least B, C, and the
step size) become functions of the input token itself.** The model can
now *select* what to write into its state and what to read out, instead of
applying the same fixed dynamics to every token. This breaks the
convolutional-kernel trick S4/S4D rely on (the kernel is no longer fixed
across time), so Mamba pairs the selective recurrence with a
hardware-aware implementation (fusing the scan, avoiding materializing the
full state in slow GPU memory) to stay fast in practice.

**Mamba-2 / SSD** (Dao & Gu, 2024) — model 4 — asks what happens if you
restrict the state matrix to a scalar times the identity (still
input-dependent, just less expressive per-channel). The answer is *State
Space Duality*: this restricted recurrence is mathematically identical to
a particular masked form of linear attention. That equivalence is why
Mamba-2 can be computed as chunked matrix multiplications on tensor cores
instead of a sequential scan — a large real-world speedup — at the cost of
the same weakness plain linear attention has: reduced accuracy at precise,
fine-grained retrieval compared to Mamba-1's per-channel diagonal state.

**Mamba-3** (Gu, Dao, Kolter et al., 2026) — model 5 — closes the loop back
to S4's original complex HiPPO theory. Three changes: (1) trapezoidal
instead of Euler discretization, a more accurate approximation of the
underlying continuous dynamics; (2) a complex-valued state update
specifically to help with state-tracking (see §4); (3) a MIMO
formulation. At 1.5B-parameter scale, the paper reports beating Gated
DeltaNet by roughly 1.8 points of downstream accuracy while matching
Mamba-2's perplexity with half the state size.

## 3. Linear-attention / delta-rule family

A parallel line of work arrives at the same design space from the
attention side — "state" as online-learned associative memory rather than
a continuous dynamical system.

- **RetNet** (model 6) applies a fixed multi-scale exponential decay —
  no gating, no input-dependence in the transition at all.
- **GLA** (Gated Linear Attention, model 7) adds a data-dependent forget
  gate on top of linear attention — the first input-dependent transition
  in this sub-family.
- **DeltaNet → Gated DeltaNet** (model 8) reframes the state update as an
  online gradient-descent / delta-rule correction: instead of only
  *adding* new key-value associations, it can *overwrite* what's already
  stored for a given key. Gated DeltaNet combines that correction rule
  with gating, dropped into an otherwise ordinary Llama-style
  macro-architecture (self-attention swapped for the gated delta rule).
- **RWKV-7 "Goose"** (model 9) generalizes further with vector-valued
  gating and per-step learning rates. It's provably able to perform state
  tracking and recognize all regular languages — something plain linear
  attention cannot do — while still training in parallel.
- **Kimi Delta Attention** (Moonshot, "Kimi Linear", Oct 2025) refines
  Gated DeltaNet with finer channel-wise gating and is reported as the
  first linear-attention hybrid to beat full attention in a fair
  comparison across short-context, long-context, and RL regimes — the
  current production high-water mark of this family. Not yet in the
  model list below; a strong candidate once the 14 are done (see
  `ROADMAP.md`).
- **xLSTM** (model 10) takes the *other* route back to this same design
  space: modernize the LSTM itself — exponential gating, a parallelizable
  matrix-valued memory (mLSTM) — rather than starting from attention. An
  October 2025 scaling-law study found xLSTM Pareto-dominant over
  Transformers at matched training compute.

## 4. Why state-tracking is its own axis

Merrill, Petty & Sabharwal ("The Illusion of State in State-Space
Models," 2024) formalize a specific limitation: any sequence model whose
state transition is a **diagonal**, decay-only linear map is restricted to
computations in TC⁰ — the same complexity class as plain softmax
attention. S5 permutation composition is NC¹-complete and is the standard
benchmark used to demonstrate this in practice (parity, also in this
repo's benchmark suite, is a gentler TC⁰ sanity check).

This is precisely the gap DeltaNet's delta-rule correction, RWKV-7's
generalized delta rule, and Mamba-3's complex-valued state update are each,
in their own way, designed to close — and precisely why this repo expects
(and, per the S4D results already in `notebooks/01_s4.ipynb`, observes)
a plain diagonal-decay model to struggle on the state-tracking benchmarks
specifically, not on every benchmark uniformly.

## 5. Hybrids: why nobody ships any of this pure

- **Jamba** (AI21) interleaves Transformer and Mamba layers at roughly a
  1:7 attention-to-Mamba ratio, with MoE layered in periodically.
- **Nemotron-H** (NVIDIA), **Zamba2** (Zyphra), and **Bamba** (IBM) follow
  close variants of the same idea — Nemotron-H specifically keeps
  self-attention in only about 8% of its layers.
- **Griffin** (Google DeepMind) hybridizes a gated linear recurrence
  (RG-LRU) with local sliding-window attention, and matched Llama-2's
  performance on over six times fewer training tokens (shipped
  publicly as RecurrentGemma).

The reason every serious hybrid keeps some real attention is
well-documented empirically: 2025 benchmarking consistently found hybrids
winning on long-context memory efficiency, while pure attention still
wins specifically on associative recall and in-context learning — the
practical shadow of the Jelassi et al. copying-capacity result below.

**Copying and the fixed-state ceiling.** Jelassi, Brandfonbrener, Kakade &
Malach ("Repeat After Me," 2024) prove that any model with state size
independent of sequence length — any SSM, any linear-attention/RNN
variant — cannot copy an arbitrary string once it exceeds what that fixed
state can hold, while attention (whose "state" is the entire growing KV
cache) can. `ssm_lab.benchmarks.copying` makes this failure directly
observable by sweeping copy length at fixed state size.

## 6. Frontier: memory and other domains

**Titans** (Google, 2024/25, model 13, "mini" version in this repo) adds a
small neural-memory module that learns *at test time* what to memorize
from historical context, while still training fast and in parallel —
attention handles short-term memory, the neural module handles long-term.

**Vision Mamba** (model 12) turns the scan bidirectional to build a vision
backbone that beats DeiT on ImageNet classification and downstream
detection/segmentation while being more compute- and memory-efficient —
the one deliberately non-causal model in this repo.

**HyenaDNA / Caduceus** (staged; see `ROADMAP.md`) apply the same
recurrence to genomics: HyenaDNA reaches million-token, single-nucleotide
resolution context, a large jump over prior attention-based genomic
models.

## 7. A methodological note on Long Range Arena

LRA (Tay et al., 2020) is the benchmark suite S4, S5, and early Mamba all
built their reputations on. Recent analysis (Miralles-González et al.,
2025) found most LRA tasks solvable by small convolutions with a
receptive field under 30 tokens — a lot of the published "long-range"
gains on LRA were really just locality shortcuts, not genuine long-range
reasoning. This repo keeps a synthetic LRA task (ListOps) as a legacy
comparison point precisely because it's the benchmark the field used to
compare against — but treats MQAR / associative-recall sweeps as the more
predictive diagnostic, per current practice (see Zoology, below).

## References

- Gu, Dao, Ermon, Rudra & Ré. *HiPPO: Recurrent Memory with Optimal
  Polynomial Projections.* NeurIPS 2020.
- Gu, Goel & Ré. *Efficiently Modeling Long Sequences with Structured
  State Spaces.* ICLR 2022. (S4)
- Gu, Gupta, Goel & Ré. *On the Parameterization and Initialization of
  Diagonal State Space Models.* NeurIPS 2022. (S4D)
- Smith, Warrington & Linderman. *Simplified State Space Layers for
  Sequence Modeling.* ICLR 2023. (S5)
- Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces.* 2023.
- Dao & Gu. *Transformers are SSMs: Generalized Models and Efficient
  Algorithms Through Structured State Space Duality.* 2024. (Mamba-2)
- Gu, Dao, Kolter et al. *Mamba-3.* 2026 (ICLR 2026 submission).
- Sun et al. *Retentive Network: A Successor to Transformer for Large
  Language Models.* 2023. (RetNet)
- Yang, Wang, Shen, Panda & Kim. *Gated Linear Attention Transformers with
  Hardware-Efficient Training.* 2024. (GLA)
- Yang, Wang, Zhang, Shen & Kim. *Parallelizing Linear Transformers with
  the Delta Rule over Sequence Length.* 2024. (DeltaNet)
- Yang et al. *Gated Delta Networks: Improving Mamba2 with Delta Rule.*
  2024/25.
- Peng et al. *RWKV-7 "Goose" with Expressive Dynamic State Evolution.*
  2025.
- Moonshot AI / Kimi Team. *Kimi Linear: An Expressive and Efficient
  Attention Architecture.* Oct 2025.
- Beck et al. *xLSTM: Extended Long Short-Term Memory.* 2024, with 2025
  scaling-law follow-ups.
- Lieber et al. (AI21). *Jamba: A Hybrid Transformer-Mamba Language
  Model.* 2024.
- NVIDIA. *Nemotron-H: A Family of Accurate and Efficient Hybrid
  Mamba-Transformer Models.* 2025.
- De, Smith, Fernando et al. (Google DeepMind). *Griffin: Mixing Gated
  Linear Recurrences with Local Attention for Efficient Language Models.*
  2024.
- Behrouz et al. (Google). *Titans: Learning to Memorize at Test Time.*
  2024/25.
- Zhu et al. *Vision Mamba: Efficient Visual Representation Learning with
  Bidirectional State Space Model.* 2024.
- Nguyen et al. *HyenaDNA: Long-Range Genomic Sequence Modeling at Single
  Nucleotide Resolution.* 2023.
- Schiff et al. *Caduceus: Bi-Directional Equivariant Long-Range DNA
  Sequence Modeling.* 2024.
- Jelassi, Brandfonbrener, Kakade & Malach. *Repeat After Me: Transformers
  are Better than State Space Models at Copying.* 2024.
- Merrill, Petty & Sabharwal. *The Illusion of State in State-Space
  Models.* 2024.
- Arora, Eyuboglu, Zhang et al. *Zoology: Measuring and Improving Recall
  in Efficient Language Models.* 2024.
- Olsson et al. (Anthropic). *In-context Learning and Induction Heads.*
  2022.
- Tay et al. *Long Range Arena: A Benchmark for Efficient Transformers.*
  2020.
- Miralles-González et al. *(critique of LRA's locality confound).* 2025.

Where a claim above matters enough to double-check before citing it
yourself (exact numbers, exact dates), verify against the primary source —
paper titles/venues here are given as pointers for further reading, not
guaranteed bibliographic detail.
