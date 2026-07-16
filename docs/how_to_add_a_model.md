# How to add the next model

S4D (`src/ssm_lab/layers/s4d.py`) is the template every subsequent model
follows. This doc is the checklist — copy it per model, check items off.

## 1. Implement the layer

`src/ssm_lab/layers/<name>.py`, subclassing `SequenceMixer`
(`src/ssm_lab/layers/base.py`):

- [ ] `forward(self, x, **kwargs)`: `(batch, seq_len, d_model) -> (batch, seq_len, d_model)`, causal.
- [ ] `step(self, x_t, state)` + `init_state(self, batch_size, ...)` if the
      architecture supports O(1) autoregressive decoding (every model in
      this repo so far does; plain softmax attention, if it's ever added
      as an explicit baseline, would be the one that legitimately can't —
      say so loudly in that case, per the docstring in `base.py`, rather
      than silently falling back to an O(L) cache).
- [ ] Module docstring: paper citation(s), the one-sentence "what changed
      relative to the previous model in the lineage," and an **honest
      scope note** for anything simplified relative to the paper (S4D's
      docstring is the template — say what's traded away and why, don't
      just not mention it).
- [ ] Reuse `ssm_lab.utils.scan` if the architecture reduces to a diagonal
      (or block-diagonal / low-rank-plus-diagonal) linear recurrence —
      most of this lineage does, with different definitions of $(a_t,
      b_t)$. If it genuinely doesn't (e.g. DeltaNet's rank-1 state update
      needs a Householder-product scan, not the elementwise one this repo
      currently has), extend `scan.py` with a new primitive rather than
      duplicating scan logic inside the layer file.
- [ ] If the new model needs a computation an earlier model already has
      inline (S5 needed S4D's diagonal-HiPPO spectrum, just unrepeated
      across channels), extract it into `ssm_lab/utils/` rather than
      copy-pasting a second copy — then **immediately re-run the earlier
      model's full test suite** to confirm the extraction changed nothing
      about its behavior, before the new model ever uses the extracted
      function. (`ssm_lab.utils.hippo.hippo_diag_spectrum` is the example:
      pulled out of `s4d.py` for S5, with `tests/test_s4d.py` re-verified
      passing unchanged as the very next step.)

## 2. Register it

Add one entry to `MIXER_REGISTRY` in `src/ssm_lab/layers/registry.py`
(uncomment the relevant placeholder line, fill in `cls`, `default_kwargs`,
`paper`, `year`, `family`).

## 3. Test it (`tests/test_<name>.py`)

Minimum bar, matching `tests/test_s4d.py`:

- [ ] Output shape/dtype.
- [ ] **Forward matches an independent step-by-step (or chunked, for
      attention-shaped models) recurrence computation** — this is the
      single most valuable test in the whole repo; it is what actually
      catches subtle discretization/scan bugs rather than just checking
      "did it run."
- [ ] Causality (perturb a future token, confirm earlier outputs are
      unchanged).
- [ ] Gradients finite; optimizer step doesn't error (matters most for
      any model with complex or otherwise non-standard parameters).

## 4. Notebook (`notebooks/0N_<name>.ipynb`)

Copy `notebooks/01_s4.ipynb`'s structure exactly:

1. Markdown header: lineage position, the recurrence, citations, honest
   scope note.
2. Setup cell (verbatim — same `REPO_URL` / environment-detection pattern
   in every notebook).
3. Device cell, `FAST_DEV_RUN` flag.
4. `run_benchmark` helper (verbatim, just swap the imported layer class).
5. The same sections in the same order: MQAR, copying, induction
   heads, state-tracking (parity + S5), LRA (once past ListOps-only),
   **real language modeling** (`ssm_lab.data`, same pattern as
   `01_s4.ipynb` section 5 — tiny-shakespeare for `FAST_DEV_RUN`, note
   pointing at WikiText-103/FineWeb-Edu per `docs/scaling_up.md` for the
   real run), throughput-vs-length. Remember to scale the throughput
   section's own benchmark-model size down under `FAST_DEV_RUN`, not just
   the sequence lengths — `01_s4.ipynb`'s first draft didn't, and a fixed
   128-d_model/4-layer benchmark model made "fast" dev runs anything but,
   on CPU. Optionally, a 7th section demonstrating gradient checkpointing
   + accumulation + checkpoint/resume (see `01_s4.ipynb` section 7) — not
   required per-model since it exercises shared infrastructure rather than
   anything model-specific, but cheap to include and a real, running proof
   rather than a claim.
6. Summary cell saving `results/<name>_results.json` — **keep the exact
   key names** (`params`, `final_val_loss`, `final_val_acc`,
   `train_seconds`) so `20_benchmark_suite.ipynb` can load every model's
   results file generically once more models exist.
7. Closing "what to expect" markdown: state the *structural* reason this
   model should beat or lose to S4D on each benchmark, before running it —
   then let the actual numbers confirm or contradict that prediction. A
   surprising result is worth a note in this section, not a quiet edit to
   match expectations.

Tune `FAST_DEV_RUN` configs (steps, model size) the same way §"copying"
in `notebooks/01_s4.ipynb` was tuned: verify convergence with a few
step-count trials before shipping, or explicitly document that a
single-seed dev-run sweep may be noisier than the real run (see that
notebook's copying section for the exact wording to reuse).

## 5. Docs (`docs/0N_<name>.md`)

Same structure as `docs/01_s4.md`: the recurrence, parameterization
choices and why, what's simplified relative to the paper and why that's
defensible, correctness evidence, results and what they mean.

## 6. Update `ROADMAP.md` and the README lineage section

Flip the checkbox, add one line to the results table once real
(`FAST_DEV_RUN=False`, GPU) numbers exist.

## The one rule underneath all of this

Every one of these checklists exists to protect the same thing: a reader
should be able to open any model's notebook and doc page and see *exactly*
what changed relative to the previous model, verified against an
independent computation, with expectations stated before results and
honest notes wherever something didn't come out clean. That's what makes
this a comparative study instead of a model zoo.
