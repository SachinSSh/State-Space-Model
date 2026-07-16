"""
One trainer, reused for all 14+ models on all 6+ benchmarks (including real
text -- see `ssm_lab.data.text_lm`).

Every benchmark generator in `ssm_lab.benchmarks` (and `ssm_lab.data.text_lm`)
returns the same triple:

    x, y, loss_mask = batch_fn()
    # x, y:        (batch, seq_len) long
    # loss_mask:   (batch, seq_len) bool -- True at positions that are
    #              actually part of the *task* (e.g. the query positions in
    #              MQAR, or every position for real-text LM), False at
    #              positions that are unpredictable context / filler and
    #              would just inject noise into the loss.

so the trainer never needs benchmark-specific code: it computes
cross-entropy and accuracy only where loss_mask is True. Swap
`train_batch_fn` and you've swapped benchmarks (or corpora); swap `model`
and you've swapped architectures. Nothing else in this file changes.

Three things in this file exist specifically to make training *larger*
models than the benchmark-scale defaults workable on free Kaggle/Colab
GPUs, where a single session is capped (Kaggle: 12h; Colab free: variable,
often much less):

  - `global_step` persists on the Trainer instance rather than resetting
    every call to `fit()`, and `save_checkpoint`/`load_checkpoint` capture
    it alongside model/optimizer/RNG state -- so training can stop and
    resume across sessions instead of needing to fit inside one.
  - `grad_accum_steps` accumulates gradients over several micro-batches
    before stepping the optimizer, simulating a larger effective batch
    size than fits in memory at once (the standard way around "I need a
    bigger batch for stable training but not enough VRAM for it").
  - (Gradient *checkpointing* -- trading compute for activation memory --
    lives in `ssm_lab.models.backbone.BackboneConfig`, not here, since
    it's a property of the model's forward pass, but the two are meant to
    be reached for together; see docs/scaling_up.md.)
"""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
BatchFn = Callable[[], Batch]


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 100
    max_steps: int = 2000
    grad_clip: float = 1.0
    grad_accum_steps: int = 1  # >1 simulates a grad_accum_steps-times-larger effective batch
    eval_every: int = 200
    eval_iters: int = 20
    log_every: int = 50
    checkpoint_dir: str | None = None  # if set (with checkpoint_every>0), auto-saves periodically
    checkpoint_every: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype: str = "bf16"  # "bf16" | "fp16" | "fp32" -- fp32 disables autocast entirely
    seed: int = 0


class Trainer:
    def __init__(self, model, cfg: TrainConfig, run_name: str = "run"):
        torch.manual_seed(cfg.seed)
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.run_name = run_name
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas
        )
        self.history: dict[str, list[float]] = {
            "step": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        }
        self.global_step = 0
        self._autocast_enabled = cfg.amp_dtype != "fp32" and cfg.device.startswith("cuda")
        self._autocast_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16

    def _lr_at(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return self.cfg.lr * step / max(1, self.cfg.warmup_steps)
        progress = (step - self.cfg.warmup_steps) / max(1, self.cfg.max_steps - self.cfg.warmup_steps)
        progress = min(progress, 1.0)
        return 0.5 * self.cfg.lr * (1 + math.cos(math.pi * progress))

    def _forward_loss(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, y, mask = batch
        x = x.to(self.cfg.device)
        y = y.to(self.cfg.device)
        mask = mask.to(self.cfg.device)
        with torch.autocast(
            device_type="cuda" if self.cfg.device.startswith("cuda") else "cpu",
            dtype=self._autocast_dtype,
            enabled=self._autocast_enabled,
        ):
            logits = self.model(x)
            per_tok_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="none"
            ).reshape(y.shape)
            mask_f = mask.float()
            denom = mask_f.sum().clamp_min(1.0)
            loss = (per_tok_loss * mask_f).sum() / denom
            with torch.no_grad():
                correct = ((logits.argmax(-1) == y) & mask).sum()
                acc = correct.float() / denom
        return loss, acc

    def fit(self, train_batch_fn: BatchFn, val_batch_fn: BatchFn, max_steps: int | None = None, verbose: bool = True):
        """Runs from `self.global_step` up to `max_steps` (default:
        `cfg.max_steps`). Calling `fit` again later on the same Trainer --
        or on a fresh Trainer that just called `load_checkpoint` -- resumes
        from wherever `global_step` left off rather than restarting."""
        target_steps = max_steps if max_steps is not None else self.cfg.max_steps
        accum_steps = max(1, self.cfg.grad_accum_steps)
        self.model.train()
        t0 = time.time()
        while self.global_step < target_steps:
            self.global_step += 1
            step = self.global_step
            for g in self.opt.param_groups:
                g["lr"] = self._lr_at(step)

            self.opt.zero_grad(set_to_none=True)
            accum_loss, accum_acc = 0.0, 0.0
            for _ in range(accum_steps):
                loss, acc = self._forward_loss(train_batch_fn())
                (loss / accum_steps).backward()
                accum_loss += loss.item() / accum_steps
                accum_acc += acc.item() / accum_steps
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()

            if verbose and step % self.cfg.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"[{self.run_name}] step {step:5d}/{target_steps} | "
                    f"train loss {accum_loss:.4f} acc {accum_acc:.3f} | "
                    f"lr {self._lr_at(step):.2e} | {elapsed:.1f}s"
                )

            if step % self.cfg.eval_every == 0 or step == target_steps:
                val_loss, val_acc = self.evaluate(val_batch_fn, self.cfg.eval_iters)
                self.history["step"].append(step)
                self.history["train_loss"].append(accum_loss)
                self.history["train_acc"].append(accum_acc)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                if verbose:
                    print(f"[{self.run_name}] step {step:5d} |  val  loss {val_loss:.4f} acc {val_acc:.3f}")
                self.model.train()

            if self.cfg.checkpoint_dir and self.cfg.checkpoint_every and step % self.cfg.checkpoint_every == 0:
                self.save_checkpoint(os.path.join(self.cfg.checkpoint_dir, f"{self.run_name}_step{step}.pt"))
        return self.history

    @torch.no_grad()
    def evaluate(self, val_batch_fn: BatchFn, n_iters: int) -> tuple[float, float]:
        self.model.eval()
        losses, accs = [], []
        for _ in range(n_iters):
            loss, acc = self._forward_loss(val_batch_fn())
            losses.append(loss.item())
            accs.append(acc.item())
        return sum(losses) / len(losses), sum(accs) / len(accs)

    # ---- checkpoint save / resume, for training that spans more than one
    #      Kaggle/Colab session ------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        """Captures model, optimizer, global_step, history, and the *global*
        RNG streams (torch/python/numpy-legacy). Honest limitation: every
        loader in `ssm_lab.benchmarks` / `ssm_lab.data.text_lm` draws from
        its *own* independent generator (a `torch.Generator()` or
        `np.random.default_rng()` instance private to that loader closure,
        for its own reproducibility) -- that generator's internal state is
        NOT part of this checkpoint, so resuming does not bit-exactly
        replay the pre-checkpoint batch sequence. It does not need to:
        resumed training still sees a valid, unbiased stream of fresh
        batches, which is what actually matters for training to continue
        correctly. What this checkpoint *does* guarantee exactly: reloading
        it reproduces the identical model (verified in
        tests/test_checkpoint.py by checking eval loss matches before/after
        a save+load round-trip with no training in between) and the
        identical optimizer state, so training resumes as if it had never
        stopped."""
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "global_step": self.global_step,
            "history": self.history,
            "cfg": self.cfg,
            "torch_rng_state": torch.get_rng_state(),
            "python_random_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(payload, path)

    def load_checkpoint(self, path: str, map_location: str | None = None) -> "Trainer":
        payload = torch.load(path, map_location=map_location or self.cfg.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        self.opt.load_state_dict(payload["optimizer_state_dict"])
        self.global_step = payload["global_step"]
        self.history = payload["history"]
        torch.set_rng_state(payload["torch_rng_state"])
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        saved_max_steps = getattr(payload.get("cfg"), "max_steps", None)
        if saved_max_steps is not None and saved_max_steps != self.cfg.max_steps:
            print(
                f"note: checkpoint was saved with cfg.max_steps={saved_max_steps}, "
                f"this Trainer has cfg.max_steps={self.cfg.max_steps} -- the LR "
                f"schedule will follow the *current* cfg, not the saved one."
            )
        return self
