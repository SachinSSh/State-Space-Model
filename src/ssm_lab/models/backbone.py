"""
The one backbone every model in this repo shares.

This is deliberately a boring, standard pre-norm residual stack (embedding ->
[norm -> mixer -> residual -> norm -> FFN -> residual] x n_layers -> norm ->
head). The entire point of this repo is that the *mixer* is the only thing
that changes between "S4D", "Mamba-3", "RWKV-7", etc. -- everything else
(depth, width, normalization, FFN, optimizer, schedule, the benchmarks) is
held fixed by reusing this class and `ssm_lab.training.trainer.Trainer`
for every one of the 14+ architectures, so differences in a results table
are attributable to the mixer and not to incidental implementation drift.

`use_gradient_checkpointing` is the one config flag in this file aimed
specifically at training larger models than the benchmark-scale defaults:
recomputing each layer's activations during backward instead of storing
them trades compute (one extra forward pass per layer) for memory (no
stored activations to hold onto), which is usually the difference between
"fits on a free 16GB GPU" and "doesn't" once d_model/n_layers/seq_len grow
past diagnostic scale. See docs/scaling_up.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


@dataclass
class BackboneConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    d_ff_mult: int = 4
    dropout: float = 0.0
    tie_weights: bool = True
    use_gradient_checkpointing: bool = False
    mixer_kwargs: dict[str, Any] = field(default_factory=dict)


class SequenceBackbone(nn.Module):
    def __init__(self, mixer_cls, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.norms1 = nn.ModuleList()
        self.mixers = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for i in range(cfg.n_layers):
            self.norms1.append(nn.LayerNorm(cfg.d_model))
            self.mixers.append(mixer_cls(d_model=cfg.d_model, layer_idx=i, **cfg.mixer_kwargs))
            self.norms2.append(nn.LayerNorm(cfg.d_model))
            self.ffns.append(
                nn.Sequential(
                    nn.Linear(cfg.d_model, cfg.d_ff_mult * cfg.d_model),
                    nn.GELU(),
                    nn.Linear(cfg.d_ff_mult * cfg.d_model, cfg.d_model),
                )
            )
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        # complex-valued parameters (e.g. S4D's A/B/C) hold 2x the real scalars
        # their .numel() reports; count real degrees of freedom consistently
        # across mixers so param-matched comparisons are actually param-matched.
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in params)

    def _block_forward(self, x, norm1, mixer, norm2, ffn):
        x = x + self.dropout(mixer(norm1(x)))
        x = x + self.dropout(ffn(norm2(x)))
        return x

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        use_ckpt = self.cfg.use_gradient_checkpointing and self.training
        for norm1, mixer, norm2, ffn in zip(self.norms1, self.mixers, self.norms2, self.ffns):
            if use_ckpt:
                # capture the four modules as default args (not closure-over-loop-var)
                # so each call checkpoints the *right* layer, not whichever the loop
                # variable happens to point to when the backward pass replays this.
                def _fn(x, norm1=norm1, mixer=mixer, norm2=norm2, ffn=ffn):
                    return self._block_forward(x, norm1, mixer, norm2, ffn)

                x = checkpoint(_fn, x, use_reentrant=False)
            else:
                x = self._block_forward(x, norm1, mixer, norm2, ffn)
        x = self.norm_f(x)
        return self.lm_head(x)
