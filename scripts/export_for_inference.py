"""
Export a Trainer checkpoint to a clean, minimal inference artifact.

This is the handoff boundary between this repo (research, training,
benchmarking -- "Project A") and the separate inference-serving project
("Project B") that loads exactly this artifact format and nothing else
from here. Project B does not depend on `ssm_lab` at all, on purpose:
training code optimizes for flexibility and experimentation (optimizer
state, RNG state, resumability, 14 architectures behind one registry);
serving code optimizes for a small, auditable dependency footprint and
should not need to import a research codebase to run a forward pass.

A `Trainer.save_checkpoint(...)` file (see `ssm_lab.training.trainer`)
carries far more than a deployed model needs: full optimizer state
(2x the model size for AdamW's moment estimates alone), RNG state for
exact training resumption, and the full `TrainConfig`/`BackboneConfig`
dataclasses, which require importing `ssm_lab` to unpickle at all. This
script reduces that down to exactly three files:

    model.safetensors   -- just the weight tensors, in the safe (no
                            arbitrary code execution on load, unlike
                            pickle-based torch.save) format the rest of
                            the ML ecosystem has standardized on
    config.json          -- plain-dict architecture config (mixer name,
                            d_model, n_layers, mixer_kwargs, ...) -- enough
                            to reconstruct the model with zero ssm_lab
                            imports
    metadata.json         -- what corpus/tokenizer this was trained with,
                            how many steps, and what it achieved, so the
                            artifact is self-documenting

Usage:
    python scripts/export_for_inference.py \\
        --checkpoint checkpoints/s4d_tinyshakespeare_latest.pt \\
        --out artifacts/s4d-tinyshakespeare-v1 \\
        --mixer s4d --tokenizer byte \\
        --corpus-name tiny-shakespeare
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors.torch import save_model

from ssm_lab.layers.s4d import S4DLayer
from ssm_lab.models.backbone import BackboneConfig, SequenceBackbone
from ssm_lab.training.trainer import Trainer, TrainConfig

# Only the mixer needed to *load* a checkpoint for export, not the full
# registry -- this script still imports ssm_lab (it runs inside Project A),
# Project B's loader does not.
_MIXER_CLASSES = {"s4d": S4DLayer}


def export(
    checkpoint_path: str, out_dir: str, mixer_name: str, backbone_kwargs: dict,
    tokenizer_name: str, corpus_name: str, notes: str = "", val_batch_fn=None,
) -> None:
    if mixer_name not in _MIXER_CLASSES:
        raise ValueError(f"unknown mixer '{mixer_name}', add it to _MIXER_CLASSES first")
    mixer_cls = _MIXER_CLASSES[mixer_name]

    cfg = BackboneConfig(**backbone_kwargs)
    model = SequenceBackbone(mixer_cls, cfg)
    trainer = Trainer(model, TrainConfig(device="cpu"), run_name="export")
    trainer.load_checkpoint(checkpoint_path, map_location="cpu")

    os.makedirs(out_dir, exist_ok=True)

    # save_model (not the lower-level save_file) specifically handles PyTorch
    # weight tying correctly -- this backbone ties lm_head.weight to
    # embed.weight by default (see BackboneConfig.tie_weights), and saving
    # both as independent tensors would either double the file size or,
    # worse, silently break the tying on reload. save_model detects shared
    # storage and records it so a plain state_dict load still re-ties them.
    save_model(model, os.path.join(out_dir, "model.safetensors"))

    config = {
        "mixer": mixer_name,
        "vocab_size": cfg.vocab_size,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "d_ff_mult": cfg.d_ff_mult,
        "tie_weights": cfg.tie_weights,
        "mixer_kwargs": dict(cfg.mixer_kwargs),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    metadata = {
        "tokenizer": tokenizer_name,
        "corpus": corpus_name,
        "training_steps": trainer.global_step,
        "source_checkpoint": os.path.basename(checkpoint_path),
        "params": model.num_params(),
        "notes": notes,
    }
    if val_batch_fn is not None:
        import math

        val_loss, val_acc = trainer.evaluate(val_batch_fn, n_iters=20)
        metadata["final_val_loss"] = val_loss
        metadata["final_val_perplexity"] = math.exp(val_loss)
        metadata["final_val_accuracy"] = val_acc
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"exported {model.num_params():,}-param '{mixer_name}' model to {out_dir}/")
    print(f"  - model.safetensors ({os.path.getsize(os.path.join(out_dir, 'model.safetensors')):,} bytes)")
    print("  - config.json")
    print("  - metadata.json", "(with val metrics)" if val_batch_fn is not None else "")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mixer", default="s4d")
    p.add_argument("--d-model", type=int, required=True)
    p.add_argument("--n-layers", type=int, required=True)
    p.add_argument("--d-state", type=int, required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--tokenizer", default="byte")
    p.add_argument("--corpus-name", default="unknown")
    p.add_argument("--notes", default="")
    p.add_argument("--corpus-bin-dir", default=None, help="prepared TextLMData dir, to include real val metrics")
    p.add_argument("--seq-len", type=int, default=128, help="only used with --corpus-bin-dir")
    args = p.parse_args()

    val_batch_fn = None
    if args.corpus_bin_dir:
        from ssm_lab.data.text_lm import TextLMData

        val_data = TextLMData.from_bin_dir(args.corpus_bin_dir)
        val_batch_fn = val_data.get_batch_fn(16, args.seq_len, split="val", device="cpu", seed=0)

    export(
        checkpoint_path=args.checkpoint,
        out_dir=args.out,
        mixer_name=args.mixer,
        backbone_kwargs=dict(
            vocab_size=args.vocab_size, d_model=args.d_model, n_layers=args.n_layers,
            mixer_kwargs=dict(d_state=args.d_state),
        ),
        tokenizer_name=args.tokenizer,
        corpus_name=args.corpus_name,
        notes=args.notes,
        val_batch_fn=val_batch_fn,
    )


if __name__ == "__main__":
    main()
