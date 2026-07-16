import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from export_for_inference import export  # noqa: E402

from ssm_lab.benchmarks.copying import make_copy_loader
from ssm_lab.layers.s4d import S4DLayer
from ssm_lab.models.backbone import BackboneConfig, SequenceBackbone
from ssm_lab.training.trainer import Trainer, TrainConfig


def test_export_produces_a_loadable_artifact_matching_the_trained_model(tmp_path):
    torch.manual_seed(0)
    cfg = BackboneConfig(vocab_size=20, d_model=16, n_layers=2, mixer_kwargs=dict(d_state=8))
    model = SequenceBackbone(S4DLayer, cfg)
    tcfg = TrainConfig(max_steps=20, eval_every=20, eval_iters=2, log_every=20, device="cpu", amp_dtype="fp32")
    trainer = Trainer(model, tcfg, run_name="export_test")
    train_fn = make_copy_loader(8, 10, 20, 3, seed=1)
    val_fn = make_copy_loader(8, 10, 20, 3, seed=2)
    trainer.fit(train_fn, val_fn, verbose=False)

    ckpt_path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(str(ckpt_path))

    out_dir = tmp_path / "exported"
    export(
        checkpoint_path=str(ckpt_path),
        out_dir=str(out_dir),
        mixer_name="s4d",
        backbone_kwargs=dict(vocab_size=20, d_model=16, n_layers=2, mixer_kwargs=dict(d_state=8)),
        tokenizer_name="byte",
        corpus_name="test",
    )

    assert (out_dir / "model.safetensors").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "metadata.json").exists()

    with open(out_dir / "config.json") as f:
        config = json.load(f)
    assert config == {
        "mixer": "s4d", "vocab_size": 20, "d_model": 16, "n_layers": 2,
        "d_ff_mult": 4, "tie_weights": True, "mixer_kwargs": {"d_state": 8},
    }

    # reload into a *fresh* model and confirm identical outputs to the
    # original trained model -- the same parity property Project B's own
    # test suite checks against, verified from this side of the boundary too
    from safetensors.torch import load_model

    backbone_kwargs = {k: v for k, v in config.items() if k != "mixer"}
    reloaded = SequenceBackbone(S4DLayer, BackboneConfig(**backbone_kwargs))
    missing, unexpected = load_model(reloaded, str(out_dir / "model.safetensors"))
    assert not missing and not unexpected
    reloaded.eval()
    model.eval()

    x = torch.randint(0, 20, (2, 10))
    with torch.no_grad():
        original_logits = model(x)
        reloaded_logits = reloaded(x)
    assert torch.allclose(original_logits, reloaded_logits, atol=1e-5)


def test_export_includes_val_metrics_when_batch_fn_given(tmp_path):
    torch.manual_seed(0)
    cfg = BackboneConfig(vocab_size=20, d_model=16, n_layers=1, mixer_kwargs=dict(d_state=8))
    model = SequenceBackbone(S4DLayer, cfg)
    trainer = Trainer(model, TrainConfig(max_steps=5, device="cpu", amp_dtype="fp32"), run_name="export_test2")
    val_fn = make_copy_loader(8, 10, 20, 3, seed=2)
    trainer.fit(val_fn, val_fn, verbose=False)
    ckpt_path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(str(ckpt_path))

    out_dir = tmp_path / "exported2"
    export(
        checkpoint_path=str(ckpt_path), out_dir=str(out_dir), mixer_name="s4d",
        backbone_kwargs=dict(vocab_size=20, d_model=16, n_layers=1, mixer_kwargs=dict(d_state=8)),
        tokenizer_name="byte", corpus_name="test", val_batch_fn=val_fn,
    )
    with open(out_dir / "metadata.json") as f:
        metadata = json.load(f)
    assert "final_val_loss" in metadata
    assert "final_val_perplexity" in metadata
