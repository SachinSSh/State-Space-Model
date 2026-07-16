import torch

from ssm_lab.benchmarks.copying import make_copy_loader
from ssm_lab.layers.s4d import S4DLayer
from ssm_lab.models.backbone import BackboneConfig, SequenceBackbone
from ssm_lab.training.trainer import Trainer, TrainConfig


def test_backbone_forward_shape():
    cfg = BackboneConfig(vocab_size=20, d_model=16, n_layers=2, mixer_kwargs=dict(d_state=8))
    model = SequenceBackbone(S4DLayer, cfg)
    x = torch.randint(0, 20, (3, 12))
    logits = model(x)
    assert logits.shape == (3, 12, 20)


def test_tied_weights_share_storage():
    cfg = BackboneConfig(vocab_size=20, d_model=16, n_layers=1, mixer_kwargs=dict(d_state=8), tie_weights=True)
    model = SequenceBackbone(S4DLayer, cfg)
    assert model.lm_head.weight is model.embed.weight


def test_num_params_counts_complex_params_as_two_reals():
    cfg = BackboneConfig(vocab_size=20, d_model=16, n_layers=1, mixer_kwargs=dict(d_state=8))
    model = SequenceBackbone(S4DLayer, cfg)
    manual_count = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
    assert model.num_params() == manual_count
    assert model.num_params() > 0


def test_end_to_end_training_reduces_loss_on_copy_task():
    """
    The one test that proves the whole stack -- benchmark generator, backbone,
    trainer, optimizer, scheduler -- actually trains a model, not just that
    each piece runs without raising. Uses a deliberately easy config so this
    stays fast and CPU-friendly as a CI smoke test.
    """
    torch.manual_seed(0)
    cfg = BackboneConfig(vocab_size=10, d_model=32, n_layers=2, mixer_kwargs=dict(d_state=16))
    model = SequenceBackbone(S4DLayer, cfg)
    tcfg = TrainConfig(
        lr=5e-3, warmup_steps=10, max_steps=250, eval_every=125, eval_iters=5,
        log_every=250, device="cpu", amp_dtype="fp32",
    )
    trainer = Trainer(model, tcfg, run_name="ci_copy_smoke")
    train_fn = make_copy_loader(batch_size=16, seq_len=10, vocab_size=10, copy_len=3, seed=1)
    val_fn = make_copy_loader(batch_size=16, seq_len=10, vocab_size=10, copy_len=3, seed=2)
    hist = trainer.fit(train_fn, val_fn, verbose=False)
    assert hist["val_acc"][-1] > hist["val_acc"][0]  # accuracy improved over training
    assert hist["val_acc"][-1] > 0.4  # comfortably above the ~1/9 chance rate for this vocab
