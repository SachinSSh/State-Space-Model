import torch

from ssm_lab.benchmarks.copying import generate_copy_batch, make_copy_loader
from ssm_lab.layers.s4d import S4DLayer
from ssm_lab.models.backbone import BackboneConfig, SequenceBackbone
from ssm_lab.training.trainer import Trainer, TrainConfig


def _build_trainer(seed, max_steps=100):
    torch.manual_seed(seed)
    cfg = BackboneConfig(vocab_size=16, d_model=24, n_layers=2, mixer_kwargs=dict(d_state=8))
    model = SequenceBackbone(S4DLayer, cfg)
    tcfg = TrainConfig(
        lr=3e-3, warmup_steps=10, max_steps=max_steps, eval_every=max_steps,
        eval_iters=3, log_every=max_steps, device="cpu", amp_dtype="fp32",
    )
    return Trainer(model, tcfg, run_name="ckpt_test")


def test_checkpoint_roundtrip_reproduces_identical_model(tmp_path):
    """The core guarantee: reloading a checkpoint into a *differently
    initialized* fresh model must reproduce bit-for-bit identical outputs
    on a fixed batch -- this is what makes 'stop and resume later' safe."""
    train_fn = make_copy_loader(16, 12, 16, 4, seed=1)
    val_fn = make_copy_loader(16, 12, 16, 4, seed=2)
    fixed_batch = generate_copy_batch(16, 12, 16, 4, generator=torch.Generator().manual_seed(12345))
    fixed_val_fn = lambda: fixed_batch  # noqa: E731 -- always returns the same batch, on purpose

    trainer_a = _build_trainer(seed=42)
    trainer_a.fit(train_fn, val_fn, verbose=False)
    loss_before, _ = trainer_a.evaluate(fixed_val_fn, n_iters=1)

    ckpt_path = tmp_path / "ckpt.pt"
    trainer_a.save_checkpoint(str(ckpt_path))

    trainer_b = _build_trainer(seed=999)  # deliberately different init
    trainer_b.load_checkpoint(str(ckpt_path))
    loss_after, _ = trainer_b.evaluate(fixed_val_fn, n_iters=1)

    assert abs(loss_before - loss_after) < 1e-6


def test_checkpoint_restores_global_step_and_optimizer_state(tmp_path):
    trainer_a = _build_trainer(seed=1, max_steps=50)
    train_fn = make_copy_loader(16, 12, 16, 4, seed=1)
    val_fn = make_copy_loader(16, 12, 16, 4, seed=2)
    trainer_a.fit(train_fn, val_fn, verbose=False)
    assert trainer_a.global_step == 50

    ckpt_path = tmp_path / "ckpt.pt"
    trainer_a.save_checkpoint(str(ckpt_path))

    trainer_b = _build_trainer(seed=1, max_steps=50)
    assert trainer_b.global_step == 0
    trainer_b.load_checkpoint(str(ckpt_path))
    assert trainer_b.global_step == 50
    # AdamW's per-parameter running moment estimates should be restored, not reset
    for group_a, group_b in zip(
        trainer_a.opt.state_dict()["state"].values(), trainer_b.opt.state_dict()["state"].values()
    ):
        assert torch.equal(group_a["exp_avg"], group_b["exp_avg"])


def test_fit_resumes_past_the_checkpointed_step(tmp_path):
    train_fn = make_copy_loader(16, 12, 16, 4, seed=1)
    val_fn = make_copy_loader(16, 12, 16, 4, seed=2)

    trainer_a = _build_trainer(seed=7, max_steps=200)
    trainer_a.fit(train_fn, val_fn, max_steps=100, verbose=False)
    assert trainer_a.global_step == 100

    ckpt_path = tmp_path / "ckpt.pt"
    trainer_a.save_checkpoint(str(ckpt_path))

    trainer_b = _build_trainer(seed=123, max_steps=200)
    trainer_b.load_checkpoint(str(ckpt_path))
    trainer_b.fit(train_fn, val_fn, max_steps=200, verbose=False)
    assert trainer_b.global_step == 200


def test_rng_stream_continues_identically_after_checkpoint_roundtrip(tmp_path):
    """Saving a checkpoint captures the global RNG state at that instant;
    loading it -- into any Trainer, regardless of what RNG-consuming code
    ran in between -- must restore that exact state, so the very next draw
    after a reload matches whatever draw would have happened right after
    the save with no interruption at all."""
    trainer = _build_trainer(seed=0, max_steps=10)
    torch.manual_seed(7)
    _ = torch.randn(3)  # advance the global RNG to an arbitrary, known point
    ckpt_path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(str(ckpt_path))
    draw_without_reload = torch.randn(3)

    # Deliberately churn the global RNG through unrelated operations before
    # reloading, to prove load_checkpoint's restoration doesn't care what
    # happened in between -- it resets to exactly the saved instant.
    torch.manual_seed(31415)
    _ = torch.randn(100)
    trainer2 = _build_trainer(seed=999, max_steps=10)  # its own __init__ also reseeds internally
    trainer2.load_checkpoint(str(ckpt_path))
    draw_after_reload = torch.randn(3)

    assert torch.equal(draw_without_reload, draw_after_reload)
