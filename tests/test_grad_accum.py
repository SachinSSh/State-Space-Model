import torch

from ssm_lab.benchmarks.copying import generate_copy_batch
from ssm_lab.layers.s4d import S4DLayer
from ssm_lab.models.backbone import BackboneConfig, SequenceBackbone
from ssm_lab.training.trainer import Trainer, TrainConfig


def test_grad_accum_matches_single_large_batch():
    """grad_accum_steps=k over k micro-batches of size B must produce
    gradients equivalent to a single backward pass over one batch of size
    k*B containing the exact same examples -- the whole point of gradient
    accumulation is that it's mathematically a no-op, just spread over less
    peak memory."""
    torch.manual_seed(0)
    cfg = BackboneConfig(vocab_size=16, d_model=24, n_layers=2, mixer_kwargs=dict(d_state=8))

    micro_bs, k = 8, 4
    x_full, y_full, m_full = generate_copy_batch(
        micro_bs * k, 12, 16, 4, generator=torch.Generator().manual_seed(1)
    )
    chunks = [
        (x_full[i * micro_bs : (i + 1) * micro_bs], y_full[i * micro_bs : (i + 1) * micro_bs], m_full[i * micro_bs : (i + 1) * micro_bs])
        for i in range(k)
    ]

    torch.manual_seed(0)
    model_baseline = SequenceBackbone(S4DLayer, cfg)
    trainer_baseline = Trainer(model_baseline, TrainConfig(device="cpu", amp_dtype="fp32"), run_name="baseline")
    loss, _ = trainer_baseline._forward_loss((x_full, y_full, m_full))
    loss.backward()

    torch.manual_seed(0)
    model_accum = SequenceBackbone(S4DLayer, cfg)
    trainer_accum = Trainer(
        model_accum, TrainConfig(device="cpu", amp_dtype="fp32", grad_accum_steps=k), run_name="accum"
    )
    trainer_accum.opt.zero_grad()
    for chunk in chunks:
        chunk_loss, _ = trainer_accum._forward_loss(chunk)
        (chunk_loss / k).backward()

    for (name_a, p_a), (name_b, p_b) in zip(model_baseline.named_parameters(), model_accum.named_parameters()):
        assert name_a == name_b
        assert torch.allclose(p_a.grad, p_b.grad, atol=1e-4), f"gradient mismatch at {name_a}"


def test_fit_runs_correct_number_of_optimizer_steps_with_accumulation():
    """With grad_accum_steps=k, `global_step` (and therefore the LR
    schedule and eval cadence) should still advance once per *optimizer*
    step, not once per micro-batch -- accumulation changes what happens
    inside a step, not how many steps `fit` takes."""
    from ssm_lab.benchmarks.copying import make_copy_loader

    torch.manual_seed(0)
    cfg = BackboneConfig(vocab_size=12, d_model=16, n_layers=1, mixer_kwargs=dict(d_state=8))
    model = SequenceBackbone(S4DLayer, cfg)
    tcfg = TrainConfig(
        max_steps=10, grad_accum_steps=3, eval_every=10, eval_iters=2,
        log_every=10, warmup_steps=2, device="cpu", amp_dtype="fp32",
    )
    trainer = Trainer(model, tcfg, run_name="accum_steps")
    train_fn = make_copy_loader(8, 10, 12, 3, seed=1)
    val_fn = make_copy_loader(8, 10, 12, 3, seed=2)
    trainer.fit(train_fn, val_fn, verbose=False)
    assert trainer.global_step == 10
