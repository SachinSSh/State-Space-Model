import torch

from ssm_lab.layers.s4d import S4DLayer


def _make_layer(seed=0, d_model=8, d_state=4):
    torch.manual_seed(seed)
    return S4DLayer(d_model=d_model, d_state=d_state)


def test_output_shape_and_dtype():
    layer = _make_layer()
    x = torch.randn(2, 16, 8)
    y = layer(x)
    assert y.shape == x.shape
    assert not torch.is_complex(y)


def test_forward_matches_step_by_step_recurrence():
    """
    The scan-based forward() and a from-first-principles python loop over
    step() are two independent code paths. Agreement between them is real
    evidence the discretization/scan math is correct, not just
    self-consistent with itself.
    """
    layer = _make_layer(seed=1, d_model=6, d_state=5)
    x = torch.randn(3, 12, 6)

    y_forward = layer(x)

    state = layer.init_state(batch_size=3)
    y_steps = []
    for t in range(x.shape[1]):
        y_t, state = layer.step(x[:, t], state)
        y_steps.append(y_t)
    y_step = torch.stack(y_steps, dim=1)

    assert torch.allclose(y_forward, y_step, atol=1e-4)


def test_gradients_are_finite_through_complex_params():
    layer = _make_layer(seed=2)
    x = torch.randn(2, 10, 8, requires_grad=False)
    y = layer(x)
    y.pow(2).mean().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_adamw_step_on_complex_params_does_not_error():
    layer = _make_layer(seed=3)
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    x = torch.randn(2, 10, 8)
    layer(x).pow(2).mean().backward()
    opt.step()  # should not raise


def test_causal_no_peeking_at_future():
    """Perturbing a future token must not change the current output."""
    layer = _make_layer(seed=4, d_model=4, d_state=4)
    x = torch.randn(1, 10, 4)
    y1 = layer(x)
    x2 = x.clone()
    x2[:, 7:] = torch.randn_like(x2[:, 7:])  # change everything from t=7 onward
    y2 = layer(x2)
    assert torch.allclose(y1[:, :7], y2[:, :7], atol=1e-5)
