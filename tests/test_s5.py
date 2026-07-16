import torch

from ssm_lab.layers.s5 import S5Layer


def _make_layer(seed=0, d_model=8, d_state=12):
    torch.manual_seed(seed)
    return S5Layer(d_model=d_model, d_state=d_state)


def test_output_shape_and_dtype():
    layer = _make_layer()
    x = torch.randn(2, 16, 8)
    y = layer(x)
    assert y.shape == x.shape
    assert not torch.is_complex(y)


def test_forward_matches_step_by_step_recurrence():
    """Same verification pattern as S4D: the scan-based forward() and a
    from-first-principles python loop over step() are independent code
    paths. Agreement here is evidence the ZOH discretization and the
    B@u_t / C@h_t matrix-vector products (S5's actual difference from
    S4D) are implemented correctly, not just that something runs."""
    layer = _make_layer(seed=1, d_model=6, d_state=10)
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
    x = torch.randn(2, 10, 8)
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
    layer = _make_layer(seed=4, d_model=4, d_state=6)
    x = torch.randn(1, 10, 4)
    y1 = layer(x)
    x2 = x.clone()
    x2[:, 7:] = torch.randn_like(x2[:, 7:])
    y2 = layer(x2)
    assert torch.allclose(y1[:, :7], y2[:, :7], atol=1e-5)


def test_state_is_shared_across_channels_not_per_channel():
    """The actual architectural claim this model makes, checked directly:
    S5's recurrent state has shape (batch, d_state) -- no d_model
    dimension at all -- unlike S4D's (batch, d_model, d_state). If this
    ever regressed to a per-channel state, S5 would quietly become a
    second copy of S4D instead of the MIMO model it's supposed to be."""
    layer = _make_layer(d_model=8, d_state=12)
    state = layer.init_state(batch_size=3)
    assert state.shape == (3, 12)


def test_B_and_C_actually_mix_channels():
    """A structural sanity check on the MIMO claim: perturbing a *single*
    input channel should, in general, affect *multiple* output channels
    (because B routes every channel into a shared state, and C reads
    every state back out into every channel) -- unlike S4D, where
    channels are independent and a single-channel perturbation only ever
    affects that same output channel."""
    layer = _make_layer(seed=5, d_model=6, d_state=16)
    x = torch.zeros(1, 5, 6)
    y0 = layer(x)
    x_perturbed = x.clone()
    x_perturbed[:, :, 0] = 1.0  # perturb only channel 0
    y1 = layer(x_perturbed)
    diff = (y1 - y0).abs()
    channels_affected = (diff.sum(dim=(0, 1)) > 1e-6).sum().item()
    assert channels_affected > 1, "expected perturbing one input channel to affect multiple output channels"


def test_default_registry_kwargs_build_a_working_layer():
    from ssm_lab.layers.registry import get_mixer

    entry = get_mixer("s5")
    layer = entry["cls"](d_model=16, **entry["default_kwargs"])
    x = torch.randn(2, 8, 16)
    y = layer(x)
    assert y.shape == x.shape
