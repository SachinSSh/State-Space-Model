from ssm_lab.layers.s4d import S4DLayer
from ssm_lab.models.backbone import SequenceBackbone
from ssm_lab.models.presets import PRESETS, get_preset


def test_all_presets_build_valid_configs():
    for name in PRESETS:
        cfg, train_overrides = get_preset(name, vocab_size=256)
        model = SequenceBackbone(S4DLayer, cfg)
        assert model.num_params() > 0
        assert isinstance(train_overrides, dict)
        assert "grad_accum_steps" in train_overrides


def test_preset_scale_is_monotonically_increasing():
    order = ["tiny", "small", "medium", "large"]
    counts = []
    for name in order:
        cfg, _ = get_preset(name, vocab_size=256)
        counts.append(SequenceBackbone(S4DLayer, cfg).num_params())
    assert counts == sorted(counts), "each preset should be strictly larger than the last"


def test_larger_presets_default_to_gradient_checkpointing():
    cfg_small, _ = get_preset("small", vocab_size=256)
    cfg_medium, _ = get_preset("medium", vocab_size=256)
    cfg_large, _ = get_preset("large", vocab_size=256)
    assert cfg_small.use_gradient_checkpointing is False
    assert cfg_medium.use_gradient_checkpointing is True
    assert cfg_large.use_gradient_checkpointing is True


def test_backbone_kwarg_override():
    cfg, _ = get_preset("medium", vocab_size=256, use_gradient_checkpointing=False)
    assert cfg.use_gradient_checkpointing is False
    cfg2, _ = get_preset("medium", vocab_size=256, n_layers=3)
    assert cfg2.n_layers == 3


def test_unknown_preset_raises():
    import pytest

    with pytest.raises(KeyError):
        get_preset("nonexistent", vocab_size=256)
