"""Model loading and intervention semantics."""

import pytest
import torch

from pacneuron.model import Intervention, compose, load_sf_checkpoint, orthogonal_init_like
from pacneuron.hub import checkpoint_path


def test_strict_load_and_frozen_normaliser(model):
    assert not model.training
    assert model.obs_mean.shape == (4, 84, 84) and model.policy.out_features == 9
    x = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    before = model.obs_mean.clone()
    model(x)
    assert torch.equal(before, model.obs_mean)  # normaliser never updates at inference


def test_fc_clamp_equals_logit_bias(model, played):
    """No RNN + linear heads: clamping fc unit n to v shifts logits by exactly (v - h_n) * W_pi[:, n]."""
    x = torch.as_tensor(played["obs"][:64])
    base = model(x)
    n, v = 7, 3.0
    out = model(x, edit=Intervention("fc", [n], v))
    expected = base["logits"] + (v - base["fc"][:, n:n + 1]) * model.policy.weight[:, n]
    torch.testing.assert_close(out["logits"], expected, atol=1e-5, rtol=1e-5)


def test_intervention_only_touches_target(model, played):
    x = torch.as_tensor(played["obs"][:8])
    base = model(x)
    out = model(x, edit=Intervention("conv3", [5], 2.0, cell=(3, 4)))
    diff = (out["conv3"] - base["conv3"]).abs() > 0
    assert diff[:, 5, 3, 4].all() and diff.sum() == 8  # exactly one unit per row changed at the edited site
    torch.testing.assert_close(out["conv2"], base["conv2"])  # upstream untouched


def test_row_gating_and_add_mode(model, played):
    x = torch.as_tensor(played["obs"][:4])
    base = model(x)
    rows = torch.tensor([True, False, True, False])
    out = model(x, edit=Intervention("fc", [1, 2], 1.5, mode="add", rows=rows))
    torch.testing.assert_close(out["fc"][rows][:, [1, 2]], base["fc"][rows][:, [1, 2]] + 1.5)
    torch.testing.assert_close(out["fc"][~rows], base["fc"][~rows])


def test_zero_set_equals_ablation_and_compose(model, played):
    x = torch.as_tensor(played["obs"][:4])
    a = model(x, edit=compose(Intervention("fc", [3], 0.0), Intervention("fc", [4], 0.0)))
    b = model(x, edit=Intervention("fc", [3, 4], 0.0))
    torch.testing.assert_close(a["logits"], b["logits"])
    assert (a["fc"][:, [3, 4]] == 0).all()


def test_orthogonal_control_differs_but_shares_normaliser(model):
    ctrl = orthogonal_init_like(model, seed=0)
    assert torch.equal(ctrl.obs_mean, model.obs_mean)
    assert not torch.allclose(ctrl.fc.weight, model.fc.weight)
    w = ctrl.conv2.weight.flatten(1)  # orthogonal rows (64 x 512)
    torch.testing.assert_close(w @ w.T, torch.eye(64), atol=1e-4, rtol=0)


def test_milestone_checkpoint_loads():
    path = checkpoint_path("1111", "checkpoint_p0/milestones/checkpoint_000025152_51511296.pth")
    assert load_sf_checkpoint(path).fc.weight.shape == (512, 3136)
