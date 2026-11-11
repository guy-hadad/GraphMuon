import copy
import math

import pytest
import torch
from torch import nn

from graphmuon import GraphMuon


def spectral_ns(matrix, steps):
    """Independent singular-value reference for the quintic NS polynomial."""
    u, values, vh = torch.linalg.svd(matrix, full_matrices=False)
    values = values / values.norm().clamp_min(1e-7)
    for _ in range(steps):
        values = 3.4445 * values - 4.775 * values**3 + 2.0315 * values**5
    return (u * values) @ vh


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("diagonal", [False, True])
def test_paper_equations(axis, diagonal):
    torch.manual_seed(7)
    x = torch.randn(20, 5, dtype=torch.float64) * torch.arange(1, 6)
    weight = nn.Parameter(torch.randn((5, 3) if axis == 0 else (3, 5), dtype=torch.float64))
    gradient = torch.randn_like(weight)
    before = weight.detach().clone()
    optimizer = GraphMuon(
        [weight],
        lr=0.03,
        tau=0.4,
        momentum=0.8,
        weight_decay=0.02,
        max_full_dim=0 if diagonal else 256,
        ns_dtype=torch.float64,
    )
    optimizer.observe(weight, x, input_axis=axis)
    weight.grad = gradient.clone()
    optimizer.step()
    covariance = x.T @ x / len(x)
    if diagonal:
        covariance = covariance.diag().diag()
    normalized = covariance / covariance.diag().mean() + 1e-4 * torch.eye(5)
    values, vectors = torch.linalg.eigh(normalized)
    plus = (vectors * values.pow(0.2)) @ vectors.T
    minus = (vectors * values.pow(-0.2)) @ vectors.T
    momentum = (1 + 0.8) * (gradient if axis == 0 else gradient.T)
    direction = minus @ spectral_ns(plus @ momentum, 5)
    scale = ((direction * (covariance @ direction)).sum() / direction.square().sum() + 1e-8).sqrt()
    alpha = 0.03 * math.sqrt(5 / 3)
    expected = (
        before * (1 - alpha * 0.02) - alpha * (direction if axis == 0 else direction.T) / scale
    )
    torch.testing.assert_close(weight, expected, rtol=1e-8, atol=1e-9)


def test_representation_bound_uses_raw_energy():
    torch.manual_seed(9)
    x = 17 * torch.randn(31, 5, dtype=torch.float64)
    p = nn.Parameter(torch.zeros(5, 3, dtype=torch.float64))
    optimizer = GraphMuon([p], lr=0.1, ns_dtype=torch.float64)
    optimizer.observe(p, x, input_axis=0)
    p.grad = torch.randn_like(p)
    covariance = x.T @ x / len(x)
    direction = optimizer._direction(1.95 * p.grad, covariance, optimizer.param_groups[0])
    optimizer.step()
    impact = (x @ p).norm() / math.sqrt(len(x))
    bound = 0.1 * math.sqrt(5 / 3) * direction.norm()
    assert impact <= bound + 1e-10
    torch.testing.assert_close(impact, bound, rtol=1e-7, atol=1e-8)


def test_adamw_matches_torch_and_skips_absent_gradients():
    torch.manual_seed(3)
    p = nn.Parameter(torch.randn(4, 3, dtype=torch.float64))
    q = nn.Parameter(p.detach().clone())
    optimizer = GraphMuon([{"params": [p], "algorithm": "adamw"}], lr=0.002, weight_decay=0.1)
    reference = torch.optim.AdamW([q], lr=0.002, weight_decay=0.1)
    for _ in range(4):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        optimizer.step()
        reference.step()
    torch.testing.assert_close(p, q, rtol=1e-12, atol=1e-12)
    before = p.detach().clone()
    p.grad = None
    optimizer.step()
    assert optimizer.state[p]["step"] == 4
    torch.testing.assert_close(p, before)


def test_missing_observation_fails_before_any_weight_changes():
    p, q = nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(3, 2))
    optimizer = GraphMuon([p, q])
    optimizer.register_graph(q)
    p.grad, q.grad = torch.ones_like(p), torch.ones_like(q)
    with pytest.raises(RuntimeError, match="Missing propagated"):
        optimizer.step()
    torch.testing.assert_close(p, torch.ones_like(p))


def test_accumulation_and_update_frequency():
    p = nn.Parameter(torch.randn(2, 3))
    optimizer = GraphMuon([p], update_freq=2)
    x, y = torch.randn(3, 3), torch.randn(7, 3)
    optimizer.observe(p, x)
    optimizer.observe(p, y)
    p.grad = torch.ones_like(p)
    optimizer.step()
    expected = (x.T @ x + y.T @ y) / 10
    torch.testing.assert_close(optimizer.state[p]["covariance"], expected)
    assert not optimizer.needs_observation(p)
    optimizer.step()  # Cached covariance is allowed between refreshes.
    with pytest.raises(RuntimeError, match="Missing propagated"):
        optimizer.step()
    optimizer.zero_grad()
    optimizer.observe(p, y)
    assert p in optimizer._pending
    optimizer.zero_grad()
    assert not optimizer._pending


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_resume(dtype):
    p = nn.Parameter(torch.randn(3, 3).to(dtype))
    optimizer = GraphMuon([p], update_freq=3)
    optimizer.observe(p, torch.randn(10, 3).to(dtype))
    p.grad = torch.ones_like(p)
    optimizer.step()
    q = nn.Parameter(p.detach().clone())
    resumed = GraphMuon([q])
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    q.grad = p.grad.clone()
    optimizer.step()
    resumed.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert resumed.state[q]["step"] == 2


def test_watch_eval_autocast_and_scheduler():
    layer = nn.Linear(3, 2)
    optimizer = GraphMuon(
        [{"params": [layer.weight]}, {"params": [layer.bias], "algorithm": "adamw", "lr": 0.001}]
    )
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    optimizer.watch(layer)
    x = torch.randn(10, 3)
    layer.eval()
    layer(x)
    assert not optimizer._pending
    layer.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        layer(x).float().square().sum().backward()
    optimizer.step()
    assert optimizer.state[layer.weight]["covariance"].dtype == torch.float32
    schedule.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.002)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.0001)
    optimizer.remove_hooks()
    assert not layer._forward_pre_hooks


def test_closure_and_zero_covariance():
    layer = nn.Linear(3, 2)
    optimizer = GraphMuon(layer.parameters())
    optimizer.watch(layer)

    def closure():
        optimizer.zero_grad()
        loss = layer(torch.zeros(4, 3)).square().sum()
        loss.backward()
        return loss

    assert torch.isfinite(optimizer.step(closure))
    assert torch.isfinite(layer.weight).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tau": 2},
        {"update_freq": 0},
        {"sample_size": -1},
        {"lr": -1},
        {"eps": 0},
        {"ns_steps": 1.5},
        {"momentum": 1},
        {"lr": float("nan")},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        GraphMuon([nn.Parameter(torch.ones(2, 3))], **kwargs)


def test_observation_shape_validation():
    p = nn.Parameter(torch.ones(2, 3))
    optimizer = GraphMuon([p])
    with pytest.raises(ValueError, match="shape"):
        optimizer.observe(p, torch.ones(5, 4))
    with pytest.raises(ValueError, match="zero rows"):
        optimizer.observe(p, torch.ones(0, 3))
