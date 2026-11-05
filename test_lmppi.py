import math

import numpy as np
import torch

from lmppi.data import H_HIST, collect, to_transitions, windows
from lmppi.models import (EnsembleModel, KoopmanModel, RFFGPModel, _delta, deadzone, greybox_model,
                          oracle_model)
from lmppi.mppi import MPPI
from lmppi.plant import TRUE, MysteryCartPole, cartpole_deriv, wrap


def test_torch_oracle_matches_numpy_plant():
    """The torch physics model used for prediction/MPPI and the numpy plant must agree."""
    plant = MysteryCartPole(sensor_std=np.zeros(4))
    x0 = np.array([0.1, 2.8, 0.3, -1.0])
    plant.reset(x0)
    m = oracle_model()
    m.substeps = 20  # same 1 kHz integration as the plant
    U = np.array([5.0, -6.0, 3.0, 8.0, -2.5])
    xs_plant = [plant.step(u) for u in U]
    X = m.rollout(x0[None], np.zeros((1, H_HIST)), U[None]).numpy()[0, 1:]
    # Only difference is the smooth (eps = 0.02 N) vs hard deadzone corner.
    assert np.max(np.abs(X - np.array(xs_plant))) < 2e-3


def test_deadzone_is_monotone_and_matches_hard_version_away_from_corner():
    u = torch.linspace(-10, 10, 2001, dtype=torch.float64)
    y = deadzone(u, torch.tensor(1.0, dtype=torch.float64))
    assert torch.all(y[1:] >= y[:-1])
    far = u.abs() > 1.2
    hard = torch.sign(u) * (u.abs() - 1.0)
    assert torch.max(torch.abs(y[far] - hard[far])) < 1e-3


def test_delta_wraps_angle():
    x = torch.tensor([[0.0, math.pi - 0.01, 0.0, 0.0]])
    xn = torch.tensor([[0.0, -math.pi + 0.01, 0.0, 0.0]])
    assert abs(_delta(xn, x)[0, 1].item() - 0.02) < 1e-6


def test_rff_features_approximate_rbf_kernel():
    m = RFFGPModel(D=20000)
    g = torch.Generator().manual_seed(0)
    ell = 1.5
    m.W = torch.randn(3, m.D, generator=g) / ell
    m.b = 2 * math.pi * torch.rand(m.D, generator=g)
    a, b = torch.tensor([[0.2, -0.5, 1.0]]), torch.tensor([[0.9, 0.1, 0.3]])
    approx = (m._phi(a) * m._phi(b)).sum().item()
    exact = math.exp(-((a - b) ** 2).sum().item() / (2 * ell**2))
    assert abs(approx - exact) < 0.02


def test_cartpole_upright_is_an_equilibrium():
    d = cartpole_deriv(np.array([0.0, math.pi, 0.0, 0.0]), 0.0, TRUE)
    assert np.allclose(d, 0.0, atol=1e-12)


def test_small_pipeline_end_to_end():
    """Collect a little data, fit fast versions of each learned model, and check they beat
    a constant-state predictor over 10 steps."""
    eps = collect(12, 150, seed=5)
    tr, va = eps[:9], eps[9:]
    w = windows(va, 10)
    base = np.mean(np.abs(wrap(w["xt"][:, -1, 1] - w["x0"][:, 1])))
    for m in [EnsembleModel(M=2, epochs=8), RFFGPModel(D=300), KoopmanModel(n_rbf=20)]:
        m.fit(tr, va, log=lambda *_: None)
        X = m.rollout(w["x0"], w["uh"], w["u"]).double().numpy()
        err = np.mean(np.abs(wrap(X[:, -1, 1] - w["xt"][:, -1, 1])))
        assert err < base, m.name
    assert to_transitions(eps)["uh"].shape[1] == H_HIST


def test_greybox_gradient_reduces_loss():
    eps = collect(6, 100, seed=6)
    m = greybox_model()
    before = m.params()[5].item()
    m.fit(eps, eps, log=lambda *_: None, iters=40, batch=64)
    after = m.params()[5].item()
    assert after > before  # actuator time constant moves from 0.03 s toward the true 0.10 s


def test_mppi_balances_with_true_model():
    plant = MysteryCartPole(rng=np.random.default_rng(1))
    obs = plant.reset([0.0, math.pi - 0.15, 0.0, 0.0])
    ctrl = MPPI(oracle_model(), K=256, H=30, seed=0)
    uh = np.zeros(H_HIST)
    for _ in range(75):
        u = ctrl.act(obs, uh)
        obs = plant.step(u)
        uh = np.concatenate([[u], uh[:-1]])
    assert abs(wrap(obs[1] - math.pi)) < math.radians(5)
