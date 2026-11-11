"""Model-predictive path integral control (Williams et al. 2017), batched in torch.

Each control step samples K perturbed control sequences around the current plan, rolls them
out through the model, and re-weights the perturbations by exp(-cost / lambda). With an
ensemble model, sample k is rolled out through member k mod M, so the plan is scored against
every plausible model rather than against the mean.
"""
import time
from dataclasses import dataclass

import numpy as np
import torch

from .data import H_HIST
from .plant import DT, U_MAX, MysteryCartPole, wrap

TRACK_HALF_LENGTH_M = 2.0  # beyond this the cart is off the track; also the success bound


@dataclass(frozen=True)
class CostWeights:
    upright: float = 8.0          # times (1 + cos theta): 0 upright, 16 hanging down
    cart_position: float = 0.5    # per m^2
    cart_velocity: float = 0.01   # per (m/s)^2
    pole_rate: float = 0.02       # per (rad/s)^2
    force: float = 1e-3           # per N^2
    track_limit: float = 200.0    # per m^2 beyond the track end
    terminal_scale: float = 5.0   # the final state's cost counts this many times


COST = CostWeights()


def stage_cost(x, u, w: CostWeights = COST):
    th = x[..., 1]
    return (w.upright * (1.0 + torch.cos(th)) + w.cart_position * x[..., 0] ** 2 + w.cart_velocity * x[..., 2] ** 2
            + w.pole_rate * x[..., 3] ** 2 + w.force * u**2
            + w.track_limit * torch.relu(x[..., 0].abs() - TRACK_HALF_LENGTH_M) ** 2)


class MPPI:
    def __init__(self, model, K=512, H=50, sigma=4.0, lam=1.0, member="split", seed=0):
        self.model, self.K, self.H, self.sigma, self.lam, self.member = model, K, H, sigma, lam, member
        self.gen = torch.Generator().manual_seed(seed)
        self.U = torch.zeros(H, dtype=model.dtype)

    @torch.no_grad()
    def act(self, x, uh):
        m = self.model
        eps = self.sigma * torch.randn(self.K, self.H, generator=self.gen).to(m.dtype)
        eps[0] = 0.0  # always evaluate the unperturbed plan
        U = torch.clamp(self.U[None] + eps, -U_MAX, U_MAX)
        xb = torch.as_tensor(x, dtype=m.dtype).expand(self.K, 4).clone()
        aux = m.init_aux(torch.as_tensor(uh, dtype=m.dtype).expand(self.K, H_HIST).clone())
        cost = torch.zeros(self.K, dtype=m.dtype)
        member = self.member if hasattr(m, "M") else None
        for t in range(self.H):
            xb, aux = m.step(xb, U[:, t], aux, member)
            xb = xb.to(m.dtype)
            cost = cost + stage_cost(xb, U[:, t])
        cost = cost + COST.terminal_scale * stage_cost(xb, torch.zeros_like(cost))
        # a diverged rollout (NaN or inf) loses every comparison instead of poisoning the softmax
        cost = torch.nan_to_num(cost, nan=1e9, posinf=1e9)
        w = torch.softmax(-(cost - cost.min()) / self.lam, dim=0)
        self.U = torch.clamp(self.U + (w[:, None] * (U - self.U[None])).sum(0), -U_MAX, U_MAX)
        u0 = float(self.U[0])
        self.U = torch.cat([self.U[1:], self.U[-1:]])  # shift for the next step
        return u0


UPRIGHT_TOLERANCE_DEG = 12.0
SUCCESS_WINDOW_S = 2.0  # the episode succeeds if the pole stays upright for its last 2 s


def run_episode(model, seed, T=400, K=512, H=50, log_path=None):
    """Swing-up and balance on the true plant for T control steps. Returns metrics."""
    rng = np.random.default_rng(seed)
    plant = MysteryCartPole(rng=np.random.default_rng(10_000 + seed))
    obs = plant.reset([0.0, rng.normal(0, 0.05), 0.0, 0.0])
    ctrl = MPPI(model, K=K, H=H, seed=seed)
    uh = np.zeros(H_HIST)
    xs, us, times = [obs], [], []
    for _ in range(T):
        t0 = time.perf_counter()
        u = ctrl.act(obs, uh)
        times.append(time.perf_counter() - t0)
        obs = plant.step(u)
        uh = np.concatenate([[u], uh[:-1]])
        xs.append(obs)
        us.append(u)
    xs = np.array(xs)
    err = np.abs(wrap(xs[:, 1] - np.pi))
    tolerance = np.radians(UPRIGHT_TOLERANCE_DEG)
    last = xs[-round(SUCCESS_WINDOW_S / DT):]
    success = bool(np.all(np.abs(wrap(last[:, 1] - np.pi)) < tolerance) and np.all(np.abs(last[:, 0]) < TRACK_HALF_LENGTH_M))
    up = np.where(err < tolerance)[0]
    t_up = float(up[0] * DT) if len(up) else float("nan")
    if log_path:
        np.savetxt(log_path, np.column_stack([np.arange(len(xs)) * DT, xs, np.append(us, np.nan)]), delimiter=",",
                   header="t,p,theta,p_dot,theta_dot,u", comments="")
    return {"success": success, "t_first_upright": t_up, "final_angle_err_deg": float(np.degrees(err[-1])),
            "mean_ms": 1e3 * float(np.mean(times)), "p99_ms": 1e3 * float(np.percentile(times, 99)),
            "max_abs_p": float(np.max(np.abs(xs[:, 0])))}
