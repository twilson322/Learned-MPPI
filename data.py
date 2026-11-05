"""Excitation design and dataset handling.

Every transition carries the same information for every model class:
    x_k (4,), u_k, a command history (u_{k-1} ... u_{k-H_HIST}), and x_{k+1}.
The actuator state is hidden, so a model can only infer it from the command history.
"""
import numpy as np

from .plant import DT, U_MAX, MysteryCartPole

H_HIST = 10


def excitation(n, rng):
    """Band-limited random force: an Ornstein-Uhlenbeck process with random amplitude and
    occasional step holds, so the data covers both slow pushes and fast reversals."""
    amp = rng.uniform(2.0, U_MAX)
    tau = rng.uniform(0.05, 0.6)
    u = np.zeros(n)
    a = np.exp(-DT / tau)
    for k in range(1, n):
        u[k] = a * u[k - 1] + np.sqrt(1 - a * a) * amp * rng.standard_normal()
        if rng.random() < 0.02:  # step hold
            u[k] = rng.uniform(-U_MAX, U_MAX)
    return np.clip(u, -U_MAX, U_MAX)


def collect(n_episodes, steps, seed, upright_frac=0.4):
    """Returns a list of episodes, each a dict with x (steps+1, 4) and u (steps,)."""
    rng = np.random.default_rng(seed)
    eps = []
    for i in range(n_episodes):
        plant = MysteryCartPole(rng=np.random.default_rng(seed * 1000 + i))
        if rng.random() < upright_frac:  # start near the top so the data covers it
            x0 = [rng.uniform(-0.5, 0.5), np.pi + rng.uniform(-0.6, 0.6), rng.uniform(-1, 1), rng.uniform(-2, 2)]
        else:
            x0 = [rng.uniform(-0.5, 0.5), rng.uniform(-1.0, 1.0), rng.uniform(-1, 1), rng.uniform(-3, 3)]
        obs = [plant.reset(x0)]
        u = excitation(steps, rng)
        for k in range(steps):
            # A weak PD "track keeper" on the cart keeps episodes on the 6 m track without
            # making the excitation deterministic (closed-loop excitation).
            u[k] = np.clip(u[k] - 2.0 * obs[-1][0] - 1.5 * obs[-1][2], -U_MAX, U_MAX)
            obs.append(plant.step(u[k]))
            if abs(plant.x[0]) > 3.0:  # off the end of the track: stop the episode
                u = u[: k + 1]
                break
        eps.append({"x": np.array(obs), "u": u})
    return eps


def to_transitions(episodes):
    X, U, Uh, Xn = [], [], [], []
    for e in episodes:
        x, u = e["x"], e["u"]
        upad = np.concatenate([np.zeros(H_HIST), u])
        for k in range(len(u)):
            X.append(x[k])
            U.append(u[k])
            Uh.append(upad[k : k + H_HIST][::-1])  # u_{k-1}, u_{k-2}, ...
            Xn.append(x[k + 1])
    return {"x": np.array(X), "u": np.array(U), "uh": np.array(Uh), "xn": np.array(Xn)}


def windows(episodes, H):
    """Multi-step windows for open-loop evaluation and grey-box training:
    x0 (N,4), uhist (N,H_HIST), U (N,H), Xtrue (N,H+1,4)."""
    X0, UH, U, XT = [], [], [], []
    for e in episodes:
        x, u = e["x"], e["u"]
        upad = np.concatenate([np.zeros(H_HIST), u])
        for k in range(0, len(u) - H, max(1, H // 2)):
            X0.append(x[k])
            UH.append(upad[k : k + H_HIST][::-1])
            U.append(u[k : k + H])
            XT.append(x[k : k + H + 1])
    return {"x0": np.array(X0), "uh": np.array(UH), "u": np.array(U), "xt": np.array(XT)}
