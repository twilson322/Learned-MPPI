"""Six dynamics models behind one interface.

    init_aux(uh)                  -> aux   (hidden context built from the command history)
    step(x, u, aux, member=None)  -> (x_next, aux)
    rollout(x0, uh, U)            -> X (B, H+1, 4)

Everything is batched torch (float64 for physics, float32 for networks) so the same objects
serve multi-step evaluation and the MPPI controller.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn

from .data import H_HIST, windows
from .plant import COULOMB_SMOOTHING_M_S, DT, NOMINAL, TRUE, CartPoleParams



def soft_hinge(z, eps=0.02):
    return 0.5 * (z + torch.sqrt(z * z + eps * eps))


def deadzone(u, d, eps=0.02):
    """Smooth, monotone version of u - sign(u) d for |u| > d, 0 otherwise."""
    return soft_hinge(u - d, eps) - soft_hinge(-u - d, eps)


def cartpole_deriv_t(x, f, mc, mp, l, g, b_cart, fc_cart, b_pivot):
    th, v, w = x[..., 1], x[..., 2], x[..., 3]
    s, c = torch.sin(th), torch.cos(th)
    fn = f - b_cart * v - fc_cart * torch.tanh(v / COULOMB_SMOOTHING_M_S)  # same smoothing as the plant
    tp = -b_pivot * w
    den = mc + mp * s * s
    pdd = (fn + mp * s * (l * w * w + g * c) - c * tp / l) / den
    tdd = (-fn * c - mp * l * w * w * c * s - (mc + mp) * g * s) / (l * den) + tp * (mc + mp) / (mp * l * l * den)
    return torch.stack([v, w, pdd, tdd], dim=-1)


class DynModel:
    name = "base"

    def init_aux(self, uh):
        return uh

    def step(self, x, u, aux, member=None):
        raise NotImplementedError

    @torch.no_grad()
    def rollout(self, x0, uh, U, member=None):
        x = torch.as_tensor(x0, dtype=self.dtype)
        aux = self.init_aux(torch.as_tensor(uh, dtype=self.dtype))
        U = torch.as_tensor(U, dtype=self.dtype)
        xs = [x]
        for t in range(U.shape[1]):
            x, aux = self.step(x, U[:, t], aux, member)
            xs.append(x)
        return torch.stack(xs, dim=1)

    def fit(self, train_eps, val_eps, log=print):
        return {}


# ----------------------------------------------------------------------------- physics models
class PhysicsModel(DynModel):
    """Cart-pole with actuator lag + deadzone + friction. Parameters can be fixed (nominal,
    oracle) or identified by gradient descent through the simulator (grey box)."""

    dtype = torch.float64

    def __init__(self, name, p: CartPoleParams, tau_act=1e-4, dz=0.0, trainable=False, substeps=2):
        self.name = name
        self.substeps = substeps
        self.mc = p.mc
        self.g = p.g
        init = [p.mp, p.l, max(p.b_cart, 1e-3), max(p.fc_cart, 1e-3), max(p.b_pivot, 1e-5), tau_act, max(dz, 1e-3)]
        self.logp = torch.tensor(np.log(init), dtype=torch.float64, requires_grad=trainable)
        self.trainable = trainable

    def params(self):
        mp, l, b, fc, bp, tau, dz = torch.exp(self.logp)
        return mp, l, b, fc, bp, tau, dz

    def init_aux(self, uh):
        # Rebuild the hidden actuator force by running the (identified) filter over the history.
        *_, tau, dz = self.params()
        f = torch.zeros(uh.shape[0], dtype=torch.float64)
        a = torch.exp(-DT / tau)
        for j in range(uh.shape[1] - 1, -1, -1):  # oldest first
            f = a * f + (1 - a) * deadzone(torch.clamp(uh[:, j].double(), -10, 10), dz)
        return f

    def step(self, x, u, aux, member=None):
        mp, l, b, fc, bp, tau, dz = self.params()
        h = DT / self.substeps
        ue = deadzone(torch.clamp(u.double(), -10, 10), dz)
        f = aux
        a = torch.exp(-h / tau)
        x = x.double()
        for _ in range(self.substeps):
            f = a * f + (1 - a) * ue
            k1 = cartpole_deriv_t(x, f, self.mc, mp, l, self.g, b, fc, bp)
            k2 = cartpole_deriv_t(x + 0.5 * h * k1, f, self.mc, mp, l, self.g, b, fc, bp)
            k3 = cartpole_deriv_t(x + 0.5 * h * k2, f, self.mc, mp, l, self.g, b, fc, bp)
            k4 = cartpole_deriv_t(x + h * k3, f, self.mc, mp, l, self.g, b, fc, bp)
            x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x, f

    def fit(self, train_eps, val_eps, log=print, iters=400, H=10, batch=256, seed=0):
        """Multi-step (H = 10, 0.2 s) simulation-error minimization through the simulator."""
        if not self.trainable:
            return {}
        t0 = time.time()
        w = windows(train_eps, H)
        x0 = torch.tensor(w["x0"])
        uh = torch.tensor(w["uh"])
        U = torch.tensor(w["u"])
        XT = torch.tensor(w["xt"])
        scale = torch.tensor([0.1, 0.1, 0.5, 1.0], dtype=torch.float64)
        opt = torch.optim.Adam([self.logp], lr=0.03)
        gen = torch.Generator().manual_seed(seed)
        for it in range(iters):
            idx = torch.randint(0, x0.shape[0], (batch,), generator=gen)
            x = x0[idx]
            aux = self.init_aux(uh[idx])
            loss = 0.0
            for t in range(H):
                x, aux = self.step(x, U[idx, t], aux)
                e = x - XT[idx, t + 1]
                e = torch.stack([e[:, 0], torch.atan2(torch.sin(e[:, 1]), torch.cos(e[:, 1])), e[:, 2], e[:, 3]], -1)
                loss = loss + ((e / scale) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 100 == 0 or it == iters - 1:
                names = ["mp", "l", "b_cart", "fc_cart", "b_pivot", "tau_act", "deadzone"]
                vals = {n: float(v) for n, v in zip(names, torch.exp(self.logp).detach())}
                log(f"  [{self.name}] iter {it:4d} loss {loss.item() / H:.4f} " + " ".join(f"{k}={v:.3f}" for k, v in vals.items()))
        self.logp.requires_grad_(False)
        names = ["mp", "l", "b_cart", "fc_cart", "b_pivot", "tau_act", "deadzone"]
        return {"fit_s": time.time() - t0, **{n: float(v) for n, v in zip(names, torch.exp(self.logp))}}


def nominal_model():
    return PhysicsModel("nominal", NOMINAL, tau_act=1e-4, dz=0.0)


def oracle_model():
    m = PhysicsModel("oracle", TRUE, tau_act=0.10, dz=1.0)
    return m


def greybox_model():
    return PhysicsModel("greybox", CartPoleParams(mp=0.3, l=0.5, b_cart=0.1, fc_cart=0.1, b_pivot=1e-3), tau_act=0.03,
                        dz=0.1, trainable=True)


# ----------------------------------------------------------------------------- learned models
def features(x, u, uh):
    """Translation-invariant inputs: [sin th, cos th, p_dot, th_dot, u_k, u_{k-1}, u_{k-2}]."""
    return torch.stack([torch.sin(x[:, 1]), torch.cos(x[:, 1]), x[:, 2], x[:, 3], u, uh[:, 0], uh[:, 1]], -1)


class Normalizer:
    def __init__(self, a):
        self.mu = a.mean(0)
        self.sd = a.std(0) + 1e-6


class BatchedLinear(nn.Module):
    """M independent linear layers evaluated with one batched matmul."""

    def __init__(self, M, i, o):
        super().__init__()
        self.W = nn.Parameter(torch.randn(M, i, o) * math.sqrt(2.0 / i))
        self.b = nn.Parameter(torch.zeros(M, 1, o))

    def forward(self, x):  # x (M, B, i)
        return torch.baddbmm(self.b, x, self.W)


class EnsembleNet(nn.Module):
    def __init__(self, M, i, o, h=128):
        super().__init__()
        self.l1, self.l2, self.l3 = BatchedLinear(M, i, h), BatchedLinear(M, h, h), BatchedLinear(M, h, 2 * o)
        self.o = o
        self.max_lv = nn.Parameter(torch.full((1, 1, o), 0.5))
        self.min_lv = nn.Parameter(torch.full((1, 1, o), -10.0))

    def forward(self, x):
        h = torch.nn.functional.silu(self.l1(x))
        h = torch.nn.functional.silu(self.l2(h))
        out = self.l3(h)
        mean, lv = out[..., : self.o], out[..., self.o :]
        # Soft-bounded log variance (Chua et al. 2018) so NLL cannot collapse the variance.
        lv = self.max_lv - torch.nn.functional.softplus(self.max_lv - lv)
        lv = self.min_lv + torch.nn.functional.softplus(lv - self.min_lv)
        return mean, lv


def _delta(xn, x):
    d = xn - x
    d[:, 1] = torch.atan2(torch.sin(d[:, 1]), torch.cos(d[:, 1]))
    return d


class EnsembleModel(DynModel):
    """Probabilistic MLP ensemble (PETS-style): each member predicts mean and variance of the
    normalized state change; trained with Gaussian NLL on bootstrapped data."""

    dtype = torch.float32

    def __init__(self, M=5, epochs=60, name="mlp_ensemble", residual_of=None, probabilistic=True):
        self.M, self.epochs, self.name = M, epochs, name
        self.base = residual_of  # physics model for the residual (hybrid) variant
        self.probabilistic = probabilistic

    def _targets(self, x, u, uh, xn):
        d = _delta(xn, x)
        if self.base is not None:
            with torch.no_grad():
                f = self.base.init_aux(uh.double())
                xb, _ = self.base.step(x.double(), u.double(), f)
            d = d - _delta(xb.float(), x)
        return d

    def fit(self, train_eps, val_eps, log=print, seed=0):
        from .data import to_transitions

        torch.manual_seed(seed)
        t0 = time.time()
        tr, va = to_transitions(train_eps), to_transitions(val_eps)
        T = {k: torch.tensor(v, dtype=torch.float32) for k, v in tr.items()}
        V = {k: torch.tensor(v, dtype=torch.float32) for k, v in va.items()}
        Xi = features(T["x"], T["u"], T["uh"])
        Yt = self._targets(T["x"], T["u"], T["uh"], T["xn"])
        self.nin, self.nout = Normalizer(Xi), Normalizer(Yt)
        Xv = (features(V["x"], V["u"], V["uh"]) - self.nin.mu) / self.nin.sd
        Yv = (self._targets(V["x"], V["u"], V["uh"], V["xn"]) - self.nout.mu) / self.nout.sd
        Xn, Yn = (Xi - self.nin.mu) / self.nin.sd, (Yt - self.nout.mu) / self.nout.sd
        self.net = EnsembleNet(self.M, Xn.shape[1], 4)
        opt = torch.optim.AdamW(self.net.parameters(), lr=2e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        n = Xn.shape[0]
        boot = torch.randint(0, n, (self.M, n))  # bootstrap per member
        best, best_state = float("inf"), None
        for ep in range(self.epochs):
            perm = boot[:, torch.randperm(n)]
            for i in range(0, n, 256):
                idx = perm[:, i : i + 256]
                xb, yb = Xn[idx], Yn[idx]  # (M, b, .)
                mean, lv = self.net(xb)
                if self.probabilistic:
                    loss = (((mean - yb) ** 2) * torch.exp(-lv) + lv).mean()
                    loss = loss + 0.01 * (self.net.max_lv.sum() - self.net.min_lv.sum())
                else:
                    loss = ((mean - yb) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            with torch.no_grad():
                mv, _ = self.net(Xv.expand(self.M, -1, -1))
                vmse = ((mv.mean(0) - Yv) ** 2).mean().item()
            if vmse < best:
                best, best_state = vmse, {k: v.clone() for k, v in self.net.state_dict().items()}
            if ep % 20 == 0 or ep == self.epochs - 1:
                log(f"  [{self.name}] epoch {ep:3d} train {loss.item():.4f} val mse (normalized) {vmse:.5f}")
        self.net.load_state_dict(best_state)
        return {"fit_s": time.time() - t0, "val_mse_norm": best}

    def step(self, x, u, aux, member=None):
        """member=None: ensemble mean. member='split': sample b uses member b mod M."""
        x = x.float()
        u = u.float()
        z = (features(x, u, aux) - self.nin.mu) / self.nin.sd
        with torch.no_grad():
            if member == "split":
                B = z.shape[0]
                pad = (-B) % self.M
                zz = torch.cat([z, z[:pad]]) if pad else z
                mean, _ = self.net(zz.view(-1, self.M, z.shape[1]).transpose(0, 1))
                d = mean.transpose(0, 1).reshape(-1, 4)[:B]
            else:
                mean, _ = self.net(z.expand(self.M, -1, -1))
                d = mean.mean(0)
        d = d * self.nout.sd + self.nout.mu
        if self.base is not None:
            f = self.base.init_aux(aux.double())
            xb, _ = self.base.step(x.double(), u.double(), f)
            d = d + _delta(xb.float(), x)
        xn = x + d
        return xn, torch.cat([u[:, None], aux[:, :-1]], 1)

    @torch.no_grad()
    def predict(self, x, u, uh):
        """One-step mean and total (aleatoric + epistemic) variance of the state change."""
        z = (features(x.float(), u.float(), uh.float()) - self.nin.mu) / self.nin.sd
        mean, lv = self.net(z.expand(self.M, -1, -1))
        mu = mean.mean(0) * self.nout.sd + self.nout.mu
        var = (torch.exp(lv).mean(0) + mean.var(0)) * self.nout.sd**2
        return mu, var

    @torch.no_grad()
    def epistemic_std(self, x, u, aux):
        z = (features(x.float(), u.float(), aux.float()) - self.nin.mu) / self.nin.sd
        mean, _ = self.net(z.expand(self.M, -1, -1))
        return (mean.std(0) * self.nout.sd).norm(dim=-1)


class RFFGPModel(DynModel):
    """GP regression with an RBF kernel, approximated by D random Fourier features
    (Rahimi & Recht 2007), i.e. Bayesian linear regression on cos(W z + b). One shared
    feature map, an independent output per state dimension. Lengthscale picked on validation."""

    dtype = torch.float32
    name = "rff_gp"

    def __init__(self, D=1500):
        self.D = D

    def _phi(self, z):
        return math.sqrt(2.0 / self.D) * torch.cos(z @ self.W + self.b)

    def fit(self, train_eps, val_eps, log=print, seed=0):
        from .data import to_transitions

        t0 = time.time()
        g = torch.Generator().manual_seed(seed)
        tr, va = to_transitions(train_eps), to_transitions(val_eps)
        T = {k: torch.tensor(v, dtype=torch.float32) for k, v in tr.items()}
        V = {k: torch.tensor(v, dtype=torch.float32) for k, v in va.items()}
        Xi = features(T["x"], T["u"], T["uh"])
        Yt = _delta(T["xn"], T["x"])
        self.nin, self.nout = Normalizer(Xi), Normalizer(Yt)
        Z, Y = (Xi - self.nin.mu) / self.nin.sd, (Yt - self.nout.mu) / self.nout.sd
        Zv = (features(V["x"], V["u"], V["uh"]) - self.nin.mu) / self.nin.sd
        Yv = (_delta(V["xn"], V["x"]) - self.nout.mu) / self.nout.sd
        best = None
        for ell in [0.5, 1.0, 2.0, 4.0]:
            for noise in [1e-3, 1e-2]:
                W = torch.randn(Z.shape[1], self.D, generator=g) / ell
                b = 2 * math.pi * torch.rand(self.D, generator=g)
                self.W, self.b = W, b
                P = self._phi(Z).double()
                A = P.T @ P + noise * torch.eye(self.D, dtype=torch.float64)
                L = torch.linalg.cholesky(A)
                wts = torch.cholesky_solve(P.T @ Y.double(), L)
                pred = (self._phi(Zv).double() @ wts).float()
                mse = ((pred - Yv) ** 2).mean().item()
                log(f"  [rff_gp] lengthscale {ell:.1f} noise {noise:.0e} val mse (normalized) {mse:.5f}")
                if best is None or mse < best[0]:
                    best = (mse, W, b, wts.float(), L, ell, noise)
        mse, self.W, self.b, self.wts, self.L, ell, noise = best
        # Noise variance per output from the training residuals (normalized units).
        self.sig2 = ((self._phi(Z) @ self.wts - Y) ** 2).mean(0)
        return {"fit_s": time.time() - t0, "val_mse_norm": mse, "lengthscale": ell, "noise": noise}

    @torch.no_grad()
    def predict(self, x, u, uh):
        """One-step mean and variance of the state change (GP posterior with the RFF kernel)."""
        z = (features(x.float(), u.float(), uh.float()) - self.nin.mu) / self.nin.sd
        P = self._phi(z).double()
        mean = (P @ self.wts.double()).float() * self.nout.sd + self.nout.mu
        quad = (P * torch.cholesky_solve(P.T, self.L).T).sum(-1).float()
        var = self.sig2[None] * (1.0 + quad[:, None]) * self.nout.sd**2
        return mean, var

    def step(self, x, u, aux, member=None):
        z = (features(x.float(), u.float(), aux.float()) - self.nin.mu) / self.nin.sd
        d = (self._phi(z) @ self.wts) * self.nout.sd + self.nout.mu
        return x.float() + d, torch.cat([u.float()[:, None], aux[:, :-1].float()], 1)


class KoopmanModel(DynModel):
    """Linear predictor in a lifted space (EDMD with control, Korda & Mezic 2018):
    z+ = A z + B [u_k, u_{k-1}, u_{k-2}],  x = decode(z). The state is lifted once at the start
    of a rollout and then propagated linearly, which is what makes it usable in a linear MPC."""

    dtype = torch.float64
    name = "koopman_edmd"

    def __init__(self, n_rbf=80, ridge=1e-6):
        self.n_rbf, self.ridge = n_rbf, ridge

    def lift(self, x):
        s, c, v, w = torch.sin(x[:, 1]), torch.cos(x[:, 1]), x[:, 2], x[:, 3]
        base = torch.stack([x[:, 0], s, c, v, w, s * w, c * w, s * w * w, c * w * w, s * c, torch.ones_like(s)], -1)
        q = torch.stack([s, c, v / 2.0, w / 5.0], -1)
        rbf = torch.exp(-((q[:, None, :] - self.centers[None]) ** 2).sum(-1) / (2 * 0.5**2))
        return torch.cat([base, rbf], -1)

    def fit(self, train_eps, val_eps, log=print, seed=0):
        from .data import to_transitions

        t0 = time.time()
        g = torch.Generator().manual_seed(seed)
        tr = to_transitions(train_eps)
        X = torch.tensor(tr["x"])
        idx = torch.randint(0, X.shape[0], (self.n_rbf,), generator=g)
        self.centers = torch.stack([torch.sin(X[idx, 1]), torch.cos(X[idx, 1]), X[idx, 2] / 2.0, X[idx, 3] / 5.0], -1)
        Z = self.lift(X)
        Zn = self.lift(torch.tensor(tr["xn"]))
        Uc = torch.stack([torch.tensor(tr["u"]), torch.tensor(tr["uh"][:, 0]), torch.tensor(tr["uh"][:, 1])], -1)
        G = torch.cat([Z, Uc], -1)
        sol = torch.linalg.solve(G.T @ G + self.ridge * torch.eye(G.shape[1], dtype=torch.float64), G.T @ Zn)
        self.A, self.B = sol[: Z.shape[1]].T, sol[Z.shape[1] :].T
        rho = torch.linalg.eigvals(self.A).abs().max().item()
        log(f"  [koopman] lifted dim {Z.shape[1]}, spectral radius of A = {rho:.4f}")
        return {"fit_s": time.time() - t0, "lift_dim": Z.shape[1], "spectral_radius": rho}

    def init_aux(self, uh):
        return {"uh": uh.double(), "z": None}

    def step(self, x, u, aux, member=None):
        z = aux["z"] if aux["z"] is not None else self.lift(x.double())
        uc = torch.stack([u.double(), aux["uh"][:, 0], aux["uh"][:, 1]], -1)
        zn = z @ self.A.T + uc @ self.B.T
        # decode: position and velocities are lifted coordinates, the angle comes from (sin, cos)
        th = torch.atan2(zn[:, 1], zn[:, 2])
        th = x[:, 1].double() + torch.atan2(torch.sin(th - x[:, 1]), torch.cos(th - x[:, 1]))  # unwrap
        xn = torch.stack([zn[:, 0], th, zn[:, 3], zn[:, 4]], -1)
        return xn, {"uh": torch.cat([u.double()[:, None], aux["uh"][:, :-1]], 1), "z": zn}
