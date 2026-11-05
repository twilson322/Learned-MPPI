"""The "mystery" cart-pole the controllers have to learn.

Compared with the textbook model the learner is given as a prior, the true plant has
  * a first-order force actuator (time constant 100 ms) whose state is NOT observed,
  * an actuator deadzone of +/-1 N and saturation at +/-10 N,
  * Coulomb (0.8 N) + viscous (1.0 N s/m) friction on the cart and viscous friction at the pivot,
  * a 20 % heavier, 10 % longer pole than the nominal parameters,
  * small Gaussian sensor noise.
It is integrated with RK4 at 1 kHz and sampled at 50 Hz with zero-order-hold commands.
State convention: x = [p, theta, p_dot, theta_dot], theta = 0 hanging down, pi upright.
"""
from dataclasses import dataclass, field

import numpy as np

DT = 0.02          # control / sample period [s]
SUBSTEPS = 20      # 1 kHz integration
U_MAX = 10.0       # actuator saturation [N]
COULOMB_SMOOTHING_M_S = 0.02  # tanh width standing in for sign(v); keeps RK4 well behaved at v = 0


@dataclass
class CartPoleParams:
    mc: float = 1.0
    mp: float = 0.3
    l: float = 0.5
    g: float = 9.81
    b_cart: float = 0.0       # viscous cart friction [N s/m]
    fc_cart: float = 0.0      # Coulomb cart friction [N]
    b_pivot: float = 0.0      # viscous pivot friction [N m s/rad]


NOMINAL = CartPoleParams()
TRUE = CartPoleParams(mp=0.36, l=0.55, b_cart=1.0, fc_cart=0.8, b_pivot=0.004)


def cartpole_deriv(x, force, p: CartPoleParams):
    """Continuous dynamics, vectorized over a leading batch dimension (numpy)."""
    pos, th, v, w = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    s, c = np.sin(th), np.cos(th)
    f = force - p.b_cart * v - p.fc_cart * np.tanh(v / COULOMB_SMOOTHING_M_S)
    tau_pivot = -p.b_pivot * w
    den = p.mc + p.mp * s * s
    pdd = (f + p.mp * s * (p.l * w * w + p.g * c) - c * tau_pivot / p.l) / den
    tdd = (-f * c - p.mp * p.l * w * w * c * s - (p.mc + p.mp) * p.g * s) / (p.l * den) \
        + tau_pivot * (p.mc + p.mp) / (p.mp * p.l * p.l * den)
    return np.stack([v, w, pdd, tdd], axis=-1)


@dataclass
class MysteryCartPole:
    params: CartPoleParams = field(default_factory=lambda: TRUE)
    tau_act: float = 0.10     # actuator time constant [s]
    deadzone: float = 1.0     # [N]
    sensor_std: np.ndarray = field(default_factory=lambda: np.array([1e-3, 2e-3, 1e-2, 2e-2]))
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def reset(self, x0=None):
        self.x = np.zeros(4) if x0 is None else np.array(x0, dtype=float)
        self.force = 0.0  # hidden actuator state
        return self.observe()

    def observe(self):
        return self.x + self.sensor_std * self.rng.standard_normal(4)

    def step(self, u_cmd):
        u = float(np.clip(u_cmd, -U_MAX, U_MAX))
        u = 0.0 if abs(u) < self.deadzone else u - np.sign(u) * self.deadzone
        h = DT / SUBSTEPS
        for _ in range(SUBSTEPS):
            # actuator lag (exact discretization of the first-order filter)
            self.force += (u - self.force) * (1.0 - np.exp(-h / self.tau_act))
            k1 = cartpole_deriv(self.x, self.force, self.params)
            k2 = cartpole_deriv(self.x + 0.5 * h * k1, self.force, self.params)
            k3 = cartpole_deriv(self.x + 0.5 * h * k2, self.force, self.params)
            k4 = cartpole_deriv(self.x + h * k3, self.force, self.params)
            self.x = self.x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.observe()


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
