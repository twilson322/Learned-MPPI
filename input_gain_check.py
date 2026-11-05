#!/usr/bin/env python3
"""Why the Koopman/EDMD model cannot swing the pole up: compare how each model thinks a cart
push changes the pole's angular velocity when the pole hangs and when it stands.

    python3 scripts/input_gain_check.py      (needs results/models.pkl from run_experiments.py)
"""
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
RES = pathlib.Path(__file__).resolve().parents[1] / "results"


def main():
    with open(RES / "models.pkl", "rb") as f:
        models = pickle.load(f)
    lines = ["pose,model,dtheta_dot_rad_s"]
    for th0, pose in [(0.0, "hanging"), (np.pi, "upright")]:
        for name in ["oracle", "greybox", "mlp_ensemble", "koopman_edmd"]:
            x0 = np.array([[0.0, th0, 0.0, 0.0]] * 2)
            U = np.zeros((2, 10))
            U[1] = 5.0  # +5 N for 10 steps (0.2 s) vs no push
            X = models[name].rollout(x0, np.zeros((2, 10)), U).double().numpy()
            d = X[1, -1, 3] - X[0, -1, 3]
            lines.append(f"{pose},{name},{d:.3f}")
            print(f"{pose:8s} {name:14s} change in theta_dot from a 5 N, 0.2 s push: {d:+.3f} rad/s")
    (RES / "input_gain_check.csv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
