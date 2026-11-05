#!/usr/bin/env python3
"""Ablation: does the MLP ensemble fall behind the grey-box model because it only sees the last
two commands? The actuator lag is 100 ms (5 samples), so the hidden force depends on a longer
command history than u_{k-1}, u_{k-2}. Refit the same ensemble with 2, 5 and 10 past commands
and compare open-loop prediction on the test episodes.

    python3 scripts/history_ablation.py        (needs results/data.pkl)
"""
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import lmppi.models as M  # noqa: E402
from lmppi.data import windows  # noqa: E402
from lmppi.plant import wrap  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"


def make_features(n_hist):
    def features(x, u, uh):
        cols = [torch.sin(x[:, 1]), torch.cos(x[:, 1]), x[:, 2], x[:, 3], u] + [uh[:, j] for j in range(n_hist)]
        return torch.stack(cols, -1)
    return features


def main():
    with open(RES / "data.pkl", "rb") as f:
        data = pickle.load(f)
    rows = []
    for n_hist in [2, 5, 10]:
        M.features = make_features(n_hist)  # the ensemble looks the function up at call time
        torch.manual_seed(0)
        m = M.EnsembleModel(M=5, epochs=60, name=f"ens_hist{n_hist}")
        m.fit(data["train"], data["val"], log=lambda *_: None)
        for H in [10, 25, 50]:
            w = windows(data["test"], H)
            X = m.rollout(w["x0"], w["uh"], w["u"]).double().numpy()
            e = wrap(X[:, -1, 1] - w["xt"][:, -1, 1])
            rows.append({"past_commands": n_hist, "horizon_s": H * 0.02, "rmse_theta_deg": float(np.degrees(np.sqrt(np.mean(e**2))))})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "history_ablation.csv", index=False)


if __name__ == "__main__":
    main()
