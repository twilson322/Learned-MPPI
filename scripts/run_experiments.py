#!/usr/bin/env python3
"""End-to-end pipeline: collect data, fit six model classes, score open-loop prediction,
then run MPPI swing-up episodes on the true plant with each model.

    python3 scripts/run_experiments.py --stage all          # ~35 min on 2 CPU cores
    python3 scripts/run_experiments.py --stage control --seeds 3
"""
import argparse
import json
import pathlib
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lmppi.data import collect, to_transitions, windows  # noqa: E402
from lmppi.models import (EnsembleModel, KoopmanModel, RFFGPModel, greybox_model, nominal_model,  # noqa: E402
                          oracle_model, _delta)
from lmppi.mppi import run_episode  # noqa: E402
from lmppi.plant import wrap  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
torch.manual_seed(0)


def log(msg):
    print(msg, flush=True)
    with open(RES / "log.txt", "a") as f:
        f.write(msg + "\n")


def stage_data():
    t0 = time.time()
    train = collect(100, 300, seed=1)
    val = collect(25, 300, seed=2)
    test = collect(25, 300, seed=3)
    with open(RES / "data.pkl", "wb") as f:
        pickle.dump({"train": train, "val": val, "test": test}, f)
    n = [sum(len(e["u"]) for e in d) for d in (train, val, test)]
    log(f"data: {n[0]} train / {n[1]} val / {n[2]} test transitions in {time.time() - t0:.0f} s")


def build_models(data):
    nom = nominal_model()
    models = {
        "oracle": oracle_model(),
        "nominal": nom,
        "greybox": greybox_model(),
        "mlp_ensemble": EnsembleModel(M=5, epochs=60),
        "hybrid_residual": EnsembleModel(M=5, epochs=60, name="hybrid_residual", residual_of=nominal_model()),
        "rff_gp": RFFGPModel(D=1500),
        "koopman_edmd": KoopmanModel(),
    }
    stats = {}
    for name, m in models.items():
        log(f"fitting {name}")
        stats[name] = m.fit(data["train"], data["val"], log=log)
    return models, stats


def stage_fit():
    with open(RES / "data.pkl", "rb") as f:
        data = pickle.load(f)
    models, stats = build_models(data)
    with open(RES / "models.pkl", "wb") as f:
        pickle.dump(models, f)
    with open(RES / "fit_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=float)


def stage_predict():
    with open(RES / "data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(RES / "models.pkl", "rb") as f:
        models = pickle.load(f)
    rows = []
    for H in [1, 10, 25, 50]:
        w = windows(data["test"], H)
        for name, m in models.items():
            X = m.rollout(w["x0"], w["uh"], w["u"]).double().numpy()
            e = X - w["xt"]
            e[..., 1] = wrap(e[..., 1])
            rows.append({"model": name, "horizon_steps": H, "horizon_s": H * 0.02, "n_windows": len(e),
                         "rmse_p_cm": 100 * np.sqrt(np.mean(e[:, -1, 0] ** 2)),
                         "rmse_theta_deg": np.degrees(np.sqrt(np.mean(e[:, -1, 1] ** 2))),
                         "median_theta_err_deg": np.degrees(np.median(np.abs(e[:, -1, 1]))),
                         "rmse_theta_dot": np.sqrt(np.mean(e[:, -1, 3] ** 2))})
            log(f"predict H={H:2d} {name:16s} theta RMSE {rows[-1]['rmse_theta_deg']:8.3f} deg")
    pd.DataFrame(rows).to_csv(RES / "prediction.csv", index=False)

    # Calibration of the one-step 95 % intervals for the two models that report uncertainty.
    tr = to_transitions(data["test"])
    T = {k: torch.tensor(v, dtype=torch.float32) for k, v in tr.items()}
    d = _delta(T["xn"], T["x"])
    cal = []
    for name in ["mlp_ensemble", "rff_gp"]:
        mu, var = models[name].predict(T["x"], T["u"], T["uh"])
        inside = (torch.abs(d - mu) <= 1.96 * torch.sqrt(var)).float().mean(0)
        cal.append({"model": name, **{f"coverage_{k}": float(inside[i]) for i, k in enumerate(["p", "theta", "p_dot", "theta_dot"])}})
    pd.DataFrame(cal).to_csv(RES / "calibration.csv", index=False)
    log(f"calibration (target 0.95): {cal}")


def stage_control(seeds):
    with open(RES / "models.pkl", "rb") as f:
        models = pickle.load(f)
    rows = []
    for name, m in models.items():
        for s in range(seeds):
            t0 = time.time()
            r = run_episode(m, seed=s, T=400, K=512, H=40, log_path=str(RES / f"episode_{name}_{s}.csv") if s == 0 else None)
            rows.append({"model": name, "seed": s, **r})
            log(f"control {name:16s} seed {s} success={r['success']} t_up={r['t_first_upright']:.2f} "
                f"final_err={r['final_angle_err_deg']:.2f} deg  {r['mean_ms']:.1f} ms/step  ({time.time() - t0:.0f} s)")
            pd.DataFrame(rows).to_csv(RES / "control.csv", index=False)


def stage_round2(seeds):
    """One round of data aggregation (as in PETS / DAgger): add the closed-loop episodes the
    ensemble produced to the training set, retrain, and run the controller again."""
    with open(RES / "data.pkl", "rb") as f:
        data = pickle.load(f)
    extra = []
    for s in range(seeds):
        f = RES / f"r2_src_{s}.csv"
        with open(RES / "models.pkl", "rb") as fh:
            m = pickle.load(fh)["mlp_ensemble"]
        run_episode(m, seed=100 + s, T=400, K=512, H=40, log_path=str(f))
        a = np.loadtxt(f, delimiter=",", skiprows=1)
        extra.append({"x": a[:, 1:5], "u": a[:-1, 5]})
        f.unlink()
    m2 = EnsembleModel(M=5, epochs=60, name="mlp_ensemble_round2")
    m2.fit(data["train"] + extra, data["val"], log=log)
    rows = []
    for s in range(seeds):
        r = run_episode(m2, seed=s, T=400, K=512, H=40, log_path=str(RES / f"episode_mlp_ensemble_round2_{s}.csv") if s == 0 else None)
        rows.append({"model": "mlp_ensemble_round2", "seed": s, **r})
        log(f"control mlp_ensemble_round2 seed {s} success={r['success']} final_err={r['final_angle_err_deg']:.2f}")
    pd.DataFrame(rows).to_csv(RES / "control_round2.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["data", "fit", "predict", "control", "round2", "all"])
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    if a.stage in ("data", "all"):
        stage_data()
    if a.stage in ("fit", "all"):
        stage_fit()
    if a.stage in ("predict", "all"):
        stage_predict()
    if a.stage in ("control", "all"):
        stage_control(a.seeds)
    if a.stage in ("round2", "all"):
        stage_round2(a.seeds)


if __name__ == "__main__":
    main()
