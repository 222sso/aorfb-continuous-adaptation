# -*- coding: utf-8 -*-
"""
Predictor v2 — model comparison under scaffold-grouped 5-fold CV.

Two families of metric are reported, because the deployed predictor is used for
two different jobs:

  regression  MAE / RMSE / R^2 per target (eV)
  decision    precision / recall / F1 for "is this molecule inside the joint
              target region (ΔG_solv <= -1.5 eV AND E_LUMO <= -4.0 eV)"

The second is what actually determines every target-pass number in the study, and
a model can improve R^2 while getting worse at the boundary. Both are tracked.

Run:
  python predictor_v2_260811/code/benchmark.py            # tree models
  python predictor_v2_260811/code/benchmark.py --with-nn  # + BiLSTM reference
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)

from data import load_dataset, make_scaffold_folds, TARGET_SOLV_MAX_EV, TARGET_LUMO_MAX_EV  # noqa: E402
from features import build_features, assemble, clean_matrix  # noqa: E402


def regression_metrics(y, p):
    e = p - y
    ss_res = float((e ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def decision_metrics(solv_true, lumo_true, solv_pred, lumo_pred):
    gt = (solv_true <= TARGET_SOLV_MAX_EV) & (lumo_true <= TARGET_LUMO_MAX_EV)
    pr = (solv_pred <= TARGET_SOLV_MAX_EV) & (lumo_pred <= TARGET_LUMO_MAX_EV)
    tp = int((gt & pr).sum()); fp = int((~gt & pr).sum()); fn = int((gt & ~pr).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "target_precision": prec,
        "target_recall": rec,
        "target_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


# ----------------------------------------------------------------- models
def fit_ridge(Xtr, ytr, Xte, alpha=1.0):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def fit_lgbm(Xtr, ytr, Xte, Xva=None, yva=None, **kw):
    import lightgbm as lgb
    params = dict(
        objective="regression", metric="l1", learning_rate=0.05,
        num_leaves=127, min_data_in_leaf=20, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
        verbosity=-1, n_jobs=-1, seed=0,
    )
    params.update(kw)
    n_rounds = params.pop("n_rounds", 3000)
    dtr = lgb.Dataset(Xtr, label=ytr)
    cbs, valid = [], None
    if Xva is not None:
        valid = [lgb.Dataset(Xva, label=yva)]
        cbs = [lgb.early_stopping(100, verbose=False)]
    booster = lgb.train(params, dtr, num_boost_round=n_rounds, valid_sets=valid, callbacks=cbs)
    return booster.predict(Xte), booster


def fit_catboost(Xtr, ytr, Xte, **kw):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=kw.get("iterations", 2000), learning_rate=0.05,
                          depth=8, l2_leaf_reg=3.0, loss_function="MAE",
                          random_seed=0, verbose=0, thread_count=-1)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def fit_xgb(Xtr, ytr, Xte, **kw):
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=kw.get("n_estimators", 2000), learning_rate=0.05,
                         max_depth=8, subsample=0.8, colsample_bytree=0.7,
                         reg_lambda=1.0, objective="reg:absoluteerror",
                         tree_method="hist", n_jobs=-1, random_state=0)
    m.fit(Xtr, ytr, verbose=False)
    return m.predict(Xte)


# ----------------------------------------------------------------- driver
def run_config(name, blocks, fitter, df, folds, d2d, morgan, fg, **kw):
    X, _ = assemble(d2d, morgan, fg, blocks=blocks)
    y_solv = df["solv"].to_numpy()
    y_lumo = df["lumo"].to_numpy()

    oof_solv = np.zeros(len(df))
    oof_lumo = np.zeros(len(df))
    t0 = time.time()
    n_fits = folds.n_folds * 2
    done = 0
    for fi, (tr, te) in enumerate(folds.split()):
        (Xtr, Xte), _ = clean_matrix(X[tr], X[te])
        for target, oof in (("solv", oof_solv), ("lumo", oof_lumo)):
            y = y_solv if target == "solv" else y_lumo
            ft = time.time()
            out = fitter(Xtr, y[tr], Xte, **kw)
            oof[te] = out[0] if isinstance(out, tuple) else out
            done += 1
            el = time.time() - t0
            eta = el / done * (n_fits - done)
            print(f"    fold {fi+1}/{folds.n_folds} {target:4s} "
                  f"{time.time()-ft:6.1f}s | elapsed {el/60:5.1f}m | ETA {eta/60:5.1f}m",
                  flush=True)
    dt = time.time() - t0

    row = {"model": name, "blocks": "+".join(blocks), "seconds": round(dt, 1)}
    for tgt, yt, yp in (("solv", y_solv, oof_solv), ("lumo", y_lumo, oof_lumo)):
        for k, v in regression_metrics(yt, yp).items():
            row[f"{tgt}_{k}"] = v
    row.update(decision_metrics(y_solv, y_lumo, oof_solv, oof_lumo))
    return row, oof_solv, oof_lumo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="fewer boosting rounds")
    args = ap.parse_args()

    df = load_dataset()
    folds = make_scaffold_folds(df, args.folds, seed=0)
    print(f"data {len(df):,} molecules / {df.scaffold.nunique()} scaffolds; "
          f"scaffolds per fold {folds.scaffolds_per_fold}")
    d2d, morgan, fg, _ = build_features(df["canon"].tolist(), tag="reddb31k")

    n_rounds = 400 if args.quick else 3000
    configs = [
        ("ridge_morgan",   ("morgan",),                    lambda *a, **k: fit_ridge(*a, alpha=1.0), {}),
        ("ridge_all",      ("rdkit2d", "morgan", "fg"),    lambda *a, **k: fit_ridge(*a, alpha=10.0), {}),
        ("lgbm_rdkit2d",   ("rdkit2d", "fg"),              fit_lgbm, {"n_rounds": n_rounds}),
        ("lgbm_morgan",    ("morgan",),                    fit_lgbm, {"n_rounds": n_rounds}),
        ("lgbm_all",       ("rdkit2d", "morgan", "fg"),    fit_lgbm, {"n_rounds": n_rounds}),
        ("xgb_all",        ("rdkit2d", "morgan", "fg"),    fit_xgb,  {"n_estimators": n_rounds}),
        ("catboost_all",   ("rdkit2d", "morgan", "fg"),    fit_catboost, {"iterations": n_rounds}),
    ]

    # ---- resume: keep configs already completed in a previous run ----
    rows, oofs = [], {}
    csv_path = RESULTS / "benchmark_scaffold_cv.csv"
    npz_path = RESULTS / "oof_predictions.npz"
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        rows = prev.to_dict("records")
        done_names = set(prev["model"])
        if npz_path.exists():
            z = np.load(npz_path)
            for nm in done_names:
                if f"{nm}__solv" in z:
                    oofs[nm] = (z[f"{nm}__solv"], z[f"{nm}__lumo"])
        print(f"[resume] {len(done_names)} config(s) already done: {sorted(done_names)}", flush=True)
    else:
        done_names = set()

    for name, blocks, fitter, kw in configs:
        if name in done_names:
            print(f"\n[{name}] already complete — skipped", flush=True)
            continue
        print(f"\n[{name}] blocks={blocks} ...", flush=True)
        try:
            row, os_, ol_ = run_config(name, blocks, fitter, df, folds, d2d, morgan, fg, **kw)
        except Exception as e:
            print(f"  FAILED: {e!r}")
            continue
        rows.append(row)
        oofs[name] = (os_, ol_)
        print(f"  solv MAE {row['solv_mae']:.4f} R2 {row['solv_r2']:.4f} | "
              f"lumo MAE {row['lumo_mae']:.4f} R2 {row['lumo_r2']:.4f} | "
              f"target P {row['target_precision']:.3f} R {row['target_recall']:.3f} "
              f"F1 {row['target_f1']:.3f}  ({row['seconds']}s)", flush=True)

        # checkpoint after EVERY config: a long sweep must never be all-or-nothing
        pd.DataFrame(rows).to_csv(RESULTS / "benchmark_scaffold_cv.csv", index=False)
        np.savez_compressed(
            RESULTS / "oof_predictions.npz",
            **{f"{k}__{t}": v[i] for k, v in oofs.items() for i, t in enumerate(("solv", "lumo"))},
            y_solv=df["solv"].to_numpy(), y_lumo=df["lumo"].to_numpy(), fold=folds.fold_of_row,
        )
        print(f"  [checkpoint] {len(rows)} model(s) saved", flush=True)

    out = pd.DataFrame(rows).sort_values("target_f1", ascending=False)
    out.to_csv(RESULTS / "benchmark_scaffold_cv.csv", index=False)
    print("\n=== ranked by target-region F1 ===")
    cols = ["model", "blocks", "solv_mae", "solv_r2", "lumo_mae", "lumo_r2",
            "target_precision", "target_recall", "target_f1", "seconds"]
    print(out[cols].round(4).to_string(index=False))
    print(f"\n[saved] {RESULTS/'benchmark_scaffold_cv.csv'}")


if __name__ == "__main__":
    main()
