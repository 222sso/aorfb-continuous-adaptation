# -*- coding: utf-8 -*-
"""
Train the deployable predictor: LightGBM on Morgan count fingerprints, 3 seeds.

Chosen by `benchmark.py` on scaffold-grouped 5-fold CV, ranked by target-region
F1 (the metric that matches what the predictor is actually used for):

    lgbm_morgan   solv MAE 0.087  lumo MAE 0.226  target F1 0.726  P 0.806
    bilstm        solv MAE 0.117  lumo MAE 0.268  target F1 0.688  P 0.723

Three seeds are trained so the ensemble spread can stand in for the
ensemble-uncertainty column the previous predictor produced. Seeds vary the
bagging and feature-sampling draws; all three see the full dataset, because this
is the deployment fit — the honest generalisation estimate is the CV number
above, not anything measurable from these models.

One asymmetry disappears with this model: the BiLSTM dropped every molecule
longer than MAX_LEN=70 tokens (2.3 % of the hard arm, 0.9 % of the soft arm),
silently removing exactly the long polyacidic molecules most likely to sit in
the target region. Morgan fingerprints have no length limit.

Run:
  python predictor_v2_260811/code/train_final.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARTIFACTS = HERE.parent / "artifacts" / "lgbm_morgan_final"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

from data import load_dataset  # noqa: E402
from features import build_features  # noqa: E402

SEEDS = [42, 43, 44]
N_ROUNDS = 400          # matches the benchmarked configuration
PARAMS = dict(
    objective="regression", metric="l1", learning_rate=0.05,
    num_leaves=127, min_data_in_leaf=20, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbosity=-1, n_jobs=-1,
)


def main():
    import lightgbm as lgb

    df = load_dataset()
    print(f"[data] {len(df):,} molecules / {df.scaffold.nunique()} scaffolds, units=eV", flush=True)
    _, morgan, _, _ = build_features(df["canon"].tolist(), tag="reddb31k")
    X = morgan.astype(np.float32)
    print(f"[features] Morgan counts {X.shape}", flush=True)

    y = {"solv": df["solv"].to_numpy(), "lumo": df["lumo"].to_numpy()}

    t0 = time.time()
    for seed in SEEDS:
        for target in ("solv", "lumo"):
            p = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
            booster = lgb.train(p, lgb.Dataset(X, label=y[target]), num_boost_round=N_ROUNDS)
            out = ARTIFACTS / f"lgbm_{target}_seed{seed}.txt"
            booster.save_model(str(out))
            print(f"  saved {out.name}  ({time.time()-t0:.0f}s cumulative)", flush=True)

    meta = {
        "model": "lightgbm on Morgan count fingerprint",
        "selected_by": "target-region F1 under scaffold-grouped 5-fold CV",
        "cv_reference": {
            "solv_mae": 0.0868, "solv_r2": 0.9085,
            "lumo_mae": 0.2259, "lumo_r2": 0.9032,
            "target_precision": 0.8062, "target_recall": 0.6600, "target_f1": 0.7258,
        },
        "fingerprint": {"type": "morgan_count", "radius": 2, "fpSize": 2048,
                        "countSimulation": False},
        "units": "eV",
        "trained_on": "all 31,140 RedDB molecules (deployment fit)",
        "seeds": SEEDS,
        "n_rounds": N_ROUNDS,
        "params": PARAMS,
        "no_length_limit": True,
    }
    (ARTIFACTS / "MODEL_CARD.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[done] {ARTIFACTS}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
