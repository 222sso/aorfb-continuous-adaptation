# -*- coding: utf-8 -*-
"""
BiLSTM under the SAME scaffold-grouped 5-fold CV the tree models use.

Without this the comparison is unfair in both directions:

  * `restore_bilstm.py` evaluates on the legacy single 80/10/10 scaffold split,
    which sorts scaffolds by size and fills train first — leaving the model with
    only ~32 training scaffolds and putting ~55 small scaffolds in test.
  * `benchmark.py` evaluates the tree models on balanced 5-fold groups with
    ~80 training scaffolds per fit.

The second protocol is easier. Any comparison across the two would credit the
tree models for the protocol, not the model. This script runs the identical
architecture (the one `ablation/predict_generated.py` reconstructs) through the
identical folds, and writes a row in the same schema as benchmark.py so the two
tables can be concatenated.

Run:
  python predictor_v2_260811/code/bilstm_cv.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)

from data import load_dataset, make_scaffold_folds  # noqa: E402
from benchmark import regression_metrics, decision_metrics  # noqa: E402
from restore_bilstm import encode_special_pred, build_vocab, vectorize, build_model, PAD_TOKEN, UNK_TOKEN  # noqa: E402

MAX_LEN = 70
EPOCHS = 60
BATCH = 256
LR = 1e-3
DROPOUT = 0.2
VALID_FRAC_OF_TRAIN = 0.1


def main():
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    df = load_dataset()
    df["tok"] = [encode_special_pred(s) for s in df["canon"]]
    folds = make_scaffold_folds(df, 5, seed=0)          # identical folds to benchmark.py
    print(f"data {len(df):,} molecules / {df.scaffold.nunique()} scaffolds; "
          f"scaffolds per fold {folds.scaffolds_per_fold}", flush=True)

    y_solv = df["solv"].to_numpy()
    y_lumo = df["lumo"].to_numpy()
    oof_solv = np.full(len(df), np.nan)
    oof_lumo = np.full(len(df), np.nan)

    t0 = time.time()
    for fi, (tr, te) in enumerate(folds.split()):
        tf.keras.backend.clear_session()
        np.random.seed(0)
        tf.random.set_seed(0)

        # inner validation split by scaffold, so early stopping never sees a
        # scaffold it will be tested on
        tr_df = df.iloc[tr]
        scafs = tr_df["scaffold"].unique()
        rng = np.random.default_rng(fi)
        rng.shuffle(scafs)
        n_val_scaf = max(1, int(len(scafs) * VALID_FRAC_OF_TRAIN))
        val_scafs = set(scafs[:n_val_scaf])
        is_val = tr_df["scaffold"].isin(val_scafs).to_numpy()
        tr_idx, va_idx = tr[~is_val], tr[is_val]

        vocab = build_vocab(df["tok"].iloc[tr_idx].tolist())
        stoi = {c: i for i, c in enumerate(vocab)}
        pad_id, unk_id = stoi[PAD_TOKEN], stoi[UNK_TOKEN]

        Xtr, ktr = vectorize(df["tok"].iloc[tr_idx].tolist(), stoi, MAX_LEN, pad_id, unk_id)
        Xva, kva = vectorize(df["tok"].iloc[va_idx].tolist(), stoi, MAX_LEN, pad_id, unk_id)
        Xte, kte = vectorize(df["tok"].iloc[te].tolist(), stoi, MAX_LEN, pad_id, unk_id)

        ys_tr, yl_tr = y_solv[tr_idx][ktr], y_lumo[tr_idx][ktr]
        ys_va, yl_va = y_solv[va_idx][kva], y_lumo[va_idx][kva]

        s_mean, s_scale = float(ys_tr.mean()), float(ys_tr.std()) or 1.0
        l_mean, l_scale = float(yl_tr.mean()), float(yl_tr.std()) or 1.0

        model = build_model(len(vocab), MAX_LEN, DROPOUT)
        model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                      loss={"solv": "mse", "lumo": "mse"})
        model.fit(
            Xtr, {"solv": (ys_tr - s_mean) / s_scale, "lumo": (yl_tr - l_mean) / l_scale},
            validation_data=(Xva, {"solv": (ys_va - s_mean) / s_scale,
                                   "lumo": (yl_va - l_mean) / l_scale}),
            epochs=EPOCHS, batch_size=BATCH, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                                        restore_best_weights=True)],
        )
        pred = model.predict(Xte, batch_size=512, verbose=0)
        oof_solv[te[kte]] = pred["solv"].reshape(-1) * s_scale + s_mean
        oof_lumo[te[kte]] = pred["lumo"].reshape(-1) * l_scale + l_mean
        print(f"  fold {fi+1}/5 done ({time.time()-t0:.0f}s cumulative), "
              f"test n={len(kte)} (dropped {len(te)-len(kte)} over MAX_LEN)", flush=True)

    ok = np.isfinite(oof_solv) & np.isfinite(oof_lumo)
    print(f"\nscored {int(ok.sum()):,}/{len(df):,} "
          f"({len(df)-int(ok.sum())} dropped by MAX_LEN={MAX_LEN})")

    row = {"model": "bilstm_multitask", "blocks": "char_smiles",
           "seconds": round(time.time() - t0, 1)}
    for tgt, yt, yp in (("solv", y_solv[ok], oof_solv[ok]), ("lumo", y_lumo[ok], oof_lumo[ok])):
        for k, v in regression_metrics(yt, yp).items():
            row[f"{tgt}_{k}"] = v
    row.update(decision_metrics(y_solv[ok], y_lumo[ok], oof_solv[ok], oof_lumo[ok]))
    row["n_scored"] = int(ok.sum())

    pd.DataFrame([row]).to_csv(RESULTS / "bilstm_scaffold_cv.csv", index=False)
    np.savez_compressed(RESULTS / "oof_bilstm.npz",
                        solv=oof_solv, lumo=oof_lumo, ok=ok, fold=folds.fold_of_row)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
                     indent=2))
    print(f"\n[saved] {RESULTS/'bilstm_scaffold_cv.csv'}")


if __name__ == "__main__":
    main()
