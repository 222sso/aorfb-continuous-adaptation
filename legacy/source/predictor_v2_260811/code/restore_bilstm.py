# -*- coding: utf-8 -*-
"""
RESTORATION: the multitask BiLSTM ensemble that scores generated molecules.

The deployed predictor `pred_outputs_multitask_scaffoldsplit_repeat/` has scalers
in eV, but no script in the repository produces it — the only predictor trainer
present (`ablation/train_predictor_compare.py`) reads the source CSV in Hartree
and never converts, so it produces a different (Hartree) model and writes to a
different directory. Every target-region label in the study therefore rests on
artefacts that could not be regenerated.

This script restores that training path:

  * loads RedDB and converts Hartree -> eV explicitly and assertively (data.py)
  * uses the same character tokenizer, with the ¤ (SO3H) / § (COOH) compression
    tokens, that `ablation/predict_generated.py` applies at inference
  * builds the exact architecture `predict_generated.py:build_pred_model`
    reconstructs for the "baseline" case, so the weights load unchanged
  * writes `seed{N}_best_model.weights.h5`, `seed{N}_vocab.json`,
    `seed{N}_scalers.json` — the three files the inference loader expects

The output directory is therefore a drop-in replacement for the deployed one.
It is written to a NEW directory; the original artefacts are never overwritten.

Run:
  python predictor_v2_260811/code/restore_bilstm.py --outdir predictor_v2_260811/artifacts/bilstm_restored
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import load_dataset, legacy_single_split  # noqa: E402

PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
TOK_SO3H, TOK_COOH = "¤", "§"
SO3H_EQUIV = ["O=S(=O)(O)", "O=S(O)(=O)", "S(=O)(O)=O", "S(O)(=O)=O"]
SO3H_CANON = "S(=O)(=O)O"
COOH_EQUIV = ["O=C(O)"]
COOH_CANON = "C(=O)O"


def encode_special_pred(s: str) -> str:
    out = str(s)
    for patt in SO3H_EQUIV:
        out = out.replace(patt, SO3H_CANON)
    out = out.replace(SO3H_CANON, TOK_SO3H)
    for patt in COOH_EQUIV:
        out = out.replace(patt, COOH_CANON)
    out = out.replace(COOH_CANON, TOK_COOH)
    return out


def build_vocab(tokenized) -> list:
    return [PAD_TOKEN, UNK_TOKEN] + sorted({ch for s in tokenized for ch in s})


def encode_one(tok, stoi, max_len, pad_id, unk_id):
    if len(tok) > max_len:
        return None
    ids = [stoi.get(ch, unk_id) for ch in tok]
    return np.array(ids + [pad_id] * (max_len - len(ids)), dtype=np.int32)


def vectorize(toks, stoi, max_len, pad_id, unk_id):
    X, keep = [], []
    for i, t in enumerate(toks):
        a = encode_one(t, stoi, max_len, pad_id, unk_id)
        if a is None:
            continue
        X.append(a); keep.append(i)
    return np.stack(X), np.asarray(keep)


def build_model(vocab_size, max_len, dropout=0.2):
    """Identical to ablation/predict_generated.py:build_pred_model(architecture='baseline')."""
    inp = tf.keras.layers.Input(shape=(max_len,), dtype="int32", name="inp")
    x = tf.keras.layers.Embedding(vocab_size, 128, mask_zero=True)(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False))(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out_solv = tf.keras.layers.Dense(1, name="solv")(x)
    out_lumo = tf.keras.layers.Dense(1, name="lumo")(x)
    return tf.keras.Model(inp, outputs={"solv": out_solv, "lumo": out_lumo})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(HERE.parent / "artifacts" / "bilstm_restored"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max-len", type=int, default=70)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    df = load_dataset()                      # eV, deduplicated, scaffold attached
    df["tok"] = [encode_special_pred(s) for s in df["canon"]]
    print(f"[data] {len(df):,} molecules, {df.scaffold.nunique()} scaffolds, units=eV")

    rows = []
    for seed in args.seeds:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.keras.backend.clear_session()

        tr_i, va_i, te_i = legacy_single_split(df, seed)
        tr, va, te = df.iloc[tr_i], df.iloc[va_i], df.iloc[te_i]

        vocab = build_vocab(tr["tok"].tolist())
        stoi = {c: i for i, c in enumerate(vocab)}
        pad_id, unk_id = stoi[PAD_TOKEN], stoi[UNK_TOKEN]

        Xtr, ktr = vectorize(tr["tok"].tolist(), stoi, args.max_len, pad_id, unk_id)
        Xva, kva = vectorize(va["tok"].tolist(), stoi, args.max_len, pad_id, unk_id)
        Xte, kte = vectorize(te["tok"].tolist(), stoi, args.max_len, pad_id, unk_id)

        ys_tr = tr["solv"].to_numpy()[ktr]; yl_tr = tr["lumo"].to_numpy()[ktr]
        ys_va = va["solv"].to_numpy()[kva]; yl_va = va["lumo"].to_numpy()[kva]
        ys_te = te["solv"].to_numpy()[kte]; yl_te = te["lumo"].to_numpy()[kte]

        # scalers fit on TRAIN only
        s_mean, s_scale = float(ys_tr.mean()), float(ys_tr.std()) or 1.0
        l_mean, l_scale = float(yl_tr.mean()), float(yl_tr.std()) or 1.0

        model = build_model(len(vocab), args.max_len, args.dropout)
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                      loss={"solv": "mse", "lumo": "mse"})

        ckpt = outdir / f"seed{seed}_best_model.weights.h5"
        model.fit(
            Xtr, {"solv": (ys_tr - s_mean) / s_scale, "lumo": (yl_tr - l_mean) / l_scale},
            validation_data=(Xva, {"solv": (ys_va - s_mean) / s_scale,
                                   "lumo": (yl_va - l_mean) / l_scale}),
            epochs=args.epochs, batch_size=args.batch_size, verbose=2,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(str(ckpt), save_weights_only=True,
                                                   save_best_only=True, monitor="val_loss"),
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                                 restore_best_weights=True),
            ],
        )

        pred = model.predict(Xte, batch_size=512, verbose=0)
        ps = pred["solv"].reshape(-1) * s_scale + s_mean
        pl = pred["lumo"].reshape(-1) * l_scale + l_mean

        def r2(y, p):
            return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())

        row = {
            "seed": seed, "n_train": len(Xtr), "n_valid": len(Xva), "n_test": len(Xte),
            "n_scaf_train": tr.scaffold.nunique(), "n_scaf_test": te.scaffold.nunique(),
            "solv_mae": float(np.abs(ps - ys_te).mean()), "solv_r2": r2(ys_te, ps),
            "lumo_mae": float(np.abs(pl - yl_te).mean()), "lumo_r2": r2(yl_te, pl),
        }
        rows.append(row)
        print(f"[seed {seed}] solv MAE {row['solv_mae']:.4f} R2 {row['solv_r2']:.4f} | "
              f"lumo MAE {row['lumo_mae']:.4f} R2 {row['lumo_r2']:.4f}")

        # artefacts in the exact layout ablation/predict_generated.py loads
        (outdir / f"seed{seed}_vocab.json").write_text(json.dumps({
            "vocab": vocab, "PAD_TOKEN": PAD_TOKEN, "UNK_TOKEN": UNK_TOKEN,
            "MAX_LEN": args.max_len, "architecture": "baseline", "head_units": 128,
        }, ensure_ascii=False), encoding="utf-8")
        (outdir / f"seed{seed}_scalers.json").write_text(json.dumps({
            "solv_mean": s_mean, "solv_scale": s_scale,
            "lumo_mean": l_mean, "lumo_scale": l_scale,
        }), encoding="utf-8")

    m = pd.DataFrame(rows)
    m.to_csv(outdir / "metrics_by_seed.csv", index=False)
    summ = m[["solv_mae", "solv_r2", "lumo_mae", "lumo_r2"]].agg(["mean", "std"]).T
    summ.to_csv(outdir / "metrics_summary.csv")
    (outdir / "PROVENANCE.json").write_text(json.dumps({
        "restores": "pred_outputs_multitask_scaffoldsplit_repeat",
        "units": "eV (Hartree x 27.211386245988, asserted in data.py)",
        "source_csv": "dist_reports/merged_all_inchikey_v2.csv",
        "architecture": "Embedding(128, mask_zero) -> BiLSTM(256) -> Dense(256,relu) -> Dropout -> {solv,lumo}",
        "split": "legacy single scaffold split 80/10/10 per seed",
        "seeds": args.seeds, "max_len": args.max_len,
    }, indent=2), encoding="utf-8")

    print("\n=== restored ensemble ===")
    print(m.to_string(index=False))
    print(f"\n[saved] {outdir}")


if __name__ == "__main__":
    main()
