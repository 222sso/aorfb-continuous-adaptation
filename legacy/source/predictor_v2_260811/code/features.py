# -*- coding: utf-8 -*-
"""
Predictor v2 — molecular features for DFT solvation energy and solution-phase LUMO.

Feature blocks
--------------
  rdkit2d   ~210 RDKit 2D descriptors. Carries the physics that drives both
            targets: TPSA / H-bond counts / MolLogP / molar refractivity for
            solvation, and PEOE_VSA (partial-charge surface area), EState indices,
            aromatic-ring and conjugation counts for the frontier orbital.
  morgan    Morgan *count* fingerprint (radius 2, folded). Counts rather than bits
            because these molecules differ mainly in HOW MANY acidic groups they
            carry, not whether they carry one.
  fg        Explicit counts of the functional groups the target-region analysis
            identified as discriminating: -COOH, -SO3H, C=O, aromatic N, -OH,
            -NH2, -F, plus a total acidic-group count.

`build_features` caches to .npz so model sweeps do not recompute descriptors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

CACHE = Path(__file__).resolve().parents[1] / "cache"
CACHE.mkdir(exist_ok=True)

# SMARTS for the groups that separate the target region (Results/target_region_analysis)
FG_SMARTS: Dict[str, str] = {
    "cooh": "[CX3](=O)[OX2H1]",
    "so3h": "[SX4](=O)(=O)[OX2H1]",
    "carbonyl": "[CX3]=[OX1]",
    "arom_n": "[n]",
    "oh": "[OX2H]",
    "nh2": "[NX3;H2]",
    "f": "[F]",
    "nitro": "[N+](=O)[O-]",
    "arom_ring": "c1ccccc1",
}
_FG_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in FG_SMARTS.items()}

_DESC_NAMES: List[str] = [n for n, _ in Descriptors._descList]
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, countSimulation=False)


def _rdkit2d(mol) -> np.ndarray:
    out = np.empty(len(_DESC_NAMES), dtype=np.float64)
    for i, (_, fn) in enumerate(Descriptors._descList):
        try:
            v = fn(mol)
        except Exception:
            v = np.nan
        out[i] = v
    return out


def _fg_counts(mol) -> np.ndarray:
    v = []
    for k in FG_SMARTS:
        p = _FG_PATTERNS[k]
        v.append(len(mol.GetSubstructMatches(p)) if p is not None else 0)
    v.append(v[0] + v[1])                      # total acidic groups (COOH + SO3H)
    return np.asarray(v, dtype=np.float64)


FG_NAMES = list(FG_SMARTS.keys()) + ["n_acidic_total"]


def build_features(canon: List[str], tag: str, use_cache: bool = True
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return (rdkit2d, morgan_counts, fg_counts, rdkit2d_names)."""
    path = CACHE / f"features_{tag}.npz"
    if use_cache and path.exists():
        z = np.load(path, allow_pickle=True)
        return z["d2d"], z["morgan"], z["fg"], list(z["names"])

    n = len(canon)
    d2d = np.full((n, len(_DESC_NAMES)), np.nan)
    morgan = np.zeros((n, 2048), dtype=np.float32)
    fg = np.zeros((n, len(FG_NAMES)))

    for i, s in enumerate(canon):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        d2d[i] = _rdkit2d(mol)
        fp = _MORGAN_GEN.GetCountFingerprintAsNumPy(mol)
        morgan[i] = fp.astype(np.float32)
        fg[i] = _fg_counts(mol)
        if (i + 1) % 5000 == 0:
            print(f"  featurised {i+1:,}/{n:,}", flush=True)

    np.savez_compressed(path, d2d=d2d, morgan=morgan, fg=fg, names=np.array(_DESC_NAMES, dtype=object))
    return d2d, morgan, fg, _DESC_NAMES


def assemble(d2d, morgan, fg, blocks=("rdkit2d", "morgan", "fg")) -> Tuple[np.ndarray, List[str]]:
    """Concatenate the requested blocks and return (X, feature_names)."""
    parts, names = [], []
    if "rdkit2d" in blocks:
        parts.append(d2d)
        names += [f"d2d::{n}" for n in _DESC_NAMES]
    if "morgan" in blocks:
        parts.append(morgan)
        names += [f"mo::{i}" for i in range(morgan.shape[1])]
    if "fg" in blocks:
        parts.append(fg)
        names += [f"fg::{n}" for n in FG_NAMES]
    X = np.concatenate(parts, axis=1)
    return X, names


def clean_matrix(X_train: np.ndarray, *others: np.ndarray):
    """
    Drop non-finite / constant columns using TRAIN statistics only, then median-impute.
    Fitting these on train alone keeps the scaffold split honest.
    """
    finite = np.isfinite(X_train)
    frac_finite = finite.mean(axis=0)
    keep = frac_finite > 0.99

    Xt = X_train[:, keep].copy()
    med = np.nanmedian(np.where(np.isfinite(Xt), Xt, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    def fix(X):
        Z = X[:, keep].astype(np.float64, copy=True)
        bad = ~np.isfinite(Z)
        if bad.any():
            Z[bad] = np.take(med, np.where(bad)[1])
        return Z

    Xt = fix(X_train)
    std = Xt.std(axis=0)
    nonconst = std > 1e-12
    keep_idx = np.where(keep)[0][nonconst]

    def fix2(X):
        Z = X[:, keep].astype(np.float64, copy=True)
        bad = ~np.isfinite(Z)
        if bad.any():
            Z[bad] = np.take(med, np.where(bad)[1])
        return Z[:, nonconst]

    return (fix2(X_train), *(fix2(o) for o in others)), keep_idx


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import load_dataset

    df = load_dataset()
    d2d, morgan, fg, names = build_features(df["canon"].tolist(), tag="reddb31k")
    print(f"rdkit2d {d2d.shape}  morgan {morgan.shape}  fg {fg.shape}")
    finite = np.isfinite(d2d).mean(axis=0)
    print(f"descriptors with any non-finite values: {int((finite < 1.0).sum())}/{d2d.shape[1]}")
    print("fg means:", {n: round(float(fg[:, i].mean()), 2) for i, n in enumerate(FG_NAMES)})
