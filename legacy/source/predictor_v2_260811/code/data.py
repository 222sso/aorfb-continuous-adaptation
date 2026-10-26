# -*- coding: utf-8 -*-
"""
Predictor v2 — dataset construction with EXPLICIT unit handling.

Why this file exists
--------------------
The predictor that assigns every target-region label in this study
(`pred_outputs_multitask_scaffoldsplit_repeat/`) has scalers in eV, but the only
predictor training script in the repository (`ablation/train_predictor_compare.py`)
reads `dist_reports/merged_all_inchikey_v2.csv` — which is in HARTREE — and applies
no conversion. The training script for the deployed predictor is not in the
repository, so it could not be retrained. This module restores that path with the
conversion made explicit and asserted.

Unit contract
-------------
  source CSV   : Hartree   (Solvation_energy mean ~ -0.0377)
  everything else: eV      (mean ~ -1.026)
  conversion   : 1 Hartree = 27.211386245988 eV  (CODATA 2018)

The target region is defined in eV (ΔG_solv <= -1.5, E_LUMO <= -4.0), so a silent
unit error would make every downstream label wrong. `load_dataset` asserts the
source looks like Hartree before converting and that the result looks like eV.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

HARTREE_TO_EV = 27.211386245988

BASE = Path(__file__).resolve().parents[2]
DEFAULT_CSV = BASE / "dist_reports" / "merged_all_inchikey_v2.csv"

SMILES_COL = "smiles_main"
SOLV_COL = "Solvation_energy"
LUMO_COL = "single_point_lumo_solution"

# target region (Methods / 04_METHODS_FACTS.md), in eV
TARGET_SOLV_MAX_EV = -1.5
TARGET_LUMO_MAX_EV = -4.0


def canon_smiles(s: str):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)


def murcko(s: str) -> str:
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return ""


def load_dataset(csv_path: Path = DEFAULT_CSV, assume_hartree: bool = True) -> pd.DataFrame:
    """Load RedDB, convert Hartree -> eV, canonicalise, attach Murcko scaffold."""
    df = pd.read_csv(csv_path)
    for c in (SMILES_COL, SOLV_COL, LUMO_COL):
        if c not in df.columns:
            raise ValueError(f"missing column {c!r} in {csv_path}")

    df = df.rename(columns={SMILES_COL: "smiles", SOLV_COL: "solv", LUMO_COL: "lumo"})
    df["solv"] = pd.to_numeric(df["solv"], errors="coerce")
    df["lumo"] = pd.to_numeric(df["lumo"], errors="coerce")
    df = df.dropna(subset=["smiles", "solv", "lumo"]).copy()

    if assume_hartree:
        # a Hartree solvation energy is ~ -0.04; an eV one is ~ -1.0
        if not (-0.5 < df["solv"].mean() < 0.0):
            raise ValueError(
                f"{csv_path.name}: solv mean = {df['solv'].mean():.4f}, which does not look "
                "like Hartree. Refusing to convert — check the source units."
            )
        df["solv"] = df["solv"] * HARTREE_TO_EV
        df["lumo"] = df["lumo"] * HARTREE_TO_EV

    if not (-3.0 < df["solv"].mean() < -0.3):
        raise ValueError(f"post-conversion solv mean = {df['solv'].mean():.4f} eV, out of expected range")
    if not (-6.0 < df["lumo"].mean() < -1.0):
        raise ValueError(f"post-conversion lumo mean = {df['lumo'].mean():.4f} eV, out of expected range")

    df["canon"] = [canon_smiles(s) for s in df["smiles"]]
    df = df.dropna(subset=["canon"]).copy()
    df = df.drop_duplicates(subset=["canon"]).reset_index(drop=True)
    df["scaffold"] = [murcko(s) for s in df["canon"]]
    df["in_target"] = (df["solv"] <= TARGET_SOLV_MAX_EV) & (df["lumo"] <= TARGET_LUMO_MAX_EV)
    return df


@dataclass
class ScaffoldFolds:
    """Grouped CV folds; every molecule sharing a Murcko scaffold stays together."""
    fold_of_row: np.ndarray
    n_folds: int
    scaffolds_per_fold: List[int]

    def split(self):
        for k in range(self.n_folds):
            te = np.where(self.fold_of_row == k)[0]
            tr = np.where(self.fold_of_row != k)[0]
            yield tr, te


def make_scaffold_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> ScaffoldFolds:
    """
    Greedy balanced assignment of scaffold groups to folds.

    The legacy protocol used a single 80/10/10 scaffold split, which left only
    ~11 scaffolds in test and produced R^2 estimates that swung from 0.79 to 0.93
    across seeds. Grouped k-fold uses every molecule for testing exactly once and
    gives a far more stable estimate on a dataset with only ~100 scaffolds.
    """
    rng = np.random.default_rng(seed)
    groups = [(s, idx.to_numpy()) for s, idx in df.groupby("scaffold").groups.items()]
    rng.shuffle(groups)
    groups.sort(key=lambda g: len(g[1]), reverse=True)

    fold_sizes = np.zeros(n_folds, dtype=int)
    fold_scafs = np.zeros(n_folds, dtype=int)
    fold_of_row = np.full(len(df), -1, dtype=int)

    for _, idx in groups:                      # largest group to the emptiest fold
        k = int(np.argmin(fold_sizes))
        fold_of_row[idx] = k
        fold_sizes[k] += len(idx)
        fold_scafs[k] += 1

    if (fold_of_row < 0).any():
        raise RuntimeError("some rows were not assigned to a fold")
    return ScaffoldFolds(fold_of_row, n_folds, fold_scafs.tolist())


def legacy_single_split(df: pd.DataFrame, seed: int,
                        train_frac=0.8, valid_frac=0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduces ablation/train_predictor_compare.py:split_by_scaffold for comparability."""
    rng = np.random.default_rng(seed)
    groups = [(s, idx.to_numpy().tolist()) for s, idx in df.groupby("scaffold").groups.items()]
    rng.shuffle(groups)
    groups.sort(key=lambda x: len(x[1]), reverse=True)

    n = len(df)
    t_train, t_valid = n * train_frac, n * valid_frac
    tr, va, te = [], [], []
    for _, idxs in groups:
        if len(tr) + len(idxs) <= t_train or not tr:
            tr.extend(idxs)
        elif len(va) + len(idxs) <= t_valid or not va:
            va.extend(idxs)
        else:
            te.extend(idxs)
    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))


if __name__ == "__main__":
    d = load_dataset()
    print(f"molecules: {len(d):,}   scaffolds: {d.scaffold.nunique()}")
    print(f"solv eV: mean {d.solv.mean():.4f}  min {d.solv.min():.3f}  max {d.solv.max():.3f}")
    print(f"lumo eV: mean {d.lumo.mean():.4f}  min {d.lumo.min():.3f}  max {d.lumo.max():.3f}")
    print(f"in target region: {int(d.in_target.sum()):,} ({100*d.in_target.mean():.2f}%)")
    f = make_scaffold_folds(d, 5)
    print(f"5-fold scaffold folds -> scaffolds/fold {f.scaffolds_per_fold}, "
          f"sizes {[int((f.fold_of_row==k).sum()) for k in range(5)]}")
