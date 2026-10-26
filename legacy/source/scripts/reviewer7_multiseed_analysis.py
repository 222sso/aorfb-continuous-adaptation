from __future__ import annotations

"""Score the completed 3-seed generator replication with the frozen manuscript predictor.

Existing experiment outputs are read-only. Metrics are first averaged over the same
3 x 3 decoding grid within each training seed; means and sample SDs are then computed
across independent training seeds 42, 43, and 44.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
REPLICATION = (
    ROOT / "q1_comparison_experiments_260806" / "generality_replication_260816" / "results"
)
MODEL_DIR = ROOT / "predictor_v2_260811" / "artifacts" / "lgbm_morgan_final"
ARTIFACTS = (
    ROOT
    / "q1_comparison_experiments_260806"
    / "generality_replication_260816"
    / "artifacts"
)
OUT = ROOT / "output" / "reviewer7_multiseed"
BACKBONES = {"TCN": "reddb", "LSTM": "reddb_B_LSTM", "TCN-LSTM": "reddb_C_TCN_LSTM"}
ARMS = ("equal", "hard", "soft")
SEEDS = (42, 43, 44)
ATTEMPTS_PER_CELL = 1000

RDLogger.DisableLog("rdApp.*")
MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, countSimulation=False
)


def load_models():
    models = {}
    for target in ("solv", "lumo"):
        paths = [MODEL_DIR / f"lgbm_{target}_seed{s}.txt" for s in SEEDS]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(missing)
        models[target] = [lgb.Booster(model_file=str(path)) for path in paths]
    return models


def load_runs():
    frames, audit = [], []
    for backbone, folder in BACKBONES.items():
        for arm in ARMS:
            for seed in SEEDS:
                path = REPLICATION / folder / arm / f"seed_{seed}" / "generated_molecules.csv.gz"
                if not path.exists():
                    raise FileNotFoundError(path)
                frame = pd.read_csv(path)
                required = {
                    "rdkit_valid", "canonical_smiles", "temperature", "top_p", "novel_molecule"
                }
                if missing := sorted(required - set(frame.columns)):
                    raise ValueError(f"{path}: missing columns {missing}")
                counts = frame.groupby(["temperature", "top_p"], dropna=False).size()
                if len(frame) != 9000 or len(counts) != 9 or not (counts == 1000).all():
                    raise ValueError(f"{path}: incomplete grid {counts.to_dict()}")
                frame["backbone"], frame["arm"], frame["training_seed"] = backbone, arm, seed
                frames.append(frame)
                audit.append(
                    {
                        "backbone": backbone,
                        "arm": arm,
                        "training_seed": seed,
                        "n_attempts": len(frame),
                        "n_cells": len(counts),
                        "n_valid": int(frame.rdkit_valid.astype(bool).sum()),
                        "source_file": str(path.relative_to(ROOT)),
                    }
                )
    return pd.concat(frames, ignore_index=True), audit


def score(rows, models):
    valid = rows.rdkit_valid.astype(bool) & rows.canonical_smiles.notna()
    smiles = sorted(rows.loc[valid, "canonical_smiles"].astype(str).unique())
    print(
        f"{len(rows):,} attempts; {int(valid.sum()):,} valid; "
        f"{len(smiles):,} unique valid molecules",
        flush=True,
    )
    x = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Stored canonical SMILES failed parsing: {smi}")
        x[i] = MORGAN.GetCountFingerprintAsNumPy(mol).astype(np.float32)
        if (i + 1) % 10000 == 0:
            print(f"  featurized {i + 1:,}/{len(smiles):,}", flush=True)
    lookup = pd.DataFrame({"canonical_smiles": smiles})
    for target in ("solv", "lumo"):
        pred = np.stack([model.predict(x) for model in models[target]])
        lookup[f"{target}_mean_eV"] = pred.mean(axis=0)
        lookup[f"{target}_std_eV"] = pred.std(axis=0, ddof=0)
    lookup["pred_target"] = (
        (lookup.solv_mean_eV <= -1.5) & (lookup.lumo_mean_eV <= -4.0)
    )
    out = rows.merge(lookup, on="canonical_smiles", how="left", validate="many_to_one")
    if out.loc[valid, "pred_target"].isna().any():
        raise ValueError("Some valid records were not scored")
    out["pred_target"] = out.pred_target.fillna(False).astype(bool)
    flat_lookup = {}
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        flat_lookup[smi] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    out["flat_canonical"] = out.canonical_smiles.map(flat_lookup)
    return out


def summarize_cell(group, novelty_reference):
    valid = group[group.rdkit_valid.astype(bool) & group.canonical_smiles.notna()].copy()
    unique = valid.drop_duplicates("flat_canonical")
    target = valid[valid.pred_target]
    unique_target = target.drop_duplicates("flat_canonical")
    unique_novel_target = unique_target[
        ~unique_target.flat_canonical.isin(novelty_reference)
    ]
    n_attempts, n_valid, n_unique = len(group), len(valid), len(unique)
    return pd.Series(
        {
            "n_attempts": n_attempts,
            "n_valid": n_valid,
            "n_unique": n_unique,
            "n_targets": len(target),
            "n_unique_targets": len(unique_target),
            "n_unique_novel_targets": len(unique_novel_target),
            "validity_pct": 100 * n_valid / n_attempts,
            "uniqueness_pct": 100 * n_unique / n_valid if n_valid else np.nan,
            "novelty_pct": (
                100 * (~unique.flat_canonical.isin(novelty_reference)).sum() / n_unique
                if n_unique else np.nan
            ),
            "target_pass_pct": 100 * len(target) / n_valid if n_valid else np.nan,
            "yield_raw_target_pct": 100 * len(target) / n_attempts,
            "yield_unique_target_pct": 100 * len(unique_target) / n_attempts,
            "yield_unique_novel_target_pct": 100 * len(unique_novel_target) / n_attempts,
        }
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, audit = load_runs()
    scored = score(rows, load_models())
    corpus = pd.read_csv(ARTIFACTS / "reddb.csv.gz")
    train_smiles = corpus.loc[corpus.split == "train", "canonical_smiles"].astype(str)
    novelty_reference = set()
    for smi in train_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            novelty_reference.add(
                Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            )
    keys = ["backbone", "arm", "training_seed", "temperature", "top_p"]
    cells = (
        scored.groupby(keys, sort=True, dropna=False)
        .apply(
            lambda group: summarize_cell(group, novelty_reference),
            include_groups=False,
        )
        .reset_index()
    )
    if len(cells) != 243:
        raise ValueError(f"Expected 243 cell rows, found {len(cells)}")
    metrics = [
        "validity_pct", "uniqueness_pct", "novelty_pct", "target_pass_pct",
        "yield_raw_target_pct", "yield_unique_target_pct",
        "yield_unique_novel_target_pct",
    ]
    per_seed = (
        cells.groupby(["backbone", "arm", "training_seed"], sort=True)[metrics]
        .mean().reset_index()
    )
    group = per_seed.groupby(["backbone", "arm"], sort=True)[metrics].agg(["mean", "std"])
    group.columns = [f"{metric}_{stat}" for metric, stat in group.columns]
    group = group.reset_index()

    cells.to_csv(OUT / "cell_metrics_243.csv", index=False)
    per_seed.to_csv(OUT / "seed_metrics_27.csv", index=False)
    group.to_csv(OUT / "table1_multiseed_mean_sd.csv", index=False)
    pd.DataFrame(audit).to_csv(OUT / "run_integrity_27.csv", index=False)
    checks = {
        "n_runs": len(audit),
        "n_cells": len(cells),
        "n_seed_summaries": len(per_seed),
        "n_group_summaries": len(group),
        "all_cases_9000_attempts": all(x["n_attempts"] == 9000 for x in audit),
        "all_cases_9_cells": all(x["n_cells"] == 9 for x in audit),
        "all_group_sizes_3": bool(
            (per_seed.groupby(["backbone", "arm"]).size() == 3).all()
        ),
        "predictor": {
            "model": "frozen LightGBM-Morgan count ensemble",
            "predictor_seeds": list(SEEDS),
            "thresholds_eV": {"solvation": -1.5, "lumo": -4.0},
            "model_card": str((MODEL_DIR / "MODEL_CARD.json").relative_to(ROOT)),
        },
        "generator_replication": {
            "training_seeds": list(SEEDS),
            "novelty_reference": (
                f"{len(novelty_reference)} non-isomeric canonical molecules in the common "
                "RedDB training split"
            ),
            "note": (
                "Independent reimplementation with a shared union vocabulary, shared "
                "ChEMBL checkpoint within each backbone, and common unit-weight validation "
                "selection; outputs were re-scored with the frozen manuscript predictor."
            ),
        },
    }
    (OUT / "audit.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    display = group.copy()
    for metric in metrics:
        display[metric] = display.apply(
            lambda r: f"{r[f'{metric}_mean']:.2f} ± {r[f'{metric}_std']:.2f}", axis=1
        )
    print("\n3-seed means ± sample SD across seed-level 3 x 3 decoding-grid means")
    print(display[["backbone", "arm", *metrics]].to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
