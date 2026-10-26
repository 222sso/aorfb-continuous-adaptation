# -*- coding: utf-8 -*-
"""
Re-score generated molecules with the v2 predictor (LightGBM + Morgan counts).

Emits `samples_scored_all.csv` in exactly the schema the study's analysis code
already consumes, so `exp_hard_subset_260806/code/analyze_three_way.py` can read
v2 scores without modification:

    case_id, model, gen_n, temperature, top_p, file,
    smiles_in, canon, rdkit_valid,
    solv_mean_eV, solv_std_eV, lumo_mean_eV, lumo_std_eV

Differences from the v1 (BiLSTM) scorer that matter for the comparison:

  * no MAX_LEN truncation — v1 silently dropped every molecule over 70 tokens
    (2.3 % of the hard arm vs 0.9 % of the soft arm), removing precisely the long
    polyacidic structures most likely to be inside the target region. Every
    RDKit-valid molecule is scored here.
  * mean / std are taken across the three LightGBM seeds, matching how the v1
    ensemble uncertainty columns were produced.

Run:
  python predictor_v2_260811/code/score_generated.py --arm hard
  python predictor_v2_260811/code/score_generated.py --all
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(HERE))
ARTIFACTS = HERE.parent / "artifacts" / "lgbm_morgan_final"

SEEDS = [42, 43, 44]
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, countSimulation=False)

ARM_DIRS = {
    "hard": BASE / "exp_hard_subset_260806" / "results" / "hard_primary",
    "soft": BASE / "tcnlstm_transfer_outputs_260311" / "exp_baseline",
    "equal": BASE / "tcnlstm_transfer_outputs_260311" / "exp_noweight",
}

_CASE_RE = re.compile(r"^(?P<model>.+?)_gen_n(?P<n>\d+)_T(?P<T>\d+p\d+)_p(?P<p>\d+p\d+)\.txt$")


def parse_case(path: Path):
    m = _CASE_RE.match(path.name)
    if not m:
        return None
    return {
        "case_id": path.stem,
        "model": m.group("model"),
        "gen_n": int(m.group("n")),
        "temperature": float(m.group("T").replace("p", ".")),
        "top_p": float(m.group("p").replace("p", ".")),
        "file": str(path),
    }


def load_models():
    import lightgbm as lgb
    models = {}
    for target in ("solv", "lumo"):
        models[target] = [
            lgb.Booster(model_file=str(ARTIFACTS / f"lgbm_{target}_seed{s}.txt")) for s in SEEDS
        ]
        if len(models[target]) != len(SEEDS):
            raise FileNotFoundError(f"missing {target} boosters in {ARTIFACTS}")
    return models


def score_arm(arm: str, models) -> pd.DataFrame:
    outdir = ARM_DIRS[arm]
    files = sorted(p for p in outdir.glob("*_gen_*.txt"))
    if not files:
        raise FileNotFoundError(f"no generation files in {outdir}")
    print(f"[{arm}] {len(files)} decoding settings", flush=True)

    rows = []
    for path in files:
        case = parse_case(path)
        if case is None:
            print(f"  skipped unparseable filename: {path.name}")
            continue
        smis = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for s in smis:
            mol = Chem.MolFromSmiles(s)
            rows.append({
                **case,
                "smiles_in": s,
                "canon": Chem.MolToSmiles(mol, canonical=True) if mol is not None else None,
                "rdkit_valid": 1 if mol is not None else 0,
            })

    df = pd.DataFrame(rows)
    valid = df["rdkit_valid"] == 1
    canon = df.loc[valid, "canon"].tolist()
    print(f"[{arm}] {len(df):,} generated, {int(valid.sum()):,} RDKit-valid -> scoring all "
          f"(no length limit)", flush=True)

    X = np.zeros((len(canon), 2048), dtype=np.float32)
    for i, s in enumerate(canon):
        X[i] = _GEN.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(np.float32)

    for target, mean_col, std_col in (("solv", "solv_mean_eV", "solv_std_eV"),
                                      ("lumo", "lumo_mean_eV", "lumo_std_eV")):
        preds = np.stack([b.predict(X) for b in models[target]], axis=0)
        df[mean_col] = np.nan
        df[std_col] = np.nan
        df.loc[valid, mean_col] = preds.mean(axis=0)
        df.loc[valid, std_col] = preds.std(axis=0, ddof=0)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARM_DIRS))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    arms = list(ARM_DIRS) if args.all else [args.arm]
    if arms == [None]:
        ap.error("pass --arm <name> or --all")

    models = load_models()
    for arm in arms:
        df = score_arm(arm, models)
        out = ARM_DIRS[arm] / "pred_eval_v2"
        out.mkdir(exist_ok=True)
        df.to_csv(out / "samples_scored_all.csv", index=False, encoding="utf-8-sig")

        hits = ((df.solv_mean_eV <= -1.5) & (df.lumo_mean_eV <= -4.0)).sum()
        print(f"[{arm}] target-region hits (all settings): {int(hits):,} "
              f"-> {out/'samples_scored_all.csv'}\n", flush=True)


if __name__ == "__main__":
    main()
