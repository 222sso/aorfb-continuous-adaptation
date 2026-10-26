from __future__ import annotations

"""Generate the fixed 3x3 grid, score it, and compute scaffold endpoints."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"
CONFIG = ROOT / "tcnlstm_transfer_outputs_260311" / "exp_baseline" / "config_used.json"
PREDICTORS = ROOT / "predictor_v2_260811" / "artifacts" / "lgbm_morgan_final"
TEMPERATURES = (0.70, 0.90, 1.10)
TOP_PS = (0.75, 0.85, 0.95)
GEN_N = 1000
TARGET_SOLV_MAX = -1.5
TARGET_LUMO_MAX = -4.0

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

from ablation.config import CFG  # noqa: E402
from ablation.data import detokenize_to_plain_smiles  # noqa: E402
from ablation.models import build_A_TCN  # noqa: E402
from batch_sampling_v2 import generate_smiles_batched  # noqa: E402


RDLogger.DisableLog("rdApp.*")
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, countSimulation=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, required=True, choices=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--seed", type=int, required=True, choices=(42, 43, 44))
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def beta_slug(beta: float) -> str:
    return f"beta_{beta:.2f}".replace(".", "p")


def value_slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def stable_seed(training_seed: int, temperature: float, top_p: float) -> int:
    # Common random-number seed across beta arms; beta is deliberately excluded.
    value = f"beta-frontier|{training_seed}|{temperature:.2f}|{top_p:.2f}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "little")


def load_cfg() -> CFG:
    cfg = CFG()
    payload = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    for key, value in payload.items():
        setattr(cfg, key, value)
    return cfg


def scaffold_from_mol(mol: Chem.Mol) -> str:
    value = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return value or "[ACYCLIC]"


def load_predictors():
    import lightgbm as lgb
    return {
        target: [
            lgb.Booster(model_file=str(PREDICTORS / f"lgbm_{target}_seed{seed}.txt"))
            for seed in (42, 43, 44)
        ]
        for target in ("solv", "lumo")
    }


def score_records(frame: pd.DataFrame, models: dict) -> pd.DataFrame:
    frame = frame.copy()
    mols = [Chem.MolFromSmiles(value) for value in frame.smiles_in.astype(str)]
    frame["rdkit_valid"] = [mol is not None for mol in mols]
    frame["canon"] = [Chem.MolToSmiles(mol, canonical=True) if mol is not None else None for mol in mols]
    frame["scaffold"] = [scaffold_from_mol(mol) if mol is not None else None for mol in mols]
    valid_idx = np.flatnonzero(frame.rdkit_valid.to_numpy(bool))
    x = np.zeros((len(valid_idx), 2048), dtype=np.float32)
    for row, idx in enumerate(valid_idx):
        x[row] = FPGEN.GetCountFingerprintAsNumPy(mols[idx]).astype(np.float32)
    for target, mean_col, std_col in (
        ("solv", "solv_mean_eV", "solv_std_eV"),
        ("lumo", "lumo_mean_eV", "lumo_std_eV"),
    ):
        pred = np.stack([model.predict(x) for model in models[target]], axis=0)
        frame[mean_col] = np.nan
        frame[std_col] = np.nan
        frame.loc[frame.rdkit_valid, mean_col] = pred.mean(axis=0)
        frame.loc[frame.rdkit_valid, std_col] = pred.std(axis=0, ddof=0)
    frame["is_target"] = (
        frame.rdkit_valid
        & (frame.solv_mean_eV <= TARGET_SOLV_MAX)
        & (frame.lumo_mean_eV <= TARGET_LUMO_MAX)
    )
    return frame


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else float("nan")


def summarize(frame: pd.DataFrame, reddb_scaffolds: set[str]) -> pd.DataFrame:
    records = []
    for (temperature, top_p), group in frame.groupby(["temperature", "top_p"]):
        valid = group[group.rdkit_valid].copy()
        target = valid[valid.is_target].copy()
        new = valid[~valid.scaffold.isin(reddb_scaffolds)].copy()
        new_target = new[new.is_target].copy()
        n_total = len(group)
        unique_valid = set(valid.canon.dropna().astype(str))
        unique_target = set(target.canon.dropna().astype(str))
        unique_new_target = set(new_target.canon.dropna().astype(str))
        new_scaffolds = set(new.scaffold.dropna().astype(str))
        target_new_scaffolds = set(new_target.scaffold.dropna().astype(str))
        records.append({
            "temperature": float(temperature),
            "top_p": float(top_p),
            "n_attempts": n_total,
            "n_valid_records": len(valid),
            "n_unique_valid_molecules": len(unique_valid),
            "n_target_records": len(target),
            "n_unique_target_molecules": len(unique_target),
            "n_new_scaffold_records": len(new),
            "n_new_scaffold_target_records": len(new_target),
            "n_unique_new_scaffold_target_molecules": len(unique_new_target),
            "n_distinct_new_scaffolds": len(new_scaffolds),
            "n_target_bearing_new_scaffolds": len(target_new_scaffolds),
            "validity_pct_attempts": pct(len(valid), n_total),
            "uniqueness_pct_valid_records": pct(len(unique_valid), len(valid)),
            "target_pass_pct_valid_records": pct(len(target), len(valid)),
            "raw_target_yield_pct_attempts": pct(len(target), n_total),
            "nonduplicate_target_yield_pct_attempts": pct(len(unique_target), n_total),
            "new_scaffold_supply_pct_attempts": pct(len(new), n_total),
            "target_pass_within_new_scaffold_pct": pct(len(new_target), len(new)),
            "raw_new_scaffold_target_yield_pct_attempts": pct(len(new_target), n_total),
            "nonduplicate_new_scaffold_target_yield_pct_attempts": pct(len(unique_new_target), n_total),
        })
    return pd.DataFrame(records).sort_values(["temperature", "top_p"])


def main() -> None:
    args = parse_args()
    base_dir = HERE / "smoke" if args.smoke_test else RESULTS
    run_dir = base_dir / f"seed_{args.seed}" / beta_slug(args.beta)
    if not (run_dir / "training_complete.json").exists():
        raise FileNotFoundError(f"Training incomplete: {run_dir}")
    complete = run_dir / "generation_complete.json"
    if complete.exists() and not args.smoke_test:
        print(f"SKIP generation beta={args.beta} seed={args.seed}")
        return

    cfg = load_cfg()
    cfg.SEED = args.seed
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(args.seed)
    vocab = json.loads((ARTIFACTS / "vocab.json").read_text(encoding="utf-8"))
    stoi = {str(key): int(value) for key, value in vocab["stoi"].items()}
    itos = {int(key): str(value) for key, value in vocab["itos"].items()}
    model = build_A_TCN(
        len(stoi), cfg, emb_dim=128, tcn_channels=256, tcn_kernel=3,
        tcn_dilations=(1, 2, 4, 8), dropout=0.1,
    )
    model.load_weights(run_dir / "best_common_unit_val.weights.h5")

    temperatures = (0.90,) if args.smoke_test else TEMPERATURES
    top_ps = (0.85,) if args.smoke_test else TOP_PS
    gen_n = 20 if args.smoke_test else GEN_N
    rows = []
    generation_seeds = {}
    for temperature in temperatures:
        for top_p in top_ps:
            seed = stable_seed(args.seed, temperature, top_p)
            generation_seeds[f"{temperature:.2f}|{top_p:.2f}"] = seed
            tokenized = generate_smiles_batched(
                model, itos, stoi, cfg, gen_n=gen_n, temperature=temperature,
                top_p=top_p, seed=seed, batch_size=256,
                progress=lambda done, total: print(
                    f"[beta={args.beta} seed={args.seed} T={temperature} p={top_p}] {done}/{total}",
                    flush=True,
                ),
            )
            plain = [detokenize_to_plain_smiles(value, cfg) for value in tokenized]
            path = run_dir / f"A_TCN_gen_n{gen_n}_T{value_slug(temperature)}_p{value_slug(top_p)}.txt"
            path.write_text("\n".join(plain) + "\n", encoding="utf-8")
            rows.extend({
                "beta": float(args.beta), "training_seed": args.seed,
                "temperature": temperature, "top_p": top_p,
                "attempt_index": index, "smiles_in": value,
            } for index, value in enumerate(plain))

    del model
    tf.keras.backend.clear_session()
    scored = score_records(pd.DataFrame(rows), load_predictors())
    ref = json.loads((ARTIFACTS / "reddb_reference_scaffolds.json").read_text(encoding="utf-8"))
    reddb_scaffolds = set(ref["reddb_scaffolds"])
    metrics = summarize(scored, reddb_scaffolds)
    scored.to_csv(run_dir / "samples_scored_v2.csv.gz", index=False, compression="gzip")
    metrics.to_csv(run_dir / "grid_metrics.csv", index=False)

    valid = scored[scored.rdkit_valid]
    new = valid[~valid.scaffold.isin(reddb_scaffolds)]
    new_target = new[new.is_target]
    payload = {
        "beta": float(args.beta),
        "training_seed": args.seed,
        "n_grid_points": len(metrics),
        "attempts_per_grid_point": gen_n,
        "generation_seeds_common_across_beta": generation_seeds,
        "distinct_new_scaffolds_union": int(new.scaffold.nunique()),
        "target_bearing_new_scaffolds_union": int(new_target.scaffold.nunique()),
    }
    complete.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

