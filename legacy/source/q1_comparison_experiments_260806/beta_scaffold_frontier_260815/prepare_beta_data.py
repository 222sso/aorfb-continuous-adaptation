from __future__ import annotations

"""Freeze data, augmentation, scaffold counts, and vocabulary for beta runs."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACTS = HERE / "artifacts"
CONFIG = ROOT / "tcnlstm_transfer_outputs_260311" / "exp_baseline" / "config_used.json"
PRETRAIN = next(
    (ROOT / "tcnlstm_transfer_outputs_260311" / "exp_baseline").glob(
        "ckpt_pretrain_A_TCN*.weights.h5"
    )
)
PROPERTIES = ROOT / "dist_reports" / "softft_v3" / "softft_scores_weights.csv"
FROZEN_VOCAB = (
    ROOT
    / "q1_comparison_experiments_260806"
    / "checkpoint_selection_control"
    / "artifacts"
    / "vocab.json"
)
SPLIT_SEED = 42
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TARGET_SOLV_MAX = -1.5
TARGET_LUMO_MAX = -4.0

sys.path.insert(0, str(ROOT))

from ablation.config import CFG  # noqa: E402
from ablation.data import (  # noqa: E402
    augment_train_smiles,
    boost_weights_by_energy_fraction,
    encode_special,
    load_reddb_df_with_weights,
    load_softft_weight_map,
    randomize_smiles,
    read_chembl_train_valid,
    split_train_valid_with_w,
)


RDLogger.DisableLog("rdApp.*")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cfg() -> CFG:
    cfg = CFG()
    payload = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    for key, value in payload.items():
        setattr(cfg, key, value)
    for key in ("CHEMBL_CSV", "REDDB_CSV", "WEIGHT_CSV", "PRED_DIR", "NOVEL_REF_PATH"):
        value = getattr(cfg, key, None)
        if value and not Path(value).is_absolute():
            setattr(cfg, key, str((ROOT / value).resolve()))
    cfg.SEED = SPLIT_SEED
    return cfg


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit failed on RedDB molecule: {smiles}")
    value = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return value or "[ACYCLIC]"


def capped_mean_normalizer(raw: np.ndarray, target_mean: float, cap: float = 10.0) -> float:
    """Solve mean(min(cap, c*raw)) == target_mean by monotone bisection."""
    raw = np.asarray(raw, dtype=float)
    if np.any(raw <= 0) or not np.isfinite(raw).all():
        raise ValueError("All pre-normalization weights must be finite and positive")
    low, high = 0.0, 1.0
    while np.minimum(cap, high * raw).mean() < target_mean:
        high *= 2.0
        if high > 1e9:
            raise RuntimeError("Could not bracket normalization constant")
    for _ in range(100):
        mid = (low + high) / 2.0
        if np.minimum(cap, mid * raw).mean() < target_mean:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def augment_reddb_once(frame: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    """Freeze the original augmentation logic while retaining source-row identity."""
    rng = np.random.default_rng(SPLIT_SEED + 2025)
    records: list[dict] = []
    for source_index, row in frame.reset_index(drop=True).iterrows():
        encoded = str(row.encoded_smiles)
        records.append({"source_index": source_index, "encoded_smiles": encoded})
        if rng.random() > float(cfg.REDDB_AUG_PROB):
            continue
        for _ in range(int(cfg.REDDB_AUG_N)):
            randomized = randomize_smiles(encoded)
            if randomized is not None and cfg.MIN_LEN <= len(randomized) <= cfg.MAX_LEN - 2:
                records.append({"source_index": source_index, "encoded_smiles": randomized})
    augmented = pd.DataFrame(records)
    source = frame.reset_index(drop=True)[
        ["property_weight", "scaffold", "scaffold_count", "is_target", "inchi_key"]
    ].copy()
    source["source_index"] = np.arange(len(source))
    return augmented.merge(source, on="source_index", how="left", validate="many_to_one")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()

    weight_map = load_softft_weight_map(cfg)
    reddb_all, key_col = load_reddb_df_with_weights(cfg, weight_map)
    reddb_all = boost_weights_by_energy_fraction(cfg, reddb_all)
    train, valid = split_train_valid_with_w(
        reddb_all, cfg.VALID_RATIO_REDDB, cfg.SEED + 2
    )

    props = pd.read_csv(PROPERTIES)
    prop_key = next(c for c in ("inchiKey", "inchikey", "InChIKey") if c in props.columns)
    props[prop_key] = props[prop_key].astype(str).str.strip().str.upper()
    props = props[[prop_key, "Solvation_energy", "single_point_lumo_solution"]].rename(
        columns={
            prop_key: "inchi_key",
            "Solvation_energy": "solvation_energy_eV",
            "single_point_lumo_solution": "lumo_solution_eV",
        }
    )

    def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["inchi_key"] = out[key_col].astype(str).str.strip().str.upper()
        out = out.merge(props, on="inchi_key", how="left", validate="many_to_one")
        if out[["solvation_energy_eV", "lumo_solution_eV"]].isna().any().any():
            raise RuntimeError("Missing DFT properties after RedDB merge")
        out["property_weight"] = out[cfg.W_COL].astype(float)
        out["is_target"] = (
            (out.solvation_energy_eV <= TARGET_SOLV_MAX)
            & (out.lumo_solution_eV <= TARGET_LUMO_MAX)
        )
        out["scaffold"] = out[cfg.REDDB_SMILES_COL].astype(str).map(scaffold)
        out["encoded_smiles"] = out[cfg.REDDB_SMILES_COL].astype(str).map(
            lambda value: encode_special(value, cfg)
        )
        return out.reset_index(drop=True)

    train = prepare_frame(train)
    valid = prepare_frame(valid)
    scaffold_counts = train.scaffold.value_counts().to_dict()
    train["scaffold_count"] = train.scaffold.map(scaffold_counts).astype(int)
    valid["scaffold_count"] = valid.scaffold.map(scaffold_counts).fillna(0).astype(int)
    reddb_aug = augment_reddb_once(train, cfg)

    chembl_train, _ = read_chembl_train_valid(cfg)
    chembl_aug = augment_train_smiles(
        chembl_train,
        cfg,
        n_aug_per_mol=cfg.CHEMBL_AUG_N,
        aug_prob=cfg.CHEMBL_AUG_PROB,
        seed=SPLIT_SEED + 2026,
    )
    # Reuse the audited 30-token vocabulary expected by the pretrained
    # checkpoint. RDKit randomization can otherwise introduce a previously
    # absent ring digit and shift every subsequent token index.
    frozen_vocab = json.loads(FROZEN_VOCAB.read_text(encoding="utf-8"))
    stoi = {str(key): int(value) for key, value in frozen_vocab["stoi"].items()}
    itos = {int(key): str(value) for key, value in frozen_vocab["itos"].items()}
    allowed = set(stoi)
    reddb_before_vocab_filter = len(reddb_aug)
    chembl_before_vocab_filter = len(chembl_aug)
    reddb_aug = reddb_aug[
        reddb_aug.encoded_smiles.astype(str).map(lambda value: set(value).issubset(allowed))
    ].reset_index(drop=True)
    chembl_aug = [value for value in chembl_aug if set(str(value)).issubset(allowed)]

    group_counts = np.asarray(list(scaffold_counts.values()), dtype=float)
    median_count = float(np.median(group_counts))
    target_mean = float(reddb_aug.property_weight.mean())
    beta_stats = {}
    for beta in BETAS:
        factor = (reddb_aug.scaffold_count.to_numpy(float) / median_count) ** (-beta)
        raw = reddb_aug.property_weight.to_numpy(float) * factor
        constant = capped_mean_normalizer(raw, target_mean, cap=10.0)
        joint = np.minimum(10.0, constant * raw)
        beta_stats[str(beta)] = {
            "normalization_constant": constant,
            "mean_joint_weight": float(joint.mean()),
            "mean_property_weight": target_mean,
            "min_joint_weight": float(joint.min()),
            "max_joint_weight": float(joint.max()),
            "high_pool_rows": int((joint >= float(cfg.REDDB_W_THR)).sum()),
            "high_pool_fraction": float((joint >= float(cfg.REDDB_W_THR)).mean()),
        }

    train_path = ARTIFACTS / "reddb_train_base.csv.gz"
    valid_path = ARTIFACTS / "reddb_common_unit_valid.csv.gz"
    aug_path = ARTIFACTS / "reddb_train_augmented.csv.gz"
    chembl_path = ARTIFACTS / "chembl_train_augmented.csv.gz"
    vocab_path = ARTIFACTS / "vocab.json"
    ref_path = ARTIFACTS / "reddb_reference_scaffolds.json"
    train.to_csv(train_path, index=False, compression="gzip")
    valid.to_csv(valid_path, index=False, compression="gzip")
    reddb_aug.to_csv(aug_path, index=False, compression="gzip")
    pd.DataFrame({"encoded_smiles": chembl_aug}).to_csv(
        chembl_path, index=False, compression="gzip"
    )
    vocab_path.write_text(
        json.dumps({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}, indent=2),
        encoding="utf-8",
    )
    ref_path.write_text(
        json.dumps({"reddb_scaffolds": sorted(set(reddb_all[cfg.REDDB_SMILES_COL].map(scaffold)))}, indent=2),
        encoding="utf-8",
    )

    if len(stoi) != 30:
        raise RuntimeError(f"Unexpected vocabulary size {len(stoi)}; checkpoint expects 30")

    artifacts = [train_path, valid_path, aug_path, chembl_path, vocab_path, ref_path]
    manifest = {
        "created_for": "property weight x inverse scaffold-frequency beta frontier test",
        "split_seed": SPLIT_SEED,
        "betas": list(BETAS),
        "target_definition_eV": {
            "solvation_energy_max": TARGET_SOLV_MAX,
            "lumo_solution_max": TARGET_LUMO_MAX,
        },
        "weight_rule": "min(10, c*w_property*(n_g/median_g(n_g))**(-beta))",
        "normalization_scope": "frozen augmented RedDB fine-tuning rows",
        "median_scaffold_count": median_count,
        "n_scaffold_groups_train": len(scaffold_counts),
        "counts": {
            "reddb_all": len(reddb_all),
            "reddb_train": len(train),
            "reddb_train_target": int(train.is_target.sum()),
            "reddb_validation": len(valid),
            "reddb_validation_target": int(valid.is_target.sum()),
            "reddb_train_augmented": len(reddb_aug),
            "reddb_augmented_oov_rows_dropped": reddb_before_vocab_filter - len(reddb_aug),
            "chembl_train_augmented": len(chembl_aug),
            "chembl_augmented_oov_rows_dropped": chembl_before_vocab_filter - len(chembl_aug),
            "vocab_size": len(stoi),
        },
        "beta_weight_stats": beta_stats,
        "pretrain_checkpoint": str(PRETRAIN.relative_to(ROOT)),
        "pretrain_sha256": sha256(PRETRAIN),
        "artifacts": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in artifacts
        },
    }
    manifest_path = ARTIFACTS / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

