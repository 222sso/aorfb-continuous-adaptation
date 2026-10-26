from __future__ import annotations
"""RedDB in the schema of this study's pipeline, so it is measured with the same
instrument as ZINC and QM9.

The published RedDB three-way used a separate codebase with a RedDB-specific
vocabulary and a per-arm pretraining run. Re-running it here on the shared union
vocabulary and the single shared ChEMBL pretraining checkpoint makes the RedDB numbers
directly comparable with the other corpora; it is an independent reproduction, not a
continuation of the published table, and the two are reported separately.
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")


def murcko(smiles: str) -> str:
    """Compute the scaffold here rather than reusing a stored column.

    The stored murcko_scaffold column in descriptors.csv.gz carries double-bond
    stereo marks (/ and \\) that generated scaffolds do not, so string comparison
    against it silently classifies known cores as new.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACTS = HERE / "artifacts"
SRC = ROOT / "Results" / "target_region_analysis" / "descriptors.csv.gz"
sys.path.insert(0, str(ROOT))
from ablation.config import CFG          # noqa: E402
from ablation.data import encode_special  # noqa: E402

NAME = "reddb"
VALID_RATIO = 0.01
SPLIT_SEED = 44
SOLV_MAX_EV, LUMO_MAX_EV = -1.5, -4.0

def main() -> None:
    cfg = CFG()
    d = pd.read_csv(SRC)
    global SCAFFOLDS
    with Pool(8) as pool:
        SCAFFOLDS = pool.map(murcko, d.canonical_smiles.astype(str).tolist(), chunksize=500)
    frame = pd.DataFrame({
        "canonical_smiles": d.canonical_smiles.astype(str),
        "scaffold": SCAFFOLDS,
        "Solvation_energy_eV": d.Solvation_energy.astype(float),
        "lumo_eV": d.single_point_lumo_solution.astype(float),
        "q_primary": d.q_solv.astype(float),
        "q_secondary": d.q_orb.astype(float),
        "score": d.z.astype(float),
        "property_weight": d.w.astype(float),
        "heavy_atoms": d.heavy_atoms.astype(int),
    })
    frame["is_target"] = (frame.Solvation_energy_eV <= SOLV_MAX_EV) & (frame.lumo_eV <= LUMO_MAX_EV)
    frame["encoded_smiles"] = [encode_special(s, cfg) for s in frame.canonical_smiles]
    length = frame.encoded_smiles.str.len()
    keep = (length >= cfg.MIN_LEN) & (length + 2 <= cfg.MAX_LEN)
    dropped = int((~keep).sum())
    frame = frame[keep].reset_index(drop=True)

    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(frame))
    split = np.array(["train"] * len(frame), dtype=object)
    split[order[: max(1, int(len(frame) * VALID_RATIO))]] = "valid"
    frame["split"] = split
    frame.to_csv(ARTIFACTS / f"{NAME}.csv.gz", index=False)

    train = frame[frame.split == "train"]
    manifest_path = ARTIFACTS / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["target_rules"][NAME] = {
        "primary_column": "Solvation_energy_eV", "primary_direction": "le",
        "primary_threshold": SOLV_MAX_EV,
        "secondary_column": "lumo_eV", "secondary_direction": "le",
        "secondary_threshold": LUMO_MAX_EV,
    }
    meta = {
        "source": str(SRC.relative_to(ROOT)),
        "n_molecules": int(len(frame)), "n_dropped_by_length": dropped,
        "n_scaffolds": int(frame.scaffold.nunique()),
        "molecules_per_scaffold": float(len(frame) / frame.scaffold.nunique()),
        "joint_target_pct": float(100 * frame.is_target.mean()),
        "n_target": int(frame.is_target.sum()),
        "n_target_scaffolds": int(frame.loc[frame.is_target, "scaffold"].nunique()),
        "n_train": int(len(train)), "n_valid": int((frame.split == "valid").sum()),
        "n_train_target_hard_arm": int(train.is_target.sum()),
        "high_pool_pct": float(100 * (frame.property_weight >= 1.2).mean()),
        "weights": "reused verbatim from the published SoftFT weight table (column w)",
        "judgement": "DFT endpoints; generated molecules judged by a predictor",
        "note": "independent reproduction on the shared union vocabulary and shared "
                "ChEMBL pretraining checkpoint; not a continuation of the published table",
    }
    manifest.setdefault("reproduction_variants", {})[NAME] = meta
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{NAME}] n={meta['n_molecules']:,} (dropped {dropped}) "
          f"scaffolds={meta['n_scaffolds']} target={meta['joint_target_pct']:.2f}% "
          f"({meta['n_target']:,} on {meta['n_target_scaffolds']} scaffolds) "
          f"hard_arm={meta['n_train_target_hard_arm']:,} high_pool={meta['high_pool_pct']:.2f}%")

if __name__ == "__main__":
    main()
