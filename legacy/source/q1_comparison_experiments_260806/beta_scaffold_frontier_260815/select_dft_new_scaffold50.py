from __future__ import annotations

"""Select a scaffold-diverse beta=0.25 panel for independent DFT validation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"
OUTDIR = HERE / "dft_validation"
TARGET_SOLV_MAX = -1.5
TARGET_LUMO_MAX = -4.0
MAX_MW = 600.0
MAX_HEAVY_ATOMS = 50
MAX_ROTATABLE_BONDS = 15
N_PANEL = 50

RDLogger.DisableLog("rdApp.*")


def scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or "[ACYCLIC]"


def load_chembl_scaffolds() -> set[str]:
    path = ROOT / "inputs" / "sampled_60000_chembl_smiles.csv"
    frame = pd.read_csv(path)
    column = next(value for value in ("smiles", "SMILES", "canonical_smiles") if value in frame.columns)
    return {value for value in frame[column].astype(str).map(scaffold) if value}


def descriptor_record(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES: {smiles}")
    return {
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": Descriptors.MolWt(mol),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(mol),
        "radical_electron_count": sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()),
        "fragment_count": len(Chem.GetMolFrags(mol)),
        "rotatable_bond_count": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "hbond_donor_count": Lipinski.NumHDonors(mol),
        "hbond_acceptor_count": Lipinski.NumHAcceptors(mol),
        "tpsa_A2": rdMolDescriptors.CalcTPSA(mol),
    }


def source_summary(group: pd.DataFrame) -> dict:
    settings = sorted(
        {
            f"seed{int(row.training_seed)}:T{row.temperature:.2f}:p{row.top_p:.2f}"
            for row in group.itertuples()
        }
    )
    return {
        "generation_occurrence_count": len(group),
        "generation_seed_count": group.training_seed.nunique(),
        "generation_seeds": ";".join(str(int(value)) for value in sorted(group.training_seed.unique())),
        "generation_settings": ";".join(settings),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RESULTS.glob("seed_*/beta_0p25/samples_scored_v2.csv.gz"))
    if len(files) != 3:
        raise RuntimeError(f"Expected three beta=0.25 score files, found {len(files)}")
    all_rows = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    reddb_scaffolds = set(
        json.loads((ARTIFACTS / "reddb_reference_scaffolds.json").read_text(encoding="utf-8"))[
            "reddb_scaffolds"
        ]
    )
    chembl_scaffolds = load_chembl_scaffolds()

    pool = all_rows[
        all_rows.rdkit_valid.astype(bool)
        & all_rows.is_target.astype(bool)
        & ~all_rows.scaffold.isin(reddb_scaffolds)
    ].copy()
    grouped = []
    for canon, group in pool.groupby("canon", sort=False):
        row = group.iloc[0].to_dict()
        row.update(source_summary(group))
        row.update(descriptor_record(str(canon)))
        grouped.append(row)
    candidates = pd.DataFrame(grouped)
    candidates["scaffold_in_reddb"] = candidates.scaffold.isin(reddb_scaffolds)
    candidates["scaffold_in_sampled_chembl"] = candidates.scaffold.isin(chembl_scaffolds)
    candidates["solv_margin_mean_eV"] = TARGET_SOLV_MAX - candidates.solv_mean_eV
    candidates["lumo_margin_mean_eV"] = TARGET_LUMO_MAX - candidates.lumo_mean_eV
    candidates["joint_margin_mean_eV"] = candidates[
        ["solv_margin_mean_eV", "lumo_margin_mean_eV"]
    ].min(axis=1)
    candidates["solv_margin_2sd_eV"] = TARGET_SOLV_MAX - (
        candidates.solv_mean_eV + 2.0 * candidates.solv_std_eV
    )
    candidates["lumo_margin_2sd_eV"] = TARGET_LUMO_MAX - (
        candidates.lumo_mean_eV + 2.0 * candidates.lumo_std_eV
    )
    candidates["joint_margin_2sd_eV"] = candidates[
        ["solv_margin_2sd_eV", "lumo_margin_2sd_eV"]
    ].min(axis=1)
    candidates["predictor_target_mean"] = True
    candidates["predictor_target_2sd"] = (
        (candidates.solv_margin_2sd_eV >= 0) & (candidates.lumo_margin_2sd_eV >= 0)
    )
    candidates["eligible_for_dft_panel"] = (
        (candidates.molecular_weight <= MAX_MW)
        & (candidates.heavy_atom_count <= MAX_HEAVY_ATOMS)
        & (candidates.rotatable_bond_count <= MAX_ROTATABLE_BONDS)
        & (candidates.formal_charge == 0)
        & (candidates.radical_electron_count == 0)
        & (candidates.fragment_count == 1)
    )
    eligible = candidates[candidates.eligible_for_dft_panel].copy()

    # One robust representative from every eligible new scaffold.
    anchors = (
        eligible.sort_values(
            ["scaffold", "joint_margin_2sd_eV", "molecular_weight"],
            ascending=[True, False, True],
        )
        .groupby("scaffold", as_index=False, sort=False)
        .head(1)
        .copy()
    )
    anchors["selection_stratum"] = "new_scaffold_anchor"

    # Five additional near-boundary positives, each from a different scaffold.
    remaining = eligible[~eligible.canon.isin(anchors.canon)].copy()
    remaining["boundary_distance_eV"] = remaining.joint_margin_mean_eV.abs()
    boundary = (
        remaining.sort_values(
            ["boundary_distance_eV", "molecular_weight"], ascending=[True, True]
        )
        .groupby("scaffold", as_index=False, sort=False)
        .head(1)
        .head(N_PANEL - len(anchors))
        .copy()
    )
    boundary["selection_stratum"] = "near_boundary_positive"
    selected = pd.concat([anchors, boundary], ignore_index=True)
    if len(selected) != N_PANEL:
        raise RuntimeError(
            f"Selection returned {len(selected)} molecules ({len(anchors)} anchors); expected {N_PANEL}"
        )
    if selected.canon.nunique() != N_PANEL:
        raise RuntimeError("Selected panel contains duplicate canonical molecules")
    if selected.scaffold_in_reddb.any():
        raise RuntimeError("A selected scaffold occurs in RedDB")

    selected = selected.sort_values(
        ["selection_stratum", "joint_margin_2sd_eV"], ascending=[True, False]
    ).reset_index(drop=True)
    selected.insert(0, "candidate_id", [f"B025_NEW_DFT_{index:03d}" for index in range(1, 51)])
    selected.insert(
        2,
        "scaffold_id",
        selected.scaffold.map(lambda value: "SCF_" + hashlib.sha1(value.encode()).hexdigest()[:12]),
    )
    selected["dft_solvation_energy_eV"] = np.nan
    selected["dft_solution_lumo_eV"] = np.nan
    selected["dft_target_pass"] = pd.NA
    selected["predictor_dft_agreement"] = pd.NA
    selected["dft_status"] = "not_started"

    columns = [
        "candidate_id",
        "canon",
        "scaffold_id",
        "scaffold",
        "selection_stratum",
        "predictor_target_mean",
        "predictor_target_2sd",
        "solv_mean_eV",
        "solv_std_eV",
        "lumo_mean_eV",
        "lumo_std_eV",
        "solv_margin_mean_eV",
        "lumo_margin_mean_eV",
        "joint_margin_mean_eV",
        "joint_margin_2sd_eV",
        "molecular_formula",
        "molecular_weight",
        "heavy_atom_count",
        "formal_charge",
        "radical_electron_count",
        "fragment_count",
        "rotatable_bond_count",
        "ring_count",
        "hbond_donor_count",
        "hbond_acceptor_count",
        "tpsa_A2",
        "scaffold_in_reddb",
        "scaffold_in_sampled_chembl",
        "generation_occurrence_count",
        "generation_seed_count",
        "generation_seeds",
        "generation_settings",
        "dft_solvation_energy_eV",
        "dft_solution_lumo_eV",
        "dft_target_pass",
        "predictor_dft_agreement",
        "dft_status",
    ]
    selected = selected[columns]
    csv_path = OUTDIR / "beta0p25_reddb_new_scaffold_dft50.csv"
    smi_path = OUTDIR / "beta0p25_reddb_new_scaffold_dft50.smi"
    selected.to_csv(csv_path, index=False, encoding="utf-8-sig")
    smi_path.write_text(
        "\n".join(f"{row.canon}\t{row.candidate_id}" for row in selected.itertuples()) + "\n",
        encoding="utf-8",
    )

    summary = {
        "source": "beta=0.25, seeds 42/43/44, full 3x3 decoding grid",
        "predictor": "three-seed LightGBM/Morgan ensemble (predictor_v2_260811)",
        "target_definition_eV": {
            "solvation_energy_max": TARGET_SOLV_MAX,
            "solution_lumo_max": TARGET_LUMO_MAX,
        },
        "candidate_pool": {
            "generated_records": len(all_rows),
            "reddb_new_predicted_target_records": len(pool),
            "unique_reddb_new_predicted_target_molecules": candidates.canon.nunique(),
            "unique_reddb_new_predicted_target_scaffolds": candidates.scaffold.nunique(),
            "eligible_unique_molecules": len(eligible),
            "eligible_unique_scaffolds": eligible.scaffold.nunique(),
        },
        "panel": {
            "n_molecules": len(selected),
            "n_distinct_reddb_new_scaffolds": selected.scaffold.nunique(),
            "n_scaffolds_absent_from_sampled_chembl": int(
                selected.loc[~selected.scaffold_in_sampled_chembl, "scaffold"].nunique()
            ),
            "n_molecules_on_scaffolds_absent_from_sampled_chembl": int(
                (~selected.scaffold_in_sampled_chembl).sum()
            ),
            "n_scaffold_anchors": int((selected.selection_stratum == "new_scaffold_anchor").sum()),
            "n_near_boundary_positives": int((selected.selection_stratum == "near_boundary_positive").sum()),
            "n_predictor_target_2sd": int(selected.predictor_target_2sd.sum()),
        },
        "eligibility": {
            "maximum_molecular_weight": MAX_MW,
            "maximum_heavy_atoms": MAX_HEAVY_ATOMS,
            "maximum_rotatable_bonds": MAX_ROTATABLE_BONDS,
            "formal_charge": 0,
            "radical_electrons": 0,
            "fragment_count": 1,
        },
        "interpretation": (
            "All 50 are predictor-positive. The panel estimates positive confirmation rate and false-positive "
            "rate on RedDB-new scaffolds; it does not estimate sensitivity or specificity."
        ),
    }
    (OUTDIR / "beta0p25_reddb_new_scaffold_dft50_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(selected[[
        "candidate_id", "canon", "scaffold_id", "selection_stratum",
        "solv_mean_eV", "lumo_mean_eV", "molecular_weight",
        "scaffold_in_sampled_chembl"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()

