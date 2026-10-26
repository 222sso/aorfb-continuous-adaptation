# -*- coding: utf-8 -*-
"""
Re-select the candidate shortlist on the v2 screening model and the 260311 run.

Two provenance defects are corrected here.

1. WRONG SOURCE RUN. The published shortlist was ranked over
   `tcnlstm_transfer_outputs_260303/pred_eval/samples_scored_all.csv` — an older
   experiment with 500 generations per backbone — while Table 1, the weighting
   sweep and the front-end ablation all come from the 260311 run with 9,000
   generations per backbone. The shortlist therefore described a different
   experiment from the rest of the manuscript. Reproducing the published ranking
   confirms the source exactly (combined ranks 19, 25, 42, 46, 50, 53, 55, 60 and
   all eight structures). Selection now runs on the 260311 continuous-weight arm.

2. STEREOCHEMICAL DUPLICATES. The original rule deduplicated on isomeric
   canonical SMILES, so two E/Z forms of one constitutional isomer can both enter
   the shortlist. In the larger 260311 pool they do: two entries differ only in
   double-bond stereo notation and carry identical predicted properties, because
   the Morgan fingerprint used by the screening model does not encode stereo and
   cannot distinguish them. Ranking now deduplicates on the stereo-free
   structure, so the shortlist holds eight distinct constitutional structures.

Ranking rule (otherwise unchanged, per 04_METHODS_FACTS.md):
    rank_solv = ascending solv_mean_eV
    rank_lumo = ascending lumo_mean_eV
    combined_rank = rank_solv + rank_lumo
    ties broken by lumo_mean_eV, solv_mean_eV, lumo_std_eV, solv_std_eV

Run:
  python predictor_v2_260811/code/select_candidates_v2.py
"""
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
SCORED = BASE / "tcnlstm_transfer_outputs_260311" / "exp_baseline" / "pred_eval_v2" / "samples_scored_all.csv"
OUT = BASE / "Results" / "candidate_lists" / "top8_candidates_v2_260311.csv"
TOP_N = 8


def flat_smiles(s: str):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else None


def main():
    d = pd.read_csv(SCORED)
    d = d[(d.rdkit_valid == 1) & d.canon.notna()
          & d.solv_mean_eV.notna() & d.lumo_mean_eV.notna()].copy()
    d["flat"] = [flat_smiles(s) for s in d.canon]
    d = d.dropna(subset=["flat"])

    n_iso = d.canon.nunique()
    d = d.drop_duplicates("flat").copy()
    print(f"pool: {len(d):,} distinct constitutional structures "
          f"({n_iso:,} before collapsing stereoisomers)")

    d["combined_rank"] = d.solv_mean_eV.rank(method="min") + d.lumo_mean_eV.rank(method="min")
    t = d.sort_values(["combined_rank", "lumo_mean_eV", "solv_mean_eV",
                       "lumo_std_eV", "solv_std_eV"]).head(TOP_N).reset_index(drop=True)
    t["candidate_id"] = [f"C{i+1}" for i in range(len(t))]
    t["target_region_pass"] = (t.solv_mean_eV <= -1.5) & (t.lumo_mean_eV <= -4.0)
    t["target_region_rule"] = "solv_mean_eV <= -1.5 and lumo_mean_eV <= -4.0"

    assert t.flat.nunique() == TOP_N, "shortlist still contains duplicate structures"
    assert bool(t.target_region_pass.all()), "a shortlisted candidate fails the target region"

    cols = ["candidate_id", "canon", "flat", "solv_mean_eV", "solv_std_eV",
            "lumo_mean_eV", "lumo_std_eV", "combined_rank", "model",
            "temperature", "top_p", "target_region_pass", "target_region_rule"]
    t[cols].to_csv(OUT, index=False, encoding="utf-8-sig")

    print(t[["candidate_id", "solv_mean_eV", "solv_std_eV", "lumo_mean_eV",
             "lumo_std_eV", "combined_rank", "model"]].round(3).to_string(index=False))
    print(f"\nΔG_solv range {t.solv_mean_eV.min():.3f} to {t.solv_mean_eV.max():.3f} eV")
    print(f"E_LUMO   range {t.lumo_mean_eV.min():.3f} to {t.lumo_mean_eV.max():.3f} eV")
    print(f"mean ensemble SD: solv {t.solv_std_eV.mean():.3f} eV, lumo {t.lumo_std_eV.mean():.3f} eV")
    print(f"backbones represented: {sorted(t.model.unique())}")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
