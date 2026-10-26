from __future__ import annotations

"""Confirmation rate of predictor-positive molecules on RedDB-new scaffolds.

Every new-scaffold quantity in the manuscript is a count of molecules the deployed
predictor called target-positive on cores absent from RedDB. Section 3.8 could not
resolve the arm ordering at that endpoint precisely because the conditional pass rate
inside new-scaffold records is the one factor in the decomposition that depends on the
predictor; supply does not. This panel measures that dependence directly.

The 50 molecules are all predictor-positive, drawn from the beta = 0.25 joint-weighting
arm across seeds 42/43/44 and the full decoding grid, on 45 distinct RedDB-new cores.
The design therefore estimates precision on RedDB-new scaffolds - the fraction of
predictor-positive new-scaffold molecules that DFT confirms - and the corresponding
false-positive rate. It does not estimate recall, so a confirmation rate below one
scales the reported new-scaffold counts downward but says nothing about molecules the
predictor rejected.

Runs on a partially complete panel, so it can be used while the calculations finish.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
Q1 = HERE.parent
PANEL = (Q1 / "beta_scaffold_frontier_260815" / "dft_validation"
         / "beta0p25_reddb_new_scaffold_dft50.csv")
OUT = HERE / "results"

SOLV_MAX, LUMO_MAX = -1.5, -4.0

# Published new-scaffold quantities that a precision below one rescales. Section 3.7
# for the deployed pipeline, Section 3.9 for the budget comparison.
PUBLISHED = {
    "Section 3.7, continuous, TCN: distinct new-scaffold target molecules": 121,
    "Section 3.7, hard, TCN: distinct new-scaffold target molecules": 184,
    "Section 3.7, continuous, TCN: target-bearing new cores": 17,
    "Section 3.7, hard, TCN: target-bearing new cores": 26,
    "Section 3.9, equal at 90,000 attempts, TCN: new-scaffold target molecules": 146,
    "Section 3.9, equal at 90,000 attempts, TCN: target-bearing new cores": 20,
}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def summarize(frame: pd.DataFrame, label: str) -> dict | None:
    if frame.empty:
        return None
    confirmed = int(frame.dft_target_pass.sum())
    n = int(len(frame))
    low, high = wilson(confirmed, n)
    return {
        "subset": label, "n": n, "confirmed": confirmed,
        "confirmation_rate": confirmed / n,
        "ci95_low": low, "ci95_high": high,
        "solv_mae_eV": float((frame.dft_solvation_energy_eV - frame.solv_mean_eV).abs().mean()),
        "solv_bias_eV": float((frame.solv_mean_eV - frame.dft_solvation_energy_eV).mean()),
        "lumo_mae_eV": float((frame.dft_solution_lumo_eV - frame.lumo_mean_eV).abs().mean()),
        "lumo_bias_eV": float((frame.lumo_mean_eV - frame.dft_solution_lumo_eV).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default=str(PANEL),
                        help="panel CSV with the dft_* columns filled in")
    parser.add_argument("--write-back", action="store_true",
                        help="recompute dft_target_pass and predictor_dft_agreement in the "
                             "panel file from the reported energies")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.panel)
    done = panel[panel.dft_solvation_energy_eV.notna()
                 & panel.dft_solution_lumo_eV.notna()].copy()
    print(f"panel {Path(args.panel).name}: {len(done)} of {len(panel)} molecules complete")
    if done.empty:
        print("\nNo DFT values yet. Fill dft_solvation_energy_eV and dft_solution_lumo_eV "
              "(both in eV) and rerun.")
        return

    # The pass flag is recomputed here rather than trusted, so the target rule applied to
    # DFT is the same one applied to the predictor everywhere else.
    done["dft_target_pass"] = (done.dft_solvation_energy_eV.le(SOLV_MAX)
                               & done.dft_solution_lumo_eV.le(LUMO_MAX))
    done["predictor_dft_agreement"] = done.dft_target_pass  # every row is predictor-positive
    if args.write_back:
        panel.loc[done.index, "dft_target_pass"] = done.dft_target_pass
        panel.loc[done.index, "predictor_dft_agreement"] = done.predictor_dft_agreement
        panel.loc[done.index, "dft_status"] = "complete"
        panel.to_csv(args.panel, index=False)
        print(f"wrote flags back into {args.panel}")

    rows = [summarize(done, "all predictor-positive")]
    for stratum, sub in done.groupby("selection_stratum"):
        rows.append(summarize(sub, f"stratum: {stratum}"))
    tertile = pd.qcut(done.joint_margin_mean_eV, 3,
                      labels=["narrow margin", "middle margin", "wide margin"],
                      duplicates="drop")
    for name, sub in done.groupby(tertile, observed=True):
        rows.append(summarize(sub, f"predictor margin: {name}"))
    rows = [r for r in rows if r]

    print("\n" + "=" * 96)
    print("DFT confirmation of predictor-positive molecules on RedDB-new scaffolds")
    print("=" * 96)
    print(f"{'subset':34s} {'n':>4s} {'conf':>5s} {'rate':>7s} {'95% CI':>16s} "
          f"{'solvMAE':>8s} {'lumoMAE':>8s}")
    for r in rows:
        print(f"{r['subset']:34s} {r['n']:4d} {r['confirmed']:5d} "
              f"{100*r['confirmation_rate']:6.1f}% "
              f"[{100*r['ci95_low']:5.1f},{100*r['ci95_high']:5.1f}] "
              f"{r['solv_mae_eV']:8.3f} {r['lumo_mae_eV']:8.3f}")

    overall = rows[0]
    precision, low, high = (overall["confirmation_rate"], overall["ci95_low"],
                            overall["ci95_high"])

    print("\n" + "=" * 96)
    print("What this rescales, if precision transfers from the beta arm to the reported arms")
    print("=" * 96)
    scaled = []
    for label, value in PUBLISHED.items():
        scaled.append({"quantity": label, "reported": value,
                       "confirmed_estimate": value * precision,
                       "ci95_low": value * low, "ci95_high": value * high})
        print(f"  {label:68s} {value:5d} -> {value * precision:6.1f} "
              f"[{value * low:.1f}, {value * high:.1f}]")

    n_scaffolds = int(done.scaffold.nunique())
    print(f"\n  Cores represented in the completed panel: {n_scaffolds}")
    print("  Recall is not estimated by this design, so these are corrections to the "
          "predictor-positive counts only;\n  molecules the predictor rejected are "
          "unmeasured and the true totals could be higher.")

    (OUT / "panel_confirmation.json").write_text(json.dumps({
        "panel": Path(args.panel).name,
        "n_complete": int(len(done)), "n_panel": int(len(panel)),
        "design": "all molecules predictor-positive; estimates precision and false-positive "
                  "rate on RedDB-new scaffolds, not recall",
        "target_rule": {"solvation_energy_max_eV": SOLV_MAX, "lumo_max_eV": LUMO_MAX},
        "by_subset": rows,
        "rescaled_published_quantities": scaled,
        "n_distinct_new_cores_in_panel": n_scaffolds,
    }, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "panel_confirmation_by_subset.csv", index=False)
    done.to_csv(OUT / "panel_completed_molecules.csv", index=False)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
