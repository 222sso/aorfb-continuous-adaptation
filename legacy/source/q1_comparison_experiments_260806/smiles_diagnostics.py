from __future__ import annotations

import re
from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


RING_TOKEN = re.compile(r"%\d{2}|\d")


def _outside_brackets(smiles: str) -> list[tuple[int, str]]:
    out = []
    depth = 0
    i = 0
    while i < len(smiles):
        ch = smiles[i]
        if ch == "[":
            depth += 1
        elif ch == "]" and depth:
            depth -= 1
        elif depth == 0:
            if ch == "%" and i + 2 < len(smiles) and smiles[i + 1 : i + 3].isdigit():
                out.append((i, smiles[i : i + 3]))
                i += 2
            elif ch.isdigit():
                out.append((i, ch))
        i += 1
    return out


def structural_syntax(smiles: str) -> dict:
    stack = []
    first_error = None
    in_bracket = 0
    for i, ch in enumerate(smiles):
        if ch == "[":
            in_bracket += 1
        elif ch == "]" and in_bracket:
            in_bracket -= 1
        elif in_bracket == 0 and ch == "(":
            stack.append(i)
        elif in_bracket == 0 and ch == ")":
            if stack:
                stack.pop()
            elif first_error is None:
                first_error = i
    paren_fail = bool(stack) or first_error is not None
    if stack and first_error is None:
        first_error = stack[0]

    rings: dict[str, list[int]] = {}
    for position, token in _outside_brackets(smiles):
        rings.setdefault(token, []).append(position)
    unmatched_ring_positions = [positions[-1] for positions in rings.values() if len(positions) % 2]
    ring_fail = bool(unmatched_ring_positions)
    if unmatched_ring_positions:
        ring_first = min(unmatched_ring_positions)
        first_error = ring_first if first_error is None else min(first_error, ring_first)

    norm_position = first_error / max(1, len(smiles) - 1) if first_error is not None else np.nan
    return {
        "parenthesis_fail": paren_fail,
        "ring_closure_fail": ring_fail,
        "first_detectable_structural_error_position": first_error,
        "first_detectable_structural_error_fraction": norm_position,
    }


def _scaffold(canonical_smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or "[ACYCLIC]"


def summarize_diagnostics(smiles: list[str], scored_rows=None) -> dict:
    syntax = [structural_syntax(s) for s in smiles]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    invalid = np.array([m is None for m in mols], dtype=bool)
    paren = np.array([x["parenthesis_fail"] for x in syntax], dtype=bool)
    ring = np.array([x["ring_closure_fail"] for x in syntax], dtype=bool)
    positions = np.asarray(
        [x["first_detectable_structural_error_fraction"] for x in syntax], dtype=float
    )
    detectable_invalid = invalid & np.isfinite(positions)

    bins = {"early_0_1_3": 0, "middle_1_3_2_3": 0, "late_2_3_1": 0}
    for value in positions[detectable_invalid]:
        if value < 1 / 3:
            bins["early_0_1_3"] += 1
        elif value < 2 / 3:
            bins["middle_1_3_2_3"] += 1
        else:
            bins["late_2_3_1"] += 1

    valid_canons = [Chem.MolToSmiles(m, canonical=True, isomericSmiles=False) for m in mols if m]
    valid_scaffolds = [_scaffold(s) for s in set(valid_canons)]
    valid_scaffolds = [s for s in valid_scaffolds if s]

    result = {
        "n_generated": len(smiles),
        "n_invalid": int(invalid.sum()),
        "parenthesis_failure_count": int(paren.sum()),
        "parenthesis_failure_pct_total": float(100 * paren.mean()) if len(paren) else np.nan,
        "ring_closure_failure_count": int(ring.sum()),
        "ring_closure_failure_pct_total": float(100 * ring.mean()) if len(ring) else np.nan,
        "detectable_structural_failure_count_among_invalid": int(detectable_invalid.sum()),
        "detectable_structural_failure_pct_among_invalid": (
            float(100 * detectable_invalid.sum() / invalid.sum()) if invalid.sum() else np.nan
        ),
        "median_first_error_fraction": (
            float(np.nanmedian(positions[detectable_invalid])) if detectable_invalid.any() else np.nan
        ),
        "error_position_bins": bins,
        "n_unique_valid_scaffolds": len(set(valid_scaffolds)),
        "top_valid_scaffold_share_pct": (
            float(100 * Counter(valid_scaffolds).most_common(1)[0][1] / len(valid_scaffolds))
            if valid_scaffolds else np.nan
        ),
    }

    if scored_rows is not None and len(scored_rows):
        hits = scored_rows[
            (scored_rows.rdkit_valid == 1)
            & scored_rows.solv_mean_eV.notna()
            & scored_rows.lumo_mean_eV.notna()
            & (scored_rows.solv_mean_eV <= -1.5)
            & (scored_rows.lumo_mean_eV <= -4.0)
        ]
        hit_scaffolds = [_scaffold(s) for s in set(hits.canon.dropna().astype(str))]
        hit_scaffolds = [s for s in hit_scaffolds if s]
        result["n_unique_target_scaffolds"] = len(set(hit_scaffolds))
        result["top_unique_target_scaffold_share_pct"] = (
            float(100 * Counter(hit_scaffolds).most_common(1)[0][1] / len(hit_scaffolds))
            if hit_scaffolds else np.nan
        )
    return result
