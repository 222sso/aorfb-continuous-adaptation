from __future__ import annotations

"""Leakage-controlled diagnostics for the weighting manuscript revision.

This script does not modify any training code or checkpoint. It performs two
post hoc, pre-specified diagnostics:

1. Paired, length-normalized likelihood on the RedDB validation molecules that
   were excluded from both the continuous-weight and equal-weight fine-tuning
   runs.
2. Motif alignment between the DFT target region and continuous-weight versus
   equal-weight generations at the same backbone and decoding condition.

All molecule-level and summary outputs are written to a new analysis folder.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
OUT = EXP / "results" / "structure_transfer_260812"
sys.path.insert(0, str(ROOT))

from ablation.config import CFG  # noqa: E402
from ablation.data import (  # noqa: E402
    augment_train_smiles,
    augment_train_smiles_with_w,
    boost_weights_by_energy_fraction,
    detokenize_to_plain_smiles,
    encode_special,
    load_reddb_df_with_weights,
    load_softft_weight_map,
    read_chembl_train_valid,
    read_reddb_train_valid_with_weights,
    split_train_valid_with_w,
)
from ablation.models import build_C_TCN_LSTM  # noqa: E402
from ablation.vocab import build_char_vocab, encode_smiles_list  # noqa: E402


RDLogger.DisableLog("rdApp.*")
CONFIG = ROOT / "tcnlstm_transfer_outputs_260311" / "exp_baseline" / "config_used.json"
WEIGHT_CSV = ROOT / "dist_reports" / "softft_v3" / "softft_scores_weights.csv"
SOFT_CKPT = ROOT / "tcnlstm_transfer_outputs_260311" / "exp_baseline" / (
    'ckpt_finetune_C_TCN_LSTM.weights.h5'
)
EQUAL_CKPT = ROOT / "tcnlstm_transfer_outputs_260311" / "exp_noweight" / (
    'ckpt_finetune_C_TCN_LSTM.weights.h5'
)
GENERATED = {
    "continuous_weight": ROOT / "tcnlstm_transfer_outputs_260311" / "exp_baseline" / "pred_eval" / "samples_scored_all.csv",
    "equal_weight": ROOT / "tcnlstm_transfer_outputs_260311" / "exp_noweight" / "pred_eval" / "samples_scored_all.csv",
}
TARGET_SOLV_MAX = -1.5
TARGET_LUMO_MAX = -4.0
MODEL = "C_TCN_LSTM"
TEMPERATURE = 0.90
TOP_P = 0.95
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 260812


MOTIF_SMARTS = {
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "sulfonic_acid": "[SX4](=O)(=O)[OX2H1]",
    "carbonyl": "[CX3]=[OX1]",
    "phenolic_hydroxyl": "[c][OX2H1]",
    "aliphatic_hydroxyl": "[C;!$(C=O)][OX2H1]",
    "primary_or_secondary_amine": "[NX3;H1,H2;!$(N[C,S,P]=O)]",
    "aromatic_nitrogen": "[nH0]",
    "nitrile": "[C]#N",
    "halogen": "[F,Cl,Br,I]",
}
MOTIFS = {name: Chem.MolFromSmarts(smarts) for name, smarts in MOTIF_SMARTS.items()}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cfg() -> CFG:
    cfg = CFG()
    payload = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    for key, value in payload.items():
        setattr(cfg, key, value)
    for key in ("CHEMBL_CSV", "REDDB_CSV", "WEIGHT_CSV", "PRED_DIR", "NOVEL_REF_PATH"):
        value = getattr(cfg, key, None)
        if value and not Path(value).is_absolute():
            setattr(cfg, key, str((ROOT / value).resolve()))
    return cfg


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "[INVALID]"
    value = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return value or "[ACYCLIC]"


def prepare_exact_split_and_vocab(cfg: CFG):
    # The original pipeline attaches weights, filters, then shuffles with
    # cfg.SEED + 2. Repeating those calls recovers the identical held-out rows.
    w_map = load_softft_weight_map(cfg)
    reddb_df, key_col = load_reddb_df_with_weights(cfg, w_map)
    reddb_df = boost_weights_by_energy_fraction(cfg, reddb_df)
    train_df, valid_df = split_train_valid_with_w(
        reddb_df, cfg.VALID_RATIO_REDDB, cfg.SEED + 2
    )

    props = pd.read_csv(WEIGHT_CSV)
    prop_key = next(c for c in ("inchiKey", "inchikey", "InChIKey") if c in props.columns)
    props[prop_key] = props[prop_key].astype(str).str.strip().str.upper()
    valid_df[key_col] = valid_df[key_col].astype(str).str.strip().str.upper()
    valid_df = valid_df.merge(
        props[[prop_key, "Solvation_energy", "single_point_lumo_solution"]],
        left_on=key_col,
        right_on=prop_key,
        how="left",
        suffixes=("_raw", "_eV"),
        validate="many_to_one",
    )
    valid_df["is_target"] = (
        (valid_df["Solvation_energy_eV"] <= TARGET_SOLV_MAX)
        & (valid_df["single_point_lumo_solution_eV"] <= TARGET_LUMO_MAX)
    )

    # Rebuild the character vocabulary exactly as in the original run.
    chembl_train, _ = read_chembl_train_valid(cfg)
    reddb_train, _, reddb_train_w, _, _, _ = read_reddb_train_valid_with_weights(cfg)
    chembl_train = augment_train_smiles(
        chembl_train, cfg, n_aug_per_mol=cfg.CHEMBL_AUG_N,
        aug_prob=cfg.CHEMBL_AUG_PROB, seed=cfg.SEED + 2026,
    )
    reddb_train, reddb_train_w = augment_train_smiles_with_w(
        reddb_train, reddb_train_w, cfg, n_aug_per_mol=cfg.REDDB_AUG_N,
        aug_prob=cfg.REDDB_AUG_PROB, seed=cfg.SEED + 2025,
    )
    # Match the original runner: vocabulary precedes over-length dropping.
    # A rare character in an augmented long record creates the 31st token.
    chembl_base, _ = read_chembl_train_valid(cfg)
    reddb_base, _, _, _, _, _ = read_reddb_train_valid_with_weights(cfg)
    stoi_equal, _ = build_char_vocab(chembl_base + reddb_base + ["5", "6"], cfg)
    stoi_soft, _ = build_char_vocab(chembl_base + reddb_base + ["5", "6", "7"], cfg)
    return train_df, valid_df, {"soft": stoi_equal, "equal": stoi_soft}


def model_for_checkpoint(cfg: CFG, vocab_size: int, checkpoint: Path):
    model = build_C_TCN_LSTM(
        vocab_size, cfg, emb_dim=128, tcn_channels=256, tcn_kernel=3,
        tcn_dilations=(1, 2, 4, 8), lstm_units=256, dropout=0.1,
    )
    model.load_weights(str(checkpoint))
    return model


def molecule_nll(model, encoded: np.ndarray, pad_id: int, batch_size: int = 128):
    x = encoded[:, :-1]
    y = encoded[:, 1:]
    rows = []
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size]
        yb = y[start:start + batch_size]
        pred = model(xb, training=False).numpy()
        p = np.take_along_axis(pred, yb[..., None], axis=-1)[..., 0]
        mask = yb != pad_id
        token_nll = -np.log(np.clip(p, 1e-12, 1.0)) * mask
        rows.extend((token_nll.sum(axis=1) / mask.sum(axis=1)).tolist())
    return np.asarray(rows, dtype=float)


def bootstrap_mean_ci(values: np.ndarray, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_heldout_likelihood(cfg: CFG, valid_df: pd.DataFrame, vocabs: dict[str, dict[str, int]]):
    smiles = [encode_special(s, cfg) for s in valid_df[cfg.REDDB_SMILES_COL].astype(str)]
    results = valid_df.copy()
    for arm, checkpoint in (("soft", SOFT_CKPT), ("equal", EQUAL_CKPT)):
        stoi = vocabs[arm]
        encoded = encode_smiles_list(smiles, stoi, cfg, cfg.MAX_LEN)
        if len(encoded) != len(valid_df):
            raise AssertionError("held-out encoding dropped rows unexpectedly")
        model = model_for_checkpoint(cfg, len(stoi), checkpoint)
        results[f"nll_per_token_{arm}"] = molecule_nll(model, encoded, stoi[cfg.PAD])
        del model
        tf.keras.backend.clear_session()
    results["delta_nll_soft_minus_equal"] = (
        results.nll_per_token_soft - results.nll_per_token_equal
    )
    results["soft_over_equal_token_probability_ratio"] = np.exp(
        -results.delta_nll_soft_minus_equal
    )
    results["canonical_smiles"] = results[cfg.REDDB_SMILES_COL].map(canonical)
    results["scaffold"] = results.canonical_smiles.map(scaffold)
    results.to_csv(OUT / "heldout_molecule_likelihood.csv", index=False)

    summary_rows = []
    for label, subset in (("target", results[results.is_target]), ("non_target", results[~results.is_target]), ("all", results)):
        delta_mean, delta_lo, delta_hi = bootstrap_mean_ci(
            subset.delta_nll_soft_minus_equal.to_numpy(),
            seed=BOOTSTRAP_SEED + len(summary_rows),
        )
        summary_rows.append({
            "subset": label,
            "n_molecules": len(subset),
            "n_scaffolds": subset.scaffold.nunique(),
            "soft_nll_per_token_mean": subset.nll_per_token_soft.mean(),
            "equal_nll_per_token_mean": subset.nll_per_token_equal.mean(),
            "paired_delta_nll_soft_minus_equal_mean": delta_mean,
            "paired_delta_nll_bootstrap_95ci_low": delta_lo,
            "paired_delta_nll_bootstrap_95ci_high": delta_hi,
            "soft_higher_likelihood_fraction": (subset.delta_nll_soft_minus_equal < 0).mean(),
            "geometric_mean_soft_over_equal_token_probability": float(np.exp(-delta_mean)),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "heldout_likelihood_summary.csv", index=False)
    return results, summary


def motif_frame(smiles_values, source: str) -> pd.DataFrame:
    rows = []
    for value in smiles_values:
        canon = canonical(value)
        if canon is None:
            continue
        mol = Chem.MolFromSmiles(canon)
        row = {"source": source, "canonical_smiles": canon, "scaffold": scaffold(canon)}
        for name, query in MOTIFS.items():
            row[name] = int(mol.HasSubstructMatch(query))
            row[f"n_{name}"] = len(mol.GetSubstructMatches(query))
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates("canonical_smiles")


def log_odds(a_yes: int, a_no: int, b_yes: int, b_no: int) -> float:
    # Haldane–Anscombe correction protects motifs absent from one finite sample.
    return float(np.log(((a_yes + 0.5) * (b_no + 0.5)) / ((a_no + 0.5) * (b_yes + 0.5))))


def bootstrap_prevalence_difference(a: np.ndarray, b: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    diffs = np.empty(BOOTSTRAP_REPS, dtype=float)
    for i in range(BOOTSTRAP_REPS):
        diffs[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
    return float(a.mean() - b.mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def evaluate_motif_alignment(valid_df: pd.DataFrame, cfg: CFG):
    props = pd.read_csv(WEIGHT_CSV)
    dft = motif_frame(props.smiles_main.astype(str), "reddb")
    labels = props[["smiles_main", "Solvation_energy", "single_point_lumo_solution"]].copy()
    labels["canonical_smiles"] = labels.smiles_main.map(canonical)
    labels["is_target"] = (
        (labels.Solvation_energy <= TARGET_SOLV_MAX)
        & (labels.single_point_lumo_solution <= TARGET_LUMO_MAX)
    )
    dft = dft.merge(labels[["canonical_smiles", "is_target"]].drop_duplicates(), on="canonical_smiles", how="left")

    generated_frames = []
    for arm, path in GENERATED.items():
        raw = pd.read_csv(path)
        selected = raw[
            (raw.model == MODEL)
            & np.isclose(raw.temperature, TEMPERATURE)
            & np.isclose(raw.top_p, TOP_P)
            & (raw.rdkit_valid == 1)
        ]
        generated_frames.append(motif_frame(selected.canon.dropna().astype(str), arm))
    generated = pd.concat(generated_frames, ignore_index=True)
    dft.to_csv(OUT / "dft_motif_molecule_level.csv", index=False)
    generated.to_csv(OUT / "generated_motif_molecule_level.csv", index=False)

    target = dft[dft.is_target == True]  # noqa: E712
    other = dft[dft.is_target == False]  # noqa: E712
    soft = generated[generated.source == "continuous_weight"]
    equal = generated[generated.source == "equal_weight"]
    rows = []
    for i, motif in enumerate(MOTIFS):
        ty, oy = int(target[motif].sum()), int(other[motif].sum())
        sy, ey = int(soft[motif].sum()), int(equal[motif].sum())
        dft_log_or = log_odds(ty, len(target) - ty, oy, len(other) - oy)
        gen_log_or = log_odds(sy, len(soft) - sy, ey, len(equal) - ey)
        dft_diff, dft_lo, dft_hi = bootstrap_prevalence_difference(
            target[motif].to_numpy(), other[motif].to_numpy(), BOOTSTRAP_SEED + 100 + i
        )
        gen_diff, gen_lo, gen_hi = bootstrap_prevalence_difference(
            soft[motif].to_numpy(), equal[motif].to_numpy(), BOOTSTRAP_SEED + 200 + i
        )
        # A motif can only testify about direction if BOTH differences are
        # distinguishable from zero. The original criterion (abs(diff) > 1e-12)
        # admitted pure noise: sulfonic acid entered the tally with a DFT
        # difference of -1.28 pp (95% CI -3.26 to +0.65) and a generated
        # difference of -0.20 pp (95% CI -3.13 to +2.70) — two intervals that
        # both straddle zero — and the coincidental agreement of their signs was
        # counted as a successful directional match.
        nonzero = bool(abs(dft_diff) > 1e-12 and abs(gen_diff) > 1e-12)
        dft_resolved = not (dft_lo <= 0.0 <= dft_hi)
        gen_resolved = not (gen_lo <= 0.0 <= gen_hi)
        informative = bool(nonzero and dft_resolved and gen_resolved)
        aligned = int(np.sign(dft_diff) == np.sign(gen_diff)) if informative else np.nan
        rows.append({
            "motif": motif,
            "smarts": MOTIF_SMARTS[motif],
            "dft_target_n": len(target),
            "dft_non_target_n": len(other),
            "soft_unique_valid_n": len(soft),
            "equal_unique_valid_n": len(equal),
            "dft_target_prevalence_pct": 100 * ty / len(target),
            "dft_non_target_prevalence_pct": 100 * oy / len(other),
            "dft_prevalence_difference_pp": 100 * dft_diff,
            "dft_prevalence_difference_95ci_low_pp": 100 * dft_lo,
            "dft_prevalence_difference_95ci_high_pp": 100 * dft_hi,
            "dft_target_vs_non_target_log_odds": dft_log_or,
            "soft_prevalence_pct": 100 * sy / len(soft),
            "equal_prevalence_pct": 100 * ey / len(equal),
            "soft_minus_equal_prevalence_difference_pp": 100 * gen_diff,
            "soft_minus_equal_prevalence_difference_95ci_low_pp": 100 * gen_lo,
            "soft_minus_equal_prevalence_difference_95ci_high_pp": 100 * gen_hi,
            "soft_vs_equal_log_odds": gen_log_or,
            "informative_for_direction": int(informative),
            "direction_aligned": aligned,
            # kept so the change of criterion is auditable rather than silent
            "nonzero_both_sides": int(nonzero),
            "dft_difference_resolved": int(dft_resolved),
            "generated_difference_resolved": int(gen_resolved),
        })
    motif_summary = pd.DataFrame(rows)
    motif_summary.to_csv(OUT / "equal_controlled_motif_alignment.csv", index=False)

    informative = motif_summary[motif_summary.informative_for_direction == 1]
    x = informative.dft_target_vs_non_target_log_odds.to_numpy()
    y = informative.soft_vs_equal_log_odds.to_numpy()
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(pd.Series(x).rank().corr(pd.Series(y).rank()))
    overall = {
        "architecture": MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "selection": "all RDKit-valid unique molecules; no property-screening filter",
        "dft_target_n": int(len(target)),
        "dft_non_target_n": int(len(other)),
        "dft_target_scaffolds": int(target.scaffold.nunique()),
        "soft_unique_valid_n": int(len(soft)),
        "equal_unique_valid_n": int(len(equal)),
        "direction_aligned_motifs": int(informative.direction_aligned.sum()),
        "n_motifs": int(len(motif_summary)),
        "n_informative_motifs": int(len(informative)),
        "n_motifs_nonzero_both_sides": int(motif_summary.nonzero_both_sides.sum()),
        "n_motifs_unresolved": int(
            (motif_summary.nonzero_both_sides == 1).sum() - len(informative)
        ),
        "unresolved_motifs": motif_summary.loc[
            (motif_summary.nonzero_both_sides == 1)
            & (motif_summary.informative_for_direction == 0), "motif"
        ].tolist(),
        "informative_criterion": (
            "both the DFT and the generated prevalence difference have a 95% bootstrap "
            "confidence interval excluding zero"
        ),
        "pearson_log_odds_alignment": pearson,
        "spearman_log_odds_alignment": spearman,
    }
    (OUT / "motif_alignment_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    return motif_summary, overall


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    train_df, valid_df, vocabs = prepare_exact_split_and_vocab(cfg)
    heldout, likelihood_summary = evaluate_heldout_likelihood(cfg, valid_df, vocabs)
    motif_summary, motif_overall = evaluate_motif_alignment(valid_df, cfg)

    target_summary = likelihood_summary[likelihood_summary.subset == "target"].iloc[0]
    audit = {
        "status": "PASS",
        "split_definition": "original RedDB random 1% validation split, seed cfg.SEED+2 = 44",
        "n_reddb_train": int(len(train_df)),
        "n_reddb_heldout": int(len(valid_df)),
        "n_target_heldout": int(valid_df.is_target.sum()),
        "n_target_heldout_scaffolds": int(heldout[heldout.is_target].scaffold.nunique()),
        "target_delta_nll_soft_minus_equal": float(target_summary.paired_delta_nll_soft_minus_equal_mean),
        "target_delta_nll_95ci": [
            float(target_summary.paired_delta_nll_bootstrap_95ci_low),
            float(target_summary.paired_delta_nll_bootstrap_95ci_high),
        ],
        "target_soft_higher_likelihood_fraction": float(target_summary.soft_higher_likelihood_fraction),
        "motif_alignment": motif_overall,
        "input_sha256": {
            "config": sha256(CONFIG),
            "weight_csv": sha256(WEIGHT_CSV),
            "soft_checkpoint": sha256(SOFT_CKPT),
            "equal_checkpoint": sha256(EQUAL_CKPT),
            "soft_generated": sha256(GENERATED["continuous_weight"]),
            "equal_generated": sha256(GENERATED["equal_weight"]),
        },
        "limitations": [
            "The held-out set is the original random 1% validation split, not a scaffold-disjoint test set.",
            "The target subset is finite and the bootstrap resamples molecules rather than independent training seeds.",
            "Motif alignment is descriptive and does not establish causal chemical understanding.",
        ],
    }
    (OUT / "analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
