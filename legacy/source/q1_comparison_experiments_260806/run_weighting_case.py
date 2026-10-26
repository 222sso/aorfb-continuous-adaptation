from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP))

from ablation.config import CFG  # noqa: E402
from ablation.data import (  # noqa: E402
    augment_train_smiles,
    augment_train_smiles_with_w,
    build_train_canon_set,
    detokenize_to_plain_smiles,
    read_chembl_train_valid,
    read_reddb_train_valid_with_weights,
)
from ablation.metrics import rdkit_metrics  # noqa: E402
from ablation.models import build_C_TCN_LSTM  # noqa: E402
from ablation.predict_generated import run_from_cfg as run_predictor  # noqa: E402
from ablation.train import train_once  # noqa: E402
from ablation.vocab import build_char_vocab  # noqa: E402
from batch_sampling_v2 import generate_smiles_batched  # noqa: E402


WEIGHTED_CONFIG = ROOT / "tcnlstm_transfer_outputs_260311" / "exp_baseline" / "config_used.json"
BASE_WEIGHT_FILE = ROOT / "dist_reports" / "softft_v3" / "softft_scores_weights.csv"
PRETRAIN_CKPT = next(
    (ROOT / "tcnlstm_transfer_outputs_260311" / "exp_baseline").glob(
        "ckpt_pretrain_C_TCN_LSTM*.weights.h5"
    )
)
SPLIT_SEED = 42
WEIGHT_AMPLIFICATION = 9.0
WEIGHT_CAP = 10.0
GEN_TEMPERATURE = 0.90
GEN_TOP_P = 0.95
GEN_N = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One isolated property-weighting sweep case")
    parser.add_argument("--weight-threshold", type=float, required=True, help="activation threshold in z->w")
    parser.add_argument("--pool-threshold", type=float, required=True, help="w threshold defining high pool")
    parser.add_argument("--p-high", type=float, required=True, help="probability of drawing from high pool")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--gen-n", type=int, default=GEN_N)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def load_cfg() -> CFG:
    cfg = CFG()
    payload = json.loads(WEIGHTED_CONFIG.read_text(encoding="utf-8-sig"))
    for key, value in payload.items():
        setattr(cfg, key, value)
    for key in ("CHEMBL_CSV", "REDDB_CSV", "PRED_DIR", "NOVEL_REF_PATH"):
        value = getattr(cfg, key, None)
        if value and not Path(value).is_absolute():
            setattr(cfg, key, str((ROOT / value).resolve()))
    return cfg


def slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def make_weight_file(path: Path, activation: float) -> dict:
    source = pd.read_csv(BASE_WEIGHT_FILE)
    if "z" not in source or "w" not in source:
        raise ValueError("base weight file must contain z and w")
    z = pd.to_numeric(source.z, errors="coerce")
    w = np.where(
        z < activation,
        1.0,
        1.0 + WEIGHT_AMPLIFICATION * (z - activation) / (1.0 - activation),
    )
    source["w"] = np.minimum(w, WEIGHT_CAP)
    source["use_for_ft"] = z.notna()
    path.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "n_rows": len(source),
        "n_weight_gt_1": int((source.w > 1).sum()),
        "weight_min": float(source.w.min()),
        "weight_max": float(source.w.max()),
        "weight_mean": float(source.w.mean()),
        "sha256": digest,
    }


def filter_by_maxlen(smiles: list[str], weights, cfg: CFG):
    kept_s, kept_w = [], []
    for value, weight in zip(smiles, weights):
        if (len(value) + 2) > cfg.MAX_LEN and cfg.DROP_TOO_LONG:
            continue
        kept_s.append(value)
        kept_w.append(float(weight))
    return kept_s, np.asarray(kept_w, dtype=np.float32)


def stable_case_seed(train_seed: int, activation: float, pool_threshold: float, p_high: float) -> int:
    key = f"{train_seed}|{activation:.6f}|{pool_threshold:.6f}|{p_high:.6f}|{GEN_TEMPERATURE}|{GEN_TOP_P}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


def summarize_predictions(rows: pd.DataFrame, n_total: int) -> dict:
    valid = rows[rows.rdkit_valid == 1].copy()
    scored = valid[valid.solv_mean_eV.notna() & valid.lumo_mean_eV.notna()].copy()
    hits = scored[(scored.solv_mean_eV <= -1.5) & (scored.lumo_mean_eV <= -4.0)].copy()
    unique_valid = set(valid.canon.astype(str))
    unique_hits = set(hits.canon.astype(str))

    def pct(num: int, den: int) -> float:
        return 100.0 * num / den if den else float("nan")

    return {
        "n_generated": n_total,
        "n_valid": len(valid),
        "n_unique": len(unique_valid),
        "n_scored": len(scored),
        "n_hits": len(hits),
        "n_unique_hits": len(unique_hits),
        "validity_pct": pct(len(valid), n_total),
        "uniqueness_pct": pct(len(unique_valid), len(valid)),
        "prediction_coverage_pct": pct(len(scored), len(valid)),
        "target_pass_pct": pct(len(hits), len(scored)),
        "yield_raw_target_pct": pct(len(hits), n_total),
        "yield_unique_target_pct": pct(len(unique_hits), n_total),
        "hit_multiplicity": len(hits) / len(unique_hits) if unique_hits else float("nan"),
    }


def main() -> None:
    args = parse_args()
    configure_gpu()
    if not 0 < args.weight_threshold < 1:
        raise ValueError("weight-threshold must lie in (0,1)")
    if not 0 <= args.p_high <= 1:
        raise ValueError("p-high must lie in [0,1]")

    case_name = (
        f"Tw_{slug(args.weight_threshold)}__Wthr_{slug(args.pool_threshold)}__"
        f"Phi_{slug(args.p_high)}__seed_{args.train_seed}"
    )
    outdir = EXP / "results" / "weighting_sweep" / case_name
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "case_complete.json").exists():
        print(f"SKIP complete case: {case_name}")
        return

    cfg = load_cfg()
    cfg.OUTDIR = str(outdir)
    cfg.WEIGHT_CSV = str(outdir / "weights" / "softft_scores_weights.csv")
    cfg.REDDB_W_THR = float(args.pool_threshold)
    cfg.REDDB_P_HI = float(args.p_high)
    cfg.USE_FT_WEIGHTS = True
    cfg.USE_ONLY_SOFTFT = False
    cfg.PRETRAIN_EPOCHS = 0
    if args.smoke_test:
        cfg.FINETUNE_EPOCHS = 1
        cfg.FINETUNE_MIX_STEPS = 5

    weight_meta = make_weight_file(Path(cfg.WEIGHT_CSV), args.weight_threshold)

    # Freeze split and augmentation records to seed 42 for all training seeds.
    cfg.SEED = SPLIT_SEED
    chembl_train, chembl_valid = read_chembl_train_valid(cfg)
    reddb_train, reddb_valid, reddb_train_w, reddb_valid_w, _, _ = read_reddb_train_valid_with_weights(cfg)
    chembl_train = augment_train_smiles(
        chembl_train, cfg, n_aug_per_mol=cfg.CHEMBL_AUG_N,
        aug_prob=cfg.CHEMBL_AUG_PROB, seed=SPLIT_SEED + 2026,
    )
    reddb_train, reddb_train_w = augment_train_smiles_with_w(
        reddb_train, reddb_train_w, cfg, n_aug_per_mol=cfg.REDDB_AUG_N,
        aug_prob=cfg.REDDB_AUG_PROB, seed=SPLIT_SEED + 2025,
    )
    reddb_train, reddb_train_w = filter_by_maxlen(reddb_train, reddb_train_w, cfg)
    stoi, itos = build_char_vocab(chembl_train + reddb_train, cfg)
    train_plain = [detokenize_to_plain_smiles(s, cfg) for s in chembl_train + reddb_train]
    train_reference = build_train_canon_set(train_plain)

    cfg.SEED = int(args.train_seed)
    set_seed(args.train_seed)
    model = build_C_TCN_LSTM(
        len(stoi), cfg, emb_dim=128, tcn_channels=256, tcn_kernel=3,
        tcn_dilations=(1, 2, 4, 8), lstm_units=256, dropout=0.1,
    )
    model.load_weights(str(PRETRAIN_CKPT))
    train_params = {
        "SEED": int(args.train_seed),
        "BATCH_SIZE": cfg.BATCH_SIZE,
        "BATCH_SIZE_FINETUNE": cfg.BATCH_SIZE_FINETUNE,
        "PRETRAIN_EPOCHS": 0,
        "FINETUNE_EPOCHS": cfg.FINETUNE_EPOCHS,
        "LR_PRETRAIN": cfg.LR_PRETRAIN,
        "LR_FINETUNE": cfg.LR_FINETUNE,
        "FINETUNE_MIX_RATIO": cfg.FINETUNE_MIX_RATIO,
        "FINETUNE_MIX_STEPS": cfg.FINETUNE_MIX_STEPS,
    }
    model, run_id, _, ft_ckpt = train_once(
        model=model,
        stoi=stoi,
        cfg=cfg,
        train_params=train_params,
        chembl_train=chembl_train,
        chembl_valid=chembl_valid,
        reddb_train=reddb_train,
        reddb_train_w=reddb_train_w,
        reddb_valid=reddb_valid,
        reddb_valid_w=reddb_valid_w,
        run_prefix=f"WSWEEP_{case_name}",
        use_ft_weights=True,
    )

    gen_n = 40 if args.smoke_test else int(args.gen_n)
    gen_seed = stable_case_seed(args.train_seed, args.weight_threshold, args.pool_threshold, args.p_high)
    tokenized = generate_smiles_batched(
        model, itos, stoi, cfg, gen_n=gen_n, temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P, seed=gen_seed, batch_size=min(256, gen_n),
        progress=lambda done, total: print(f"[generation] {done}/{total}", flush=True),
    )
    plain = [s.replace(cfg.TOK_SO3H, "S(=O)(=O)O").replace(cfg.TOK_COOH, "C(=O)O") for s in tokenized]
    tag = f"n{gen_n}_T{GEN_TEMPERATURE:.2f}_p{GEN_TOP_P:.2f}".replace(".", "p")
    gen_path = outdir / f"C_TCN_LSTM_gen_{tag}.txt"
    gen_path.write_text("\n".join(plain) + "\n", encoding="utf-8")
    rdkit = rdkit_metrics(plain, train_reference)

    del model
    tf.keras.backend.clear_session()
    pred_summary, pred_rows = run_predictor(cfg)
    pred_summary.to_csv(outdir / "pred_eval_summary_metrics.csv", index=False)
    pred_rows.to_csv(outdir / "pred_eval_samples_scored_all.csv", index=False)
    metrics = {
        "case": case_name,
        "weight_activation_threshold": args.weight_threshold,
        "pool_weight_threshold": args.pool_threshold,
        "p_high": args.p_high,
        "train_seed": args.train_seed,
        "split_seed": SPLIT_SEED,
        "generation_seed": gen_seed,
        "generation_temperature": GEN_TEMPERATURE,
        "generation_top_p": GEN_TOP_P,
        "run_id": run_id,
        "pretrain_checkpoint": str(PRETRAIN_CKPT.relative_to(ROOT)),
        "finetune_checkpoint": str(Path(ft_ckpt).relative_to(ROOT)),
        "weight_file": weight_meta,
        "rdkit_metrics": rdkit,
        **summarize_predictions(pred_rows, gen_n),
    }
    (outdir / "case_complete.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
