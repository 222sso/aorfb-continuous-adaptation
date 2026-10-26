from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP))

from ablation.data import (  # noqa: E402
    augment_train_smiles,
    augment_train_smiles_with_w,
    detokenize_to_plain_smiles,
    read_chembl_train_valid,
    read_reddb_train_valid_with_weights,
)
from ablation.datasets import make_tf_dataset  # noqa: E402
from ablation.metrics import build_train_canon_set, rdkit_metrics  # noqa: E402
from ablation.models import compile_model  # noqa: E402
from ablation.predict_generated import run_from_cfg as run_predictor  # noqa: E402
from ablation.train import train_once  # noqa: E402
from ablation.vocab import build_char_vocab  # noqa: E402
from batch_sampling_v2 import generate_smiles_batched  # noqa: E402
from run_weighting_case import filter_by_maxlen, load_cfg, summarize_predictions  # noqa: E402
from smiles_diagnostics import summarize_diagnostics  # noqa: E402
from tcn_ablation_models import VARIANTS, build_tcn_ablation, front_end_parameter_count  # noqa: E402


GEN_N = 1000
GEN_TEMPERATURE = 0.90
GEN_TOP_P = 0.95
SPLIT_SEED = 42
TRAIN_SEED = 42
EXTERNAL_PRETRAIN = {
    "full": ROOT / "tcnlstm_transfer_outputs_260311" / "exp_baseline"
    / 'ckpt_pretrain_C_TCN_LSTM.weights.h5',
    "no_tcn": ROOT / "tcnlstm_transfer_outputs_260311" / "exp_baseline"
    / 'ckpt_pretrain_B_LSTM.weights.h5',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def configure_gpu():
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def stable_generation_seed(variant: str) -> int:
    key = f"tcn-front-end-ablation|{variant}|{TRAIN_SEED}|{GEN_TEMPERATURE}|{GEN_TOP_P}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


def prepare_data(cfg):
    cfg.SEED = SPLIT_SEED
    chembl_train, chembl_valid = read_chembl_train_valid(cfg)
    reddb_train, reddb_valid, reddb_train_w, reddb_valid_w, _, _ = read_reddb_train_valid_with_weights(cfg)
    chembl_train = augment_train_smiles(
        chembl_train,
        cfg,
        n_aug_per_mol=cfg.CHEMBL_AUG_N,
        aug_prob=cfg.CHEMBL_AUG_PROB,
        seed=SPLIT_SEED + 2026,
    )
    reddb_train, reddb_train_w = augment_train_smiles_with_w(
        reddb_train,
        reddb_train_w,
        cfg,
        n_aug_per_mol=cfg.REDDB_AUG_N,
        aug_prob=cfg.REDDB_AUG_PROB,
        seed=SPLIT_SEED + 2025,
    )
    reddb_train, reddb_train_w = filter_by_maxlen(reddb_train, reddb_train_w, cfg)
    stoi, itos = build_char_vocab(chembl_train + reddb_train, cfg)
    train_plain = [detokenize_to_plain_smiles(s, cfg) for s in chembl_train + reddb_train]
    train_reference = build_train_canon_set(train_plain)
    return (
        chembl_train,
        chembl_valid,
        reddb_train,
        reddb_valid,
        reddb_train_w,
        reddb_valid_w,
        stoi,
        itos,
        train_reference,
    )


def pretrain_if_needed(model, variant, cfg, stoi, chembl_train, chembl_valid, outdir, smoke):
    external = EXTERNAL_PRETRAIN.get(variant)
    if external and external.exists():
        try:
            model.load_weights(str(external))
            return model, external, "existing matched pretrained checkpoint"
        except Exception as exc:
            print(f"[pretrain] external checkpoint incompatible: {exc}", flush=True)
            model = build_tcn_ablation(variant, len(stoi), cfg.MAX_LEN)

    stable = outdir / "pretrain_best.weights.h5"
    if stable.exists():
        model.load_weights(str(stable))
        return model, stable, "resumed isolated pretrained checkpoint"

    train_ds, _ = make_tf_dataset(
        chembl_train,
        stoi=stoi,
        cfg=cfg,
        max_len=cfg.MAX_LEN,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        seed=TRAIN_SEED + 10,
    )
    valid_ds, _ = make_tf_dataset(
        chembl_valid,
        stoi=stoi,
        cfg=cfg,
        max_len=cfg.MAX_LEN,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        seed=TRAIN_SEED + 11,
    )
    compile_model(model, cfg.LR_PRETRAIN)
    epochs = 1 if smoke else cfg.PRETRAIN_EPOCHS
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                str(stable), save_weights_only=True, save_best_only=True,
                monitor="val_loss", mode="min",
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True,
            ),
            tf.keras.callbacks.CSVLogger(str(outdir / "pretrain_history.csv")),
        ],
        verbose=1,
    )
    model.load_weights(str(stable))
    (outdir / "pretrain_history.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in history.history.items()}, indent=2),
        encoding="utf-8",
    )
    return model, stable, "new isolated matched pretraining"


def main():
    args = parse_args()
    configure_gpu()
    set_seed(TRAIN_SEED)
    outdir = EXP / "results" / "tcn_frontend_ablation" / args.variant
    outdir.mkdir(parents=True, exist_ok=True)
    complete = outdir / "case_complete.json"
    if complete.exists() and not args.smoke_test:
        print(f"SKIP complete: {args.variant}")
        return

    cfg = load_cfg()
    cfg.OUTDIR = str(outdir)
    cfg.SEED = SPLIT_SEED
    if args.smoke_test:
        cfg.FINETUNE_EPOCHS = 1
        cfg.FINETUNE_MIX_STEPS = 5
    data = prepare_data(cfg)
    (
        chembl_train,
        chembl_valid,
        reddb_train,
        reddb_valid,
        reddb_train_w,
        reddb_valid_w,
        stoi,
        itos,
        train_reference,
    ) = data

    model = build_tcn_ablation(args.variant, len(stoi), cfg.MAX_LEN)
    model, pretrain_ckpt, pretrain_provenance = pretrain_if_needed(
        model, args.variant, cfg, stoi, chembl_train, chembl_valid, outdir, args.smoke_test
    )
    parameter_meta = {
        "model_trainable_parameters": int(model.count_params()),
        "front_end_parameters": front_end_parameter_count(model, args.variant),
        "pretrain_checkpoint": str(pretrain_ckpt.relative_to(ROOT)),
        "pretrain_provenance": pretrain_provenance,
    }

    train_params = {
        "SEED": TRAIN_SEED,
        "BATCH_SIZE": cfg.BATCH_SIZE,
        "BATCH_SIZE_FINETUNE": cfg.BATCH_SIZE_FINETUNE,
        "PRETRAIN_EPOCHS": 0,
        "FINETUNE_EPOCHS": cfg.FINETUNE_EPOCHS,
        "LR_PRETRAIN": cfg.LR_PRETRAIN,
        "LR_FINETUNE": cfg.LR_FINETUNE,
        "FINETUNE_MIX_RATIO": cfg.FINETUNE_MIX_RATIO,
        "FINETUNE_MIX_STEPS": cfg.FINETUNE_MIX_STEPS,
    }
    cfg.SEED = TRAIN_SEED
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
        run_prefix=f"TCN_ABL_{args.variant}",
        use_ft_weights=True,
    )
    model.load_weights(ft_ckpt)

    gen_n = 40 if args.smoke_test else GEN_N
    seed = stable_generation_seed(args.variant)
    tokenized = generate_smiles_batched(
        model,
        itos,
        stoi,
        cfg,
        gen_n=gen_n,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        seed=seed,
        batch_size=min(256, gen_n),
        progress=lambda done, total: print(f"[generation] {done}/{total}", flush=True),
    )
    plain = [detokenize_to_plain_smiles(s, cfg) for s in tokenized]
    tag = f"n{gen_n}_T{GEN_TEMPERATURE:.2f}_p{GEN_TOP_P:.2f}".replace(".", "p")
    gen_path = outdir / f"TCN_ABL_{args.variant}_gen_{tag}.txt"
    gen_path.write_text("\n".join(plain) + "\n", encoding="utf-8")
    rdkit = rdkit_metrics(plain, train_reference)

    del model
    tf.keras.backend.clear_session()
    pred_summary, pred_rows = run_predictor(cfg)
    pred_summary.to_csv(outdir / "pred_eval_summary_metrics.csv", index=False)
    pred_rows.to_csv(outdir / "pred_eval_samples_scored_all.csv", index=False)
    metrics = {
        "variant": args.variant,
        "run_id": run_id,
        "split_seed": SPLIT_SEED,
        "train_seed": TRAIN_SEED,
        "generation_seed": seed,
        "generation_temperature": GEN_TEMPERATURE,
        "generation_top_p": GEN_TOP_P,
        "finetune_checkpoint": str(Path(ft_ckpt).relative_to(ROOT)),
        **parameter_meta,
        "rdkit_metrics_full_training_reference": rdkit,
        **summarize_predictions(pred_rows, gen_n),
        "syntax_and_scaffold_diagnostics": summarize_diagnostics(plain, pred_rows),
    }
    complete.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
