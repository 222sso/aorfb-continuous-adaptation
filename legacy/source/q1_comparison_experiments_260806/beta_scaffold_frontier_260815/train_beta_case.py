from __future__ import annotations

"""Fine-tune one A_TCN beta/seed arm from a frozen pretrained checkpoint."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"
CONFIG = ROOT / "tcnlstm_transfer_outputs_260311" / "exp_baseline" / "config_used.json"
PRETRAIN = next(
    (ROOT / "tcnlstm_transfer_outputs_260311" / "exp_baseline").glob(
        "ckpt_pretrain_A_TCN*.weights.h5"
    )
)

sys.path.insert(0, str(ROOT))

from ablation.config import CFG  # noqa: E402
from ablation.datasets import (  # noqa: E402
    make_tf_dataset_with_unit_w,
    make_tf_dataset_with_w,
    mix_two_datasets,
    split_by_w,
)
from ablation.models import build_A_TCN, compile_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, required=True, choices=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--seed", type=int, required=True, choices=(42, 43, 44))
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def beta_slug(beta: float) -> str:
    return f"beta_{beta:.2f}".replace(".", "p")


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
    return cfg


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def deterministic(ds: tf.data.Dataset) -> tf.data.Dataset:
    options = tf.data.Options()
    options.experimental_deterministic = True
    return ds.with_options(options)


def influence_stats(frame: pd.DataFrame, weights: np.ndarray, threshold: float, p_high: float) -> dict:
    high = weights >= threshold
    if not high.any() or high.all():
        raise RuntimeError("Joint-weight high/low pool is empty")
    influence = np.where(high, p_high * weights / high.sum(), (1.0 - p_high) * weights / (~high).sum())
    influence = influence / influence.sum()
    scaffold_mass = pd.Series(influence).groupby(frame.scaffold.reset_index(drop=True)).sum().to_numpy()
    positive = scaffold_mass[scaffold_mass > 0]
    return {
        "influence_ess_molecules": float(1.0 / np.square(influence).sum()),
        "influence_effective_scaffolds": float(np.exp(-(positive * np.log(positive)).sum())),
        "non_target_influence_fraction": float(influence[~frame.is_target.to_numpy(bool)].sum()),
        "target_influence_fraction": float(influence[frame.is_target.to_numpy(bool)].sum()),
    }


def main() -> None:
    args = parse_args()
    manifest_path = ARTIFACTS / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run prepare_beta_data.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(PRETRAIN) != manifest["pretrain_sha256"]:
        raise RuntimeError("Pretrained checkpoint hash differs from frozen manifest")

    base_dir = HERE / "smoke" if args.smoke_test else RESULTS
    run_dir = base_dir / f"seed_{args.seed}" / beta_slug(args.beta)
    run_dir.mkdir(parents=True, exist_ok=True)
    complete = run_dir / "training_complete.json"
    if complete.exists() and not args.smoke_test:
        print(f"SKIP completed beta={args.beta} seed={args.seed}")
        return

    cfg = load_cfg()
    cfg.SEED = args.seed
    cfg.OUTDIR = str(run_dir)
    if args.smoke_test:
        cfg.FINETUNE_EPOCHS = 1
        cfg.FINETUNE_MIX_STEPS = 5
    set_seed(args.seed)

    vocab = json.loads((ARTIFACTS / "vocab.json").read_text(encoding="utf-8"))
    stoi = {str(key): int(value) for key, value in vocab["stoi"].items()}
    reddb = pd.read_csv(ARTIFACTS / "reddb_train_augmented.csv.gz")
    valid = pd.read_csv(ARTIFACTS / "reddb_common_unit_valid.csv.gz")
    chembl = pd.read_csv(ARTIFACTS / "chembl_train_augmented.csv.gz")

    beta_meta = manifest["beta_weight_stats"][str(float(args.beta))]
    c = float(beta_meta["normalization_constant"])
    median_count = float(manifest["median_scaffold_count"])
    factor = (reddb.scaffold_count.to_numpy(float) / median_count) ** (-float(args.beta))
    joint = np.minimum(10.0, c * reddb.property_weight.to_numpy(float) * factor).astype(np.float32)
    if not np.isclose(joint.mean(), reddb.property_weight.mean(), rtol=0, atol=2e-6):
        raise RuntimeError("Mean-weight normalization failed")

    smiles = reddb.encoded_smiles.astype(str).tolist()
    (hi_s, hi_w), (lo_s, lo_w) = split_by_w(smiles, joint, cfg.REDDB_W_THR)
    hi_ds, n_hi = make_tf_dataset_with_w(
        hi_s, hi_w, stoi, cfg, cfg.MAX_LEN, cfg.BATCH_SIZE_FINETUNE, True, args.seed + 120
    )
    lo_ds, n_lo = make_tf_dataset_with_w(
        lo_s, lo_w, stoi, cfg, cfg.MAX_LEN, cfg.BATCH_SIZE_FINETUNE, True, args.seed + 121
    )
    if not n_hi or not n_lo:
        raise RuntimeError(f"Empty joint-weight pool: high={n_hi}, low={n_lo}")
    reddb_ds = mix_two_datasets(hi_ds, lo_ds, p_a=cfg.REDDB_P_HI, seed=args.seed + 122)
    chembl_ds, n_chembl = make_tf_dataset_with_unit_w(
        chembl.encoded_smiles.astype(str).tolist(), stoi, cfg, cfg.MAX_LEN,
        cfg.BATCH_SIZE_FINETUNE, True, args.seed + 14,
    )
    valid_ds, n_valid = make_tf_dataset_with_unit_w(
        valid.encoded_smiles.astype(str).tolist(), stoi, cfg, cfg.MAX_LEN,
        cfg.BATCH_SIZE_FINETUNE, False, 55,
    )
    train_ds = tf.data.Dataset.sample_from_datasets(
        [deterministic(reddb_ds), deterministic(chembl_ds.repeat())],
        weights=[cfg.FINETUNE_MIX_RATIO, 1.0 - cfg.FINETUNE_MIX_RATIO],
        seed=args.seed + 200,
    ).prefetch(tf.data.AUTOTUNE)
    train_ds = deterministic(train_ds)
    valid_ds = deterministic(valid_ds)

    model = build_A_TCN(
        len(stoi), cfg, emb_dim=128, tcn_channels=256, tcn_kernel=3,
        tcn_dilations=(1, 2, 4, 8), dropout=0.1,
    )
    model.load_weights(PRETRAIN)
    compile_model(model, cfg.LR_FINETUNE)
    best_path = run_dir / "best_common_unit_val.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_path, save_weights_only=True, save_best_only=True,
            monitor="val_loss", mode="min"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, mode="min"
        ),
        tf.keras.callbacks.CSVLogger(run_dir / "training_history.csv"),
    ]
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        steps_per_epoch=int(cfg.FINETUNE_MIX_STEPS),
        validation_steps=int(math.ceil(n_valid / cfg.BATCH_SIZE_FINETUNE)),
        epochs=int(cfg.FINETUNE_EPOCHS),
        callbacks=callbacks,
        verbose=2,
    )
    model.load_weights(best_path)
    val_losses = [float(value) for value in history.history["val_loss"]]
    stats = influence_stats(reddb, joint, float(cfg.REDDB_W_THR), float(cfg.REDDB_P_HI))
    payload = {
        "beta": float(args.beta),
        "training_seed": args.seed,
        "backbone": "A_TCN",
        "pretrain_checkpoint_sha256": sha256(PRETRAIN),
        "selection_objective": "common unit-weight RedDB validation loss",
        "selected_epoch": int(np.argmin(val_losses) + 1),
        "epochs_completed": len(val_losses),
        "val_losses": val_losses,
        "steps_per_epoch": int(cfg.FINETUNE_MIX_STEPS),
        "n_high_augmented": int(n_hi),
        "n_low_augmented": int(n_lo),
        "joint_weight_mean": float(joint.mean()),
        "joint_weight_min": float(joint.min()),
        "joint_weight_max": float(joint.max()),
        "normalization_constant": c,
        "n_chembl_augmented": int(n_chembl),
        "n_common_validation": int(n_valid),
        "checkpoint": str(best_path.relative_to(HERE)),
        "checkpoint_sha256": sha256(best_path),
        **stats,
    }
    complete.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

