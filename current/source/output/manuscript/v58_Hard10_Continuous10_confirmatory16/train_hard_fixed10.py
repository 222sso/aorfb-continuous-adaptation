from __future__ import annotations

"""Fine-tune one (dataset, arm, seed) case from the shared ChEMBL pretraining.

This isolated copy retains the original data/model/optimizer implementation.
The user-requested Hard model is fixed to the end of epoch10; validation is
monitored but does not stop training or restore earlier weights.
"""

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

OUT = Path(__file__).resolve().parent
HERE = OUT.parents[2] / 'q1_comparison_experiments_260806/generality_replication_260816'
ROOT = HERE.parents[1]
ARTIFACTS = HERE / "artifacts"
RESULTS = OUT / "runs"
sys.path[:0] = [str(HERE), str(ROOT)]

from ablation.config import CFG  # noqa: E402
from ablation.datasets import (  # noqa: E402
    make_tf_dataset_with_unit_w,
    make_tf_dataset_with_w,
    mix_two_datasets,
    split_by_w,
)
from ablation.models import compile_model  # noqa: E402
import backbones  # noqa: E402

HIGH_POOL_THRESHOLD = 1.2
HIGH_POOL_PROBABILITY = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="artifact stem, e.g. zinc, qm9, or a redundancy subset "
                             "such as zinc_r50_random")
    parser.add_argument("--arm", required=True, choices=("equal", "soft", "hard"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--weighted-validation", action="store_true",
                        help="select the checkpoint on the PROPERTY-WEIGHTED held-out loss, "
                             "as the published pipeline did for the continuous arm. Default "
                             "is the common unit-weight loss for every arm.")
    parser.add_argument("--vocab", default="vocab.json")
    parser.add_argument("--pretrain-meta", default=None,
                        help="defaults to the metadata for --backbone")
    backbones.add_argument(parser)
    parser.add_argument("--tag", default=None,
                        help="output subdirectory suffix, for variant runs")
    parser.add_argument("--p3a-config", default=None)
    parser.add_argument("--p3a-condition", choices=("a", "b", "c", "d", "e", "f"))
    parser.add_argument("--p3a-operating-point", default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    limit = os.environ.get("GPU_MEM_LIMIT_MB")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            if limit:
                # Hard ceiling. Memory growth alone has no upper bound and this
                # pipeline was observed to climb past 20 GB.
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=int(limit))]
                )
            else:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def deterministic(ds: tf.data.Dataset) -> tf.data.Dataset:
    options = tf.data.Options()
    options.experimental_deterministic = True
    return ds.with_options(options)


def influence_stats(frame: pd.DataFrame, arm: str, weights=None,
                    threshold=HIGH_POOL_THRESHOLD, p_high=HIGH_POOL_PROBABILITY) -> dict:
    """Normalised per-molecule influence a_i = pi_i w_i / sum_j pi_j w_j."""
    weight = (frame.property_weight.to_numpy(float) if weights is None
              else np.asarray(weights, dtype=float))
    n = len(frame)
    if arm == "soft":
        high = frame.property_weight.to_numpy(float) >= threshold
        if not high.any() or high.all():
            raise RuntimeError("soft arm high/low pool is degenerate")
        pi = np.where(
            high,
            p_high / high.sum(),
            (1.0 - p_high) / (~high).sum(),
        )
        influence = pi * weight
    else:
        influence = np.full(n, 1.0 / n)
    influence = influence / influence.sum()
    scaffold_mass = (
        pd.Series(influence).groupby(frame.scaffold.reset_index(drop=True)).sum().to_numpy()
    )
    positive = scaffold_mass[scaffold_mass > 0]
    is_target = frame.is_target.to_numpy(bool)
    return {
        "influence_ess_molecules": float(1.0 / np.square(influence).sum()),
        "influence_effective_scaffolds": float(np.exp(-(positive * np.log(positive)).sum())),
        "influence_distinct_scaffolds": int(len(positive)),
        "target_influence_fraction": float(influence[is_target].sum()),
        "non_target_influence_fraction": float(influence[~is_target].sum()),
    }



class Progress(tf.keras.callbacks.Callback):
    def __init__(self,path):super().__init__();self.path=path
    def on_epoch_begin(self,epoch,logs=None):self.epoch=epoch+1
    def on_train_batch_end(self,batch,logs=None):
        if (batch+1)%100==0:
            import datetime
            self.path.write_text(json.dumps({'epoch':self.epoch,'step':batch+1,'steps_per_epoch':800,'loss':float(logs['loss']),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))

def main() -> None:
    args = parse_args()
    assert args.arm == "hard" and args.p3a_condition == "e"
    assert args.backbone in ("A_TCN", "B_LSTM", "C_TCN_LSTM")
    spec_study = json.loads((OUT / "protocol.json").read_text())
    assert args.seed in spec_study["training_seeds"]
    assert args.dataset == "reddb"
    assert args.tag == ("confirm16_Hard10_" if args.arm == "hard" else "confirm16_Continuous10_") + {"A_TCN":"TCN", "B_LSTM":"LSTM", "C_TCN_LSTM":"TCN-LSTM"}[args.backbone]
    assert not args.weighted_validation and not args.smoke_test
    pretrain_meta = ARTIFACTS / (args.pretrain_meta
                                or backbones.pretrain_meta_name(args.backbone))
    if not pretrain_meta.exists():
        raise FileNotFoundError("Run pretrain_shared.py first")
    pretrain_ckpt = ARTIFACTS / json.loads(pretrain_meta.read_text())["checkpoint"]

    base = HERE / "smoke" if args.smoke_test else RESULTS
    stem = args.dataset if not args.tag else f"{args.dataset}_{args.tag}"
    run_dir = base / stem / args.arm / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    complete = run_dir / "training_complete.json"
    if complete.exists() and not args.smoke_test:
        print(f"SKIP completed {args.dataset}/{args.arm}/seed{args.seed}")
        return

    cfg = CFG()
    cfg.SEED = args.seed
    p3a = None
    loss_weight_enabled = True
    high_pool_threshold = HIGH_POOL_THRESHOLD
    high_pool_probability = HIGH_POOL_PROBABILITY
    if args.p3a_config:
        if not args.p3a_condition or not args.p3a_operating_point or not args.tag:
            raise ValueError("P3-a requires condition, operating point, and isolated --tag")
        spec = json.loads(Path(args.p3a_config).read_text(encoding="utf-8"))
        condition = spec["conditions"][args.p3a_condition]
        point = spec["operating_points"][args.p3a_operating_point]
        if args.arm != condition["arm"]:
            raise ValueError("P3-a condition arm disagrees with --arm")
        cfg.FINETUNE_MIX_RATIO = float(condition["reddb_mix_ratio"])
        cfg.FINETUNE_MIX_STEPS = int(condition["steps_per_epoch"])
        loss_weight_enabled = bool(condition["loss_weight_enabled"])
        high_pool_threshold = float(point["pool_threshold"])
        high_pool_probability = condition["high_pool_probability"]
        if high_pool_probability == "point":
            high_pool_probability = float(point["reported_p_high"])
        p3a = {"condition": args.p3a_condition,
               "operating_point": args.p3a_operating_point,
               "spec": condition, "point": point}
        if not 0.0 <= cfg.FINETUNE_MIX_RATIO <= 1.0:
            raise ValueError("Invalid RedDB mix ratio")
    if args.smoke_test:
        cfg.FINETUNE_EPOCHS = 1
        cfg.FINETUNE_MIX_STEPS = 5
    set_seed(args.seed)

    vocab = json.loads((ARTIFACTS / args.vocab).read_text(encoding="utf-8"))
    stoi = {str(k): int(v) for k, v in vocab["stoi"].items()}
    corpus = pd.read_csv(ARTIFACTS / f"{args.dataset}.csv.gz")
    chembl = pd.read_csv(ARTIFACTS / "chembl.csv.gz")

    train = corpus[corpus.split == "train"].reset_index(drop=True)
    valid = corpus[corpus.split == "valid"].reset_index(drop=True)
    if args.arm == "hard":
        train = train[train.is_target].reset_index(drop=True)

    smiles = train.encoded_smiles.astype(str).tolist()
    if args.arm == "soft":
        weights = train.property_weight.to_numpy(float)
        (hi_s, hi_w), (lo_s, lo_w) = split_by_w(smiles, weights, high_pool_threshold)
        if not loss_weight_enabled:
            hi_w = [1.0] * len(hi_w)
            lo_w = [1.0] * len(lo_w)
        hi_ds, n_hi = make_tf_dataset_with_w(
            hi_s, hi_w, stoi, cfg, cfg.MAX_LEN, cfg.BATCH_SIZE_FINETUNE, True, args.seed + 120
        )
        lo_ds, n_lo = make_tf_dataset_with_w(
            lo_s, lo_w, stoi, cfg, cfg.MAX_LEN, cfg.BATCH_SIZE_FINETUNE, True, args.seed + 121
        )
        if not n_hi or not n_lo:
            raise RuntimeError(f"Empty pool: high={n_hi} low={n_lo}")
        if high_pool_probability == "pool_fraction":
            high_pool_probability = n_hi / (n_hi + n_lo)
        high_pool_probability = float(high_pool_probability)
        if not 0.0 <= high_pool_probability <= 1.0:
            raise ValueError("Invalid high-pool probability")
        corpus_ds = mix_two_datasets(hi_ds, lo_ds, p_a=high_pool_probability, seed=args.seed + 122)
        n_corpus = n_hi + n_lo
    else:
        corpus_ds, n_corpus = make_tf_dataset_with_unit_w(
            smiles, stoi, cfg, cfg.MAX_LEN, cfg.BATCH_SIZE_FINETUNE, True, args.seed + 120
        )
        corpus_ds = corpus_ds.repeat()
        n_hi = n_lo = 0

    chembl_ds, n_chembl = make_tf_dataset_with_unit_w(
        chembl.encoded_smiles.astype(str).tolist(), stoi, cfg,
        cfg.MAX_LEN, cfg.BATCH_SIZE_FINETUNE, True, args.seed + 14,
    )
    if args.weighted_validation:
        valid_ds, n_valid = make_tf_dataset_with_w(
            valid.encoded_smiles.astype(str).tolist(),
            valid.property_weight.to_numpy(float), stoi, cfg,
            cfg.MAX_LEN, cfg.BATCH_SIZE_FINETUNE, False, 55,
        )
    else:
        valid_ds, n_valid = make_tf_dataset_with_unit_w(
            valid.encoded_smiles.astype(str).tolist(), stoi, cfg,
            cfg.MAX_LEN, cfg.BATCH_SIZE_FINETUNE, False, 55,
        )
    train_ds = deterministic(
        tf.data.Dataset.sample_from_datasets(
            [deterministic(corpus_ds), deterministic(chembl_ds.repeat())],
            weights=[cfg.FINETUNE_MIX_RATIO, 1.0 - cfg.FINETUNE_MIX_RATIO],
            seed=args.seed + 200,
        ).prefetch(tf.data.AUTOTUNE)
    )
    valid_ds = deterministic(valid_ds)

    model = backbones.build(args.backbone, len(stoi), cfg)
    model.load_weights(pretrain_ckpt)
    compile_model(model, cfg.LR_FINETUNE)
    best_path = run_dir / "epoch10.weights.h5"
    history_path = run_dir / "training_history.csv"
    prior_best = None
    if p3a and best_path.exists() and history_path.exists():
        previous = pd.read_csv(history_path)
        if "val_loss" in previous and len(previous):
            prior_best = float(previous.val_loss.min())
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        steps_per_epoch=int(cfg.FINETUNE_MIX_STEPS),
        validation_steps=int(math.ceil(n_valid / cfg.BATCH_SIZE_FINETUNE)),
        epochs=int(cfg.FINETUNE_EPOCHS),
        callbacks=[
            Progress(run_dir / "progress.json"),
            tf.keras.callbacks.ModelCheckpoint(
                best_path, save_weights_only=True, save_best_only=False,
                monitor="val_loss", mode="min",
                initial_value_threshold=prior_best,
            ),
            tf.keras.callbacks.CSVLogger(history_path, append=bool(p3a)),
            *([tf.keras.callbacks.BackupAndRestore(str(run_dir / "keras_backup"))] if p3a else []),
        ],
        verbose=2,
    )
    model.load_weights(best_path)
    if p3a:
        val_losses = pd.read_csv(history_path).val_loss.astype(float).tolist()
    else:
        val_losses = [float(v) for v in history.history["val_loss"]]
    payload = {
        "dataset": args.dataset,
        "arm": args.arm,
        "training_seed": args.seed,
        "backbone": args.backbone,
        "selection_objective": "fixed end of epoch10, independent of validation loss",
        "pretrain_checkpoint_sha256": sha256(pretrain_ckpt),
        "selected_epoch": 10,
        "validation_best_epoch_for_monitoring_only": int(np.argmin(val_losses) + 1),
        "optimizer_updates": int(model.optimizer.iterations.numpy()),
        "protocol_sha256": sha256(OUT / "protocol.json"),
        "training_execution": "original TensorFlow/Keras compile settings and default precision",
        "training_jit_compile": bool(model.jit_compile),
        "training_tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "epochs_completed": len(val_losses),
        "val_losses": val_losses,
        "n_corpus_train_molecules": int(len(train)),
        "n_corpus_encoded": int(n_corpus),
        "n_high_pool_encoded": int(n_hi),
        "n_low_pool_encoded": int(n_lo),
        "p3a": p3a,
        "loss_weight_enabled": loss_weight_enabled if args.arm == "soft" else False,
        "high_pool_probability": high_pool_probability if args.arm == "soft" else None,
        "reddb_mix_ratio": float(cfg.FINETUNE_MIX_RATIO),
        "steps_per_epoch": int(cfg.FINETUNE_MIX_STEPS),
        "n_chembl_encoded": int(n_chembl),
        "n_common_validation": int(n_valid),
        "checkpoint": best_path.name,
        "checkpoint_sha256": sha256(best_path),
        **influence_stats(
            train, args.arm,
            weights=(np.where(loss_weight_enabled, train.property_weight.to_numpy(float),
                              1.0) if args.arm == "soft" else None),
            threshold=high_pool_threshold,
            p_high=high_pool_probability if args.arm == "soft" else HIGH_POOL_PROBABILITY,
        ),
    }
    assert payload["epochs_completed"] == payload["selected_epoch"] == 10
    assert payload["optimizer_updates"] == 8000
    complete.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
