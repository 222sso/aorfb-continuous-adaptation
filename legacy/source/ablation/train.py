# -*- coding: utf-8 -*-
import os
import time
from typing import Dict, Tuple, Any

import numpy as np
import tensorflow as tf

from .config import CFG
from .models import compile_model
from .datasets import (
    make_tf_dataset,
    make_tf_dataset_with_w,
    make_tf_dataset_with_unit_w,
    split_by_w,
    mix_two_datasets,
)


class FinetuneSnapshotCallback(tf.keras.callbacks.Callback):
    def __init__(self, outdir: str, run_id: str, target_steps):
        super().__init__()
        target_steps = sorted({int(s) for s in (target_steps or []) if int(s) > 0})
        self.outdir = outdir
        self.run_id = run_id
        self.target_steps = target_steps
        self.global_step = 0
        self.saved = []

    def on_train_batch_end(self, batch, logs=None):
        self.global_step += 1
        if self.global_step not in self.target_steps:
            return
        path = os.path.join(
            self.outdir,
            f"ckpt_finetune_step{self.global_step:04d}_{self.run_id}.weights.h5",
        )
        self.model.save_weights(path)
        self.saved.append({"step": int(self.global_step), "path": path})
        print(f"[SNAPSHOT] saved finetune step={self.global_step} -> {path}")


# ============================================================
# Run ID (unique ckpt naming)
# ============================================================
def make_run_id(prefix: str, params: Dict[str, Any]) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    core = "_".join([f"{k}{params[k]}" for k in sorted(params.keys())])
    return f"{prefix}_{ts}_{core}"


# ============================================================
# Debug helpers
# ============================================================
def _weight_stats(name: str, w: np.ndarray) -> None:
    if w is None:
        print(f"[WSTAT] {name}: w=None")
        return
    w = np.asarray(w, dtype=np.float32)
    if w.size == 0:
        print(f"[WSTAT] {name}: empty")
        return
    print(
        f"[WSTAT] {name}: n={w.size} "
        f"min={float(np.min(w)):.4f} max={float(np.max(w)):.4f} "
        f"mean={float(np.mean(w)):.4f} "
        f"p50={float(np.percentile(w, 50)):.4f} p90={float(np.percentile(w, 90)):.4f} p99={float(np.percentile(w, 99)):.4f} "
        f"ratio(w>1)={float(np.mean(w > 1.0)):.4f}"
    )


def _print_cfg_ft(cfg: CFG, train_params: dict) -> None:
    mix_ratio = float(getattr(cfg, "FINETUNE_MIX_RATIO", train_params.get("FINETUNE_MIX_RATIO", 0.7)))
    steps = getattr(cfg, "FINETUNE_MIX_STEPS", train_params.get("FINETUNE_MIX_STEPS", None))
    w_thr = float(getattr(cfg, "REDDB_W_THR", 1.0))
    p_hi = float(getattr(cfg, "REDDB_P_HI", 0.9))

    print("[CFG/FT] MIX_RATIO=", mix_ratio, "STEPS_PER_EPOCH=", steps, "REDDB_W_THR=", w_thr, "REDDB_P_HI=", p_hi)


# ============================================================
# Train once: pretrain -> finetune
#   - baseline(use_ft_weights=True):
#       RedDB: split_by_w -> hi/lo datasets with sample_weight -> oversample mix
#   - no-weight(use_ft_weights=False):
#       RedDB: NO split_by_w, NO sample_weight, NO oversample (unit weight only)
# ============================================================
def train_once(
    model,
    stoi,
    cfg: CFG,
    train_params: dict,
    chembl_train,
    chembl_valid,
    reddb_train,
    reddb_train_w,
    reddb_valid,
    reddb_valid_w,
    run_prefix: str = "TL",
    use_ft_weights: bool = True,
    finetune_snapshot_steps=None,
    finetune_use_early_stopping: bool = True,
):
    """
    - pretrain: ChEMBL train/valid (X,Y)
    - finetune:
        (A) baseline(use_ft_weights=True)
            RedDB hi/lo split by w_thr -> oversample mix (p_hi) + sample_weight in loss
        (B) no-weight(use_ft_weights=False)
            RedDB uses SAME smiles, but:
              - no split_by_w
              - no oversample
              - no sample_weight (unit weights only)
    - finetune valid: RedDB valid only (baseline: weighted / no-weight: unit)
    - ckpt: unique run_id based file names
    """

    # -------------------------
    # params
    # -------------------------
    seed = int(train_params["SEED"])
    bs_pre = int(train_params["BATCH_SIZE"])
    bs_ft = int(train_params["BATCH_SIZE_FINETUNE"])

    PRETRAIN_EPOCHS = int(train_params["PRETRAIN_EPOCHS"])
    FINETUNE_EPOCHS = int(train_params["FINETUNE_EPOCHS"])
    LR_PRETRAIN = float(train_params["LR_PRETRAIN"])
    LR_FINETUNE = float(train_params["LR_FINETUNE"])

    MIX_RATIO = float(getattr(cfg, "FINETUNE_MIX_RATIO", train_params.get("FINETUNE_MIX_RATIO", 0.7)))
    STEPS_PER_EPOCH = getattr(cfg, "FINETUNE_MIX_STEPS", train_params.get("FINETUNE_MIX_STEPS", None))
    REDDB_W_THR = float(getattr(cfg, "REDDB_W_THR", 1.0))
    REDDB_P_HI = float(getattr(cfg, "REDDB_P_HI", 0.9))

    # -------------------------
    # Debug header
    # -------------------------
    print("\n==================== [train_once] ====================")
    print("[MODE] use_ft_weights =", bool(use_ft_weights))
    print("[SEED]", seed)
    print("[BS] pretrain =", bs_pre, "finetune =", bs_ft)
    print("[EPOCHS] pre =", PRETRAIN_EPOCHS, "ft =", FINETUNE_EPOCHS)
    print("[LR] pre =", LR_PRETRAIN, "ft =", LR_FINETUNE)
    _print_cfg_ft(cfg, train_params)

    # -------------------------
    # pretrain datasets (ChEMBL)
    # -------------------------
    chembl_train_ds, n_chem_tr = make_tf_dataset(
        chembl_train, stoi=stoi, cfg=cfg,
        max_len=cfg.MAX_LEN, batch_size=bs_pre, shuffle=True, seed=seed + 10
    )
    chembl_valid_ds, n_chem_va = make_tf_dataset(
        chembl_valid, stoi=stoi, cfg=cfg,
        max_len=cfg.MAX_LEN, batch_size=bs_pre, shuffle=False, seed=seed + 11
    )

    print("[DATA] ChEMBL n_train =", n_chem_tr, "n_valid =", n_chem_va)
    if n_chem_tr == 0:
        raise RuntimeError("ChEMBL train dataset is empty.")

    # -------------------------
    # finetune datasets
    # -------------------------
    # (A) baseline: weighted + oversample via hi/lo split
    if use_ft_weights:
        _weight_stats("RedDB train (raw)", reddb_train_w)
        _weight_stats("RedDB valid (raw)", reddb_valid_w)

        (hi_smi, hi_w), (lo_smi, lo_w) = split_by_w(reddb_train, reddb_train_w, w_thr=REDDB_W_THR)
        print(f"[RedDB split_by_w] w_thr={REDDB_W_THR} -> hi={len(hi_smi)} lo={len(lo_smi)} "
              f"(hi_ratio={len(hi_smi)/max(1,len(reddb_train)):.4f})")

        # datasets with sample_weight
        reddb_hi_ds, n_hi = make_tf_dataset_with_w(
            hi_smi, hi_w, stoi=stoi, cfg=cfg,
            max_len=cfg.MAX_LEN, batch_size=bs_ft, shuffle=True, seed=seed + 120
        )
        reddb_lo_ds, n_lo = make_tf_dataset_with_w(
            lo_smi, lo_w, stoi=stoi, cfg=cfg,
            max_len=cfg.MAX_LEN, batch_size=bs_ft, shuffle=True, seed=seed + 121
        )
        print("[DATA] RedDB hi batches ~", n_hi, "lo batches ~", n_lo)

        if n_hi == 0 and n_lo == 0:
            raise RuntimeError("RedDB train split is empty. Check REDDB_DF and w settings.")
        elif n_lo == 0:
            print("[RedDB oversample] lo=0 -> using hi only")
            reddb_train_ds = reddb_hi_ds.repeat()
        elif n_hi == 0:
            print("[RedDB oversample] hi=0 -> using lo only")
            reddb_train_ds = reddb_lo_ds.repeat()
        else:
            # hi:lo = REDDB_P_HI:(1-REDDB_P_HI)
            print(f"[RedDB oversample] mixing hi/lo with p_hi={REDDB_P_HI}")
            reddb_train_ds = mix_two_datasets(reddb_hi_ds, reddb_lo_ds, p_a=REDDB_P_HI, seed=seed + 122)

        # finetune valid: RedDB only, weighted
        reddb_valid_ds, n_red_va = make_tf_dataset_with_w(
            reddb_valid, reddb_valid_w, stoi=stoi, cfg=cfg,
            max_len=cfg.MAX_LEN, batch_size=bs_ft, shuffle=False, seed=seed + 13
        )
        ft_valid_ds = reddb_valid_ds

    # (B) no-weight: unit weights only (NO split/oversample/sample_weight)
    else:
        print("[No-weight] Disable split_by_w / oversample / sample_weight (unit weights only)")
        # (중요) 여기서는 reddb_train_w/reddb_valid_w를 아예 쓰지 않는다
        reddb_train_ds, n_red_tr = make_tf_dataset_with_unit_w(
            reddb_train, stoi=stoi, cfg=cfg,
            max_len=cfg.MAX_LEN, batch_size=bs_ft, shuffle=True, seed=seed + 120
        )
        reddb_valid_ds, n_red_va = make_tf_dataset_with_unit_w(
            reddb_valid, stoi=stoi, cfg=cfg,
            max_len=cfg.MAX_LEN, batch_size=bs_ft, shuffle=False, seed=seed + 13
        )

        print("[DATA] RedDB unit-weight n_train_batches ~", n_red_tr, "n_valid_batches ~", n_red_va)
        if n_red_tr == 0:
            raise RuntimeError("RedDB train dataset is empty.")

        reddb_train_ds = reddb_train_ds.repeat()
        ft_valid_ds = reddb_valid_ds

    # -------------------------
    # ChEMBL unit-weight dataset for finetune mix (always unit)
    # -------------------------
    chembl_ft_train_ds, n_chem_ft_tr = make_tf_dataset_with_unit_w(
        chembl_train, stoi=stoi, cfg=cfg,
        max_len=cfg.MAX_LEN, batch_size=bs_ft, shuffle=True, seed=seed + 14
    )
    chembl_ft_valid_ds, n_chem_ft_va = make_tf_dataset_with_unit_w(
        chembl_valid, stoi=stoi, cfg=cfg,
        max_len=cfg.MAX_LEN, batch_size=bs_ft, shuffle=False, seed=seed + 15
    )
    print("[DATA] ChEMBL(unit for FT) n_train_batches ~", n_chem_ft_tr, "n_valid_batches ~", n_chem_ft_va)

    # -------------------------
    # finetune mix: (RedDB) : (ChEMBL) = MIX_RATIO
    # -------------------------
    print(f"[FT MIX] sample_from_datasets weights = [RedDB {MIX_RATIO:.3f}, ChEMBL {1.0 - MIX_RATIO:.3f}]")
    ft_train_ds = tf.data.Dataset.sample_from_datasets(
        [reddb_train_ds, chembl_ft_train_ds.repeat()],
        weights=[MIX_RATIO, 1.0 - MIX_RATIO],
        seed=seed + 200
    ).prefetch(tf.data.AUTOTUNE)

    # steps_per_epoch / validation_steps
    if STEPS_PER_EPOCH is None:
        if n_red_va == 0:
            raise RuntimeError("RedDB valid dataset is empty -> cannot infer steps.")
        STEPS_PER_EPOCH = int(np.ceil(n_red_va / bs_ft)) * 5
        print("[STEPS] STEPS_PER_EPOCH inferred =", STEPS_PER_EPOCH)
    STEPS_PER_EPOCH = int(STEPS_PER_EPOCH)

    VAL_STEPS = int(np.ceil(n_red_va / bs_ft)) if n_red_va > 0 else 0
    print("[STEPS] steps_per_epoch =", STEPS_PER_EPOCH, "val_steps =", VAL_STEPS)

    # -------------------------
    # run_id & ckpt paths
    # -------------------------
    run_id = make_run_id(run_prefix, {
        "seed": seed,
        "pe": PRETRAIN_EPOCHS,
        "fe": FINETUNE_EPOCHS,
        "lrp": LR_PRETRAIN,
        "lrf": LR_FINETUNE,
        "bsp": bs_pre,
        "bsf": bs_ft,
        "mix": round(MIX_RATIO, 3),
        "spe": STEPS_PER_EPOCH,
        "w": int(bool(use_ft_weights)),
    })

    pre_ckpt = os.path.join(cfg.OUTDIR, f"ckpt_pretrain_{run_id}.weights.h5")
    ft_ckpt  = os.path.join(cfg.OUTDIR, f"ckpt_finetune_{run_id}.weights.h5")
    print("[CKPT]")
    print(" - pre:", pre_ckpt)
    print(" - ft :", ft_ckpt)

    # -------------------------
    # pretrain
    # -------------------------
    print("\n[PRETRAIN] start ...")
    compile_model(model, LR_PRETRAIN)
    model.fit(
        chembl_train_ds,
        validation_data=chembl_valid_ds,
        epochs=PRETRAIN_EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                pre_ckpt, save_weights_only=True, save_best_only=True,
                monitor="val_loss", mode="min", verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            ),
        ],
        verbose=1
    )

    # -------------------------
    # finetune
    # -------------------------
    print("\n[FINETUNE] start ...")
    compile_model(model, LR_FINETUNE)
    finetune_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ft_ckpt, save_weights_only=True, save_best_only=True,
            monitor="val_loss", mode="min", verbose=0
        ),
    ]
    snapshot_cb = None
    if finetune_snapshot_steps:
        snapshot_cb = FinetuneSnapshotCallback(cfg.OUTDIR, run_id, finetune_snapshot_steps)
        finetune_callbacks.append(snapshot_cb)
    if finetune_use_early_stopping:
        finetune_callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        )
    model.fit(
        ft_train_ds,
        validation_data=ft_valid_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=(VAL_STEPS if VAL_STEPS > 0 else None),
        epochs=FINETUNE_EPOCHS,
        callbacks=finetune_callbacks,
        verbose=1
    )

    model._finetune_snapshots = list(snapshot_cb.saved) if snapshot_cb is not None else []
    print("\n[train_once] done:", run_id)
    return model, run_id, pre_ckpt, ft_ckpt
