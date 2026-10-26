from __future__ import annotations

"""One shared A_TCN ChEMBL pretraining run on the union vocabulary.

Every dataset/arm/seed in this experiment starts from this single checkpoint, so
vocabulary and pretraining provenance are identical across all comparisons.
"""

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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from ablation.config import CFG  # noqa: E402
from ablation.datasets import make_tf_dataset_with_unit_w  # noqa: E402
from ablation.models import compile_model  # noqa: E402
import backbones  # noqa: E402

PRETRAIN_SEED = 42


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
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="vocab.json")
    ap.add_argument("--stem", default="A_TCN_union")
    backbones.add_argument(ap)
    args = ap.parse_args()
    complete = ARTIFACTS / f"pretrain_complete_{args.stem}.json" if args.stem != "A_TCN_union" \
        else ARTIFACTS / "pretrain_complete.json"
    if complete.exists():
        print("SKIP: shared pretraining already complete")
        return

    cfg = CFG()
    set_seed(PRETRAIN_SEED)

    vocab = json.loads((ARTIFACTS / args.vocab).read_text(encoding="utf-8"))
    stoi = {str(k): int(v) for k, v in vocab["stoi"].items()}
    chembl = pd.read_csv(ARTIFACTS / "chembl.csv.gz")
    smiles = chembl.encoded_smiles.astype(str).tolist()

    rng = np.random.default_rng(PRETRAIN_SEED + 1)
    order = rng.permutation(len(smiles))
    n_valid = max(1, int(len(smiles) * cfg.VALID_RATIO_CHEMBL))
    valid_idx = set(order[:n_valid].tolist())
    train_smiles = [s for i, s in enumerate(smiles) if i not in valid_idx]
    valid_smiles = [s for i, s in enumerate(smiles) if i in valid_idx]

    train_ds, n_train = make_tf_dataset_with_unit_w(
        train_smiles, stoi, cfg, cfg.MAX_LEN, cfg.BATCH_SIZE, True, PRETRAIN_SEED
    )
    valid_ds, n_valid_enc = make_tf_dataset_with_unit_w(
        valid_smiles, stoi, cfg, cfg.MAX_LEN, cfg.BATCH_SIZE, False, PRETRAIN_SEED + 1
    )
    print(f"[pretrain] train={n_train:,} valid={n_valid_enc:,} vocab={len(stoi)}")

    model = backbones.build(args.backbone, len(stoi), cfg)
    compile_model(model, cfg.LR_PRETRAIN)
    ckpt = ARTIFACTS / f"ckpt_pretrain_{args.stem}.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt, save_weights_only=True, save_best_only=True, monitor="val_loss", mode="min"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, mode="min"
        ),
        tf.keras.callbacks.CSVLogger(ARTIFACTS / "pretrain_history.csv"),
    ]
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=int(cfg.PRETRAIN_EPOCHS),
        steps_per_epoch=int(math.ceil(n_train / cfg.BATCH_SIZE)),
        validation_steps=int(math.ceil(n_valid_enc / cfg.BATCH_SIZE)),
        callbacks=callbacks,
        verbose=2,
    )
    val_losses = [float(v) for v in history.history["val_loss"]]
    payload = {
        "backbone": args.backbone,
        "corpus": "ChEMBL 60k",
        "vocabulary": "union(ChEMBL, ZINC250k, QM9)",
        "vocab_size": int(len(stoi)),
        "pretrain_seed": PRETRAIN_SEED,
        "epochs_completed": len(val_losses),
        "selected_epoch": int(np.argmin(val_losses) + 1),
        "best_val_loss": float(min(val_losses)),
        "n_train_encoded": int(n_train),
        "n_valid_encoded": int(n_valid_enc),
        "checkpoint": ckpt.name,
        "checkpoint_sha256": sha256(ckpt),
    }
    complete.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
