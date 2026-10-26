from __future__ import annotations

"""One place that resolves a backbone name to a built model.

Every script in this experiment used to construct A_TCN directly, which made the whole
sweep single-backbone without any argument recording that it was. The support arms are the
comparison most exposed to that, because the new-scaffold endpoint is where the main text
measures the largest backbone spread, so the name is now an argument and is written into
the run metadata.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ablation.config import CFG  # noqa: E402
import tensorflow as tf  # noqa: E402

from ablation.models import build_A_TCN, build_B_LSTM, build_C_TCN_LSTM  # noqa: E402

CHOICES = ("A_TCN", "B_LSTM", "B_LSTM_fast", "C_TCN_LSTM")
DEFAULT = "A_TCN"


def build(name: str, vocab_size: int, cfg: CFG):
    if name == "A_TCN":
        return build_A_TCN(vocab_size, cfg, emb_dim=128, tcn_channels=256, tcn_kernel=3,
                           tcn_dilations=(1, 2, 4, 8), dropout=0.1)
    if name == "B_LSTM":
        return build_B_LSTM(vocab_size, cfg, emb_dim=128, lstm_units=256, dropout=0.1)
    if name == "B_LSTM_fast":
        # The published LSTM sets recurrent_dropout, which rules out the cuDNN kernel and
        # costs 390 s per pretraining epoch here. Only one quantity is read from this
        # backbone -- the contrast between two adaptation sets that share a checkpoint --
        # and a recurrent regulariser common to both conditions cannot produce it, so it
        # is dropped to make the check affordable. Everything else matches build_B_LSTM.
        inp = tf.keras.Input(shape=(cfg.MAX_LEN - 1,), dtype=tf.int32, name="inp")
        x = tf.keras.layers.Embedding(vocab_size, 128, mask_zero=False, name="emb")(inp)
        x = tf.keras.layers.LSTM(256, return_sequences=True, name="lstm")(x)
        x = tf.keras.layers.Dropout(0.1, name="drop")(x)
        out = tf.keras.layers.Dense(vocab_size, activation="softmax", name="out")(x)
        return tf.keras.Model(inp, out, name="B_LSTM_fast")
    if name == "C_TCN_LSTM":
        return build_C_TCN_LSTM(vocab_size, cfg, emb_dim=128, tcn_channels=128,
                                tcn_kernel=3, dropout=0.1)
    raise ValueError(f"unknown backbone {name!r}; choose from {CHOICES}")


def add_argument(parser) -> None:
    parser.add_argument("--backbone", default=DEFAULT, choices=CHOICES,
                        help="sequence model between embedding and output")


def pretrain_meta_name(backbone: str) -> str:
    """Checkpoint metadata written by pretrain_shared.py for this backbone."""
    return "pretrain_complete.json" if backbone == DEFAULT \
        else f"pretrain_complete_{backbone}_union.json"
