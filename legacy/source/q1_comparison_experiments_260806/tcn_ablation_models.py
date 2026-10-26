from __future__ import annotations

import tensorflow as tf


VARIANTS = ("full", "reduced", "local", "param_matched_mlp", "no_tcn")


def _residual_tcn(x, channels: int, dilations: tuple[int, ...], dropout: float):
    for d in dilations:
        residual = x
        x = tf.keras.layers.Conv1D(
            channels,
            3,
            padding="causal",
            dilation_rate=d,
            activation="relu",
            name=f"tcn_conv_d{d}",
        )(x)
        x = tf.keras.layers.Dropout(dropout, name=f"tcn_drop_d{d}")(x)
        if residual.shape[-1] != x.shape[-1]:
            residual = tf.keras.layers.Dense(channels, name=f"tcn_resproj_d{d}")(residual)
        x = tf.keras.layers.Add(name=f"tcn_add_d{d}")([x, residual])
    return x


def build_tcn_ablation(
    variant: str,
    vocab_size: int,
    max_len: int,
    emb_dim: int = 128,
    channels: int = 256,
    lstm_units: int = 256,
    dropout: float = 0.1,
):
    """Build one matched TCN-front-end ablation.

    The position-wise MLP has 722,131 trainable parameters, versus 722,176
    in the four-block TCN front-end (45 fewer; 0.006% difference). It mixes
    channels at each token independently and therefore adds capacity without
    adding a temporal receptive field.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant={variant!r}; expected one of {VARIANTS}")

    inp = tf.keras.Input(shape=(max_len - 1,), dtype=tf.int32, name="inp")
    x = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=False, name="emb")(inp)

    if variant == "full":
        x = _residual_tcn(x, channels, (1, 2, 4, 8), dropout)
    elif variant == "reduced":
        x = _residual_tcn(x, channels, (1, 2), dropout)
    elif variant == "local":
        x = _residual_tcn(x, channels, (1,), dropout)
    elif variant == "param_matched_mlp":
        x = tf.keras.layers.Dense(1875, activation="relu", name="positionwise_expand")(x)
        x = tf.keras.layers.Dropout(dropout, name="positionwise_drop")(x)
        x = tf.keras.layers.Dense(channels, activation="relu", name="positionwise_project")(x)
    elif variant == "no_tcn":
        pass

    x = tf.keras.layers.LSTM(
        lstm_units,
        return_sequences=True,
        recurrent_dropout=0.1,
        name="lstm",
    )(x)
    x = tf.keras.layers.Dropout(dropout, name="drop")(x)
    out = tf.keras.layers.Dense(vocab_size, activation="softmax", name="out")(x)
    return tf.keras.Model(inp, out, name=f"TCN_ABL_{variant}")


def front_end_parameter_count(model, variant: str) -> int:
    if variant == "no_tcn":
        return 0
    names = (
        "tcn_conv_",
        "tcn_resproj_",
        "positionwise_expand",
        "positionwise_project",
    )
    return int(sum(layer.count_params() for layer in model.layers if layer.name.startswith(names)))
