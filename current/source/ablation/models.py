# -*- coding: utf-8 -*-
import tensorflow as tf
from .config import CFG

# =========================
# 7) Model definitions (원래 코드 스타일)
# =========================


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen: int, vocab_size: int, emb_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = int(maxlen)
        self.vocab_size = int(vocab_size)
        self.emb_dim = int(emb_dim)
        self.token_emb = tf.keras.layers.Embedding(vocab_size, emb_dim, name="tok_emb")
        self.pos_emb = tf.keras.layers.Embedding(maxlen, emb_dim, name="pos_emb")

    def call(self, x):
        seq_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos = self.pos_emb(positions)
        tok = self.token_emb(x)
        return tok + pos


class CausalTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_dim: int, num_heads: int, ff_dim: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=emb_dim // num_heads,
            dropout=dropout,
            name="mha",
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu", name="ff1"),
                tf.keras.layers.Dropout(dropout, name="ff_drop1"),
                tf.keras.layers.Dense(emb_dim, name="ff2"),
            ],
            name="ffn",
        )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln1")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln2")
        self.drop1 = tf.keras.layers.Dropout(dropout, name="attn_drop")
        self.drop2 = tf.keras.layers.Dropout(dropout, name="ff_drop2")

    def call(self, x, training=None):
        attn = self.att(x, x, use_causal_mask=True, training=training)
        attn = self.drop1(attn, training=training)
        x = self.norm1(x + attn)
        ff = self.ffn(x, training=training)
        ff = self.drop2(ff, training=training)
        return self.norm2(x + ff)

def build_A_TCN(vocab_size: int, cfg: CFG,
                emb_dim: int = 128,
                tcn_channels: int = 256,
                tcn_kernel: int = 3,
                tcn_dilations=(1, 2, 4, 8),
                dropout: float = 0.1):
    inp = tf.keras.Input(shape=(cfg.MAX_LEN - 1,), dtype=tf.int32, name="inp")
    x = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=False, name="emb")(inp)

    for di, d in enumerate(tcn_dilations):
        x_res = x
        x = tf.keras.layers.Conv1D(
            tcn_channels, tcn_kernel,
            padding="causal",
            dilation_rate=d,
            activation="relu",
            name=f"tcn_conv_d{d}"
        )(x)
        x = tf.keras.layers.Dropout(dropout, name=f"tcn_drop_d{d}")(x)
        if x_res.shape[-1] != x.shape[-1]:
            x_res = tf.keras.layers.Dense(tcn_channels, name=f"tcn_resproj_d{d}")(x_res)
        x = tf.keras.layers.Add(name=f"tcn_add_d{d}")([x, x_res])

    out = tf.keras.layers.Dense(vocab_size, activation="softmax", name="out")(x)
    m = tf.keras.Model(inp, out, name="A_TCN")
    return m

def build_B_LSTM(vocab_size: int, cfg: CFG,
                 emb_dim: int = 128,
                 lstm_units: int = 256,
                 dropout: float = 0.1):
    inp = tf.keras.Input(shape=(cfg.MAX_LEN - 1,), dtype=tf.int32, name="inp")
    x = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=False, name="emb")(inp)
    # ✅ CuDNN 오류 회피
    x = tf.keras.layers.LSTM(lstm_units, return_sequences=True, recurrent_dropout=0.1, name="lstm")(x)
    x = tf.keras.layers.Dropout(dropout, name="drop")(x)
    out = tf.keras.layers.Dense(vocab_size, activation="softmax", name="out")(x)
    m = tf.keras.Model(inp, out, name="B_LSTM")
    return m

def build_C_TCN_LSTM(vocab_size: int, cfg: CFG,
                     emb_dim: int = 128,
                     tcn_channels: int = 128,
                     tcn_kernel: int = 3,
                     tcn_dilations=(1, 2, 4),
                     lstm_units: int = 256,
                     dropout: float = 0.2):
    inp = tf.keras.Input(shape=(cfg.MAX_LEN - 1,), dtype=tf.int32, name="inp")
    x = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=False, name="emb")(inp)

    for d in tcn_dilations:
        x_res = x
        x = tf.keras.layers.Conv1D(
            tcn_channels, tcn_kernel,
            padding="causal",
            dilation_rate=d,
            activation="relu",
            name=f"tcn_conv_d{d}"
        )(x)
        x = tf.keras.layers.Dropout(dropout, name=f"tcn_drop_d{d}")(x)
        if x_res.shape[-1] != x.shape[-1]:
            x_res = tf.keras.layers.Dense(tcn_channels, name=f"tcn_resproj_d{d}")(x_res)
        x = tf.keras.layers.Add(name=f"tcn_add_d{d}")([x, x_res])

    # ✅ CuDNN 오류 회피
    x = tf.keras.layers.LSTM(lstm_units, return_sequences=True, recurrent_dropout=0.1, name="lstm")(x)
    x = tf.keras.layers.Dropout(dropout, name="drop")(x)
    out = tf.keras.layers.Dense(vocab_size, activation="softmax", name="out")(x)
    m = tf.keras.Model(inp, out, name="C_TCN_LSTM")
    return m


def build_D_Transformer(vocab_size: int, cfg: CFG,
                        emb_dim: int = 128,
                        num_heads: int = 4,
                        ff_dim: int = 512,
                        num_layers: int = 4,
                        dropout: float = 0.1):
    if emb_dim % num_heads != 0:
        raise ValueError(f"emb_dim({emb_dim}) must be divisible by num_heads({num_heads}).")

    inp = tf.keras.Input(shape=(cfg.MAX_LEN - 1,), dtype=tf.int32, name="inp")
    x = TokenAndPositionEmbedding(
        maxlen=cfg.MAX_LEN - 1,
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        name="tokpos_emb",
    )(inp)
    x = tf.keras.layers.Dropout(dropout, name="emb_drop")(x)

    for i in range(int(num_layers)):
        x = CausalTransformerBlock(
            emb_dim=emb_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f"tr_block_{i+1}",
        )(x)

    out = tf.keras.layers.Dense(vocab_size, activation="softmax", name="out")(x)
    m = tf.keras.Model(inp, out, name="D_Transformer")
    return m

def compile_model(model, lr: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy"
    )
    return model
