# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from .config import CFG
from .vocab import encode_smiles_list


def _shuffle_and_batch(ds, n_items: int, batch_size: int, shuffle: bool, seed: int):
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(int(n_items), 20000),
            seed=int(seed),
            reshuffle_each_iteration=True,
        )
    return ds.batch(int(batch_size)).prefetch(tf.data.AUTOTUNE)


def make_tf_dataset(smiles_list, stoi, cfg: CFG, max_len, batch_size, shuffle, seed):
    arr = encode_smiles_list(smiles_list, stoi=stoi, cfg=cfg, max_len=max_len)
    n = int(arr.shape[0])

    if n == 0:
        x = np.zeros((0, max_len - 1), dtype=np.int32)
        y = np.zeros((0, max_len - 1), dtype=np.int32)
        return tf.data.Dataset.from_tensor_slices((x, y)), 0

    x = arr[:, :-1]
    y = arr[:, 1:]
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    return _shuffle_and_batch(ds, n, batch_size, shuffle, seed), n


def make_tf_dataset_with_w(smiles_list, w_list, stoi, cfg: CFG, max_len, batch_size, shuffle, seed):
    if len(smiles_list) == 0:
        x = np.zeros((0, max_len - 1), dtype=np.int32)
        y = np.zeros((0, max_len - 1), dtype=np.int32)
        w = np.zeros((0, max_len - 1), dtype=np.float32)
        return tf.data.Dataset.from_tensor_slices((x, y, w)), 0

    arr = encode_smiles_list(smiles_list, stoi=stoi, cfg=cfg, max_len=max_len)
    n = int(arr.shape[0])
    if n == 0:
        x = np.zeros((0, max_len - 1), dtype=np.int32)
        y = np.zeros((0, max_len - 1), dtype=np.int32)
        w = np.zeros((0, max_len - 1), dtype=np.float32)
        return tf.data.Dataset.from_tensor_slices((x, y, w)), 0

    # encode_smiles_list drops too-long samples when cfg.DROP_TOO_LONG=True.
    # Rebuild aligned weights with the same drop criterion.
    aligned_w = []
    for s, wi in zip(smiles_list, w_list):
        if (len(str(s)) + 2) > int(max_len) and bool(cfg.DROP_TOO_LONG):
            continue
        aligned_w.append(float(wi))
    aligned_w = np.asarray(aligned_w, dtype=np.float32)

    if aligned_w.shape[0] != n:
        raise RuntimeError(
            f"Weight alignment mismatch: encoded_n={n}, aligned_w_n={aligned_w.shape[0]}"
        )

    x = arr[:, :-1]
    y = arr[:, 1:]
    pad_id = int(stoi[cfg.PAD])
    mask = (y != pad_id).astype(np.float32)
    w = (aligned_w[:, None] * mask).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((x, y, w))
    return _shuffle_and_batch(ds, n, batch_size, shuffle, seed), n


def make_tf_dataset_with_unit_w(smiles_list, stoi, cfg: CFG, max_len, batch_size, shuffle, seed):
    arr = encode_smiles_list(smiles_list, stoi=stoi, cfg=cfg, max_len=max_len)
    n = int(arr.shape[0])
    if n == 0:
        x = np.zeros((0, max_len - 1), dtype=np.int32)
        y = np.zeros((0, max_len - 1), dtype=np.int32)
        w = np.zeros((0, max_len - 1), dtype=np.float32)
        return tf.data.Dataset.from_tensor_slices((x, y, w)), 0

    x = arr[:, :-1]
    y = arr[:, 1:]
    pad_id = int(stoi[cfg.PAD])
    w = (y != pad_id).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((x, y, w))
    return _shuffle_and_batch(ds, n, batch_size, shuffle, seed), n


def split_by_w(smiles_list, w_list, w_thr=1.0):
    hi_smi, hi_w, lo_smi, lo_w = [], [], [], []
    for s, w in zip(smiles_list, w_list):
        if float(w) >= float(w_thr):
            hi_smi.append(s)
            hi_w.append(float(w))
        else:
            lo_smi.append(s)
            lo_w.append(float(w))
    return (hi_smi, hi_w), (lo_smi, lo_w)


def repeat_ds(ds):
    return ds.repeat()


def mix_two_datasets(ds_a, ds_b, p_a=0.7, seed=0):
    p_a = float(np.clip(float(p_a), 0.0, 1.0))
    return tf.data.Dataset.sample_from_datasets(
        [repeat_ds(ds_a), repeat_ds(ds_b)],
        weights=[p_a, 1.0 - p_a],
        seed=int(seed),
    ).prefetch(tf.data.AUTOTUNE)


def mix_datasets(ds_reddb, ds_chembl, ratio_reddb=0.7, seed=0):
    ratio_reddb = float(np.clip(float(ratio_reddb), 0.0, 1.0))
    return tf.data.Dataset.sample_from_datasets(
        [repeat_ds(ds_reddb), repeat_ds(ds_chembl)],
        weights=[ratio_reddb, 1.0 - ratio_reddb],
        seed=int(seed),
    ).prefetch(tf.data.AUTOTUNE)

