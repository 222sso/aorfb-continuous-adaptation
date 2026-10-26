# -*- coding: utf-8 -*-
import numpy as np
from typing import Dict, List, Tuple, Optional
from .config import CFG

def build_char_vocab(smiles_list: List[str], cfg: CFG) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = set()
    for s in smiles_list:
        chars.update(list(s))
    tokens = [cfg.PAD, cfg.BOS, cfg.EOS, cfg.UNK] + sorted(chars)
    stoi = {t: i for i, t in enumerate(tokens)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos

# =========================
# 5) Vocab / Encode / Decode + length stats (원래 코드와 동일)
# =========================
def encode_smiles_one(smiles: str, stoi: Dict[str, int], cfg: CFG,
                      bos_id: int, eos_id: int, pad_id: int, unk_id: int,
                      max_len: int) -> List[int]:
    """
    max_len: 최종 시퀀스 길이(패딩 포함). BOS/EOS 포함해서 max_len에 맞춤.
    DROP_TOO_LONG=True면 max_len 초과 시 [] 반환 (drop)
    """
    ids = [bos_id] + [stoi.get(ch, unk_id) for ch in str(smiles)] + [eos_id]
    if len(ids) > max_len:
        if cfg.DROP_TOO_LONG:
            return []
        ids = ids[:max_len]
        ids[-1] = eos_id
    ids = ids + [pad_id] * (max_len - len(ids))
    return ids

def decode_ids(ids: List[int], itos: Dict[int, str], cfg: CFG) -> str:
    toks = [itos.get(int(i), cfg.UNK) for i in ids]
    toks = [t for t in toks if t not in (cfg.PAD, cfg.BOS, cfg.EOS)]
    return "".join(toks)

def encode_smiles_list(smiles_list: List[str], stoi: Dict[str, int], cfg: CFG, max_len: int) -> np.ndarray:
    """
    리스트를 (N, max_len) int32로 변환.
    DROP_TOO_LONG=True면 길이 초과 샘플은 drop.
    """
    bos_id = stoi[cfg.BOS]
    eos_id = stoi[cfg.EOS]
    pad_id = stoi[cfg.PAD]
    unk_id = stoi[cfg.UNK]

    out = []
    for s in smiles_list:
        ids = encode_smiles_one(s, stoi, cfg, bos_id, eos_id, pad_id, unk_id, max_len=max_len)
        if ids:  # []면 drop
            out.append(ids)

    if len(out) == 0:
        return np.zeros((0, max_len), dtype=np.int32)

    return np.asarray(out, dtype=np.int32)

def make_lm_xy(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return X[:, :-1], X[:, 1:]

def length_stats(name: str, lst: List[str], cfg: CFG):
    lens = np.array([len(s) + 2 for s in lst], dtype=int)  # +2 for BOS/EOS
    if len(lens) == 0:
        print(f"[{name}] n=0")
        return
    print(f"[{name}] n={len(lens)} min={lens.min()} p50={int(np.percentile(lens,50))} "
          f"p90={int(np.percentile(lens,90))} p99={int(np.percentile(lens,99))} max={lens.max()}")