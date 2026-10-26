# -*- coding: utf-8 -*-
import numpy as np
from typing import List

from .config import CFG
from .data import decode_special  # 너가 이미 만들어둔 decode_special이 있다고 가정

def top_p_sample(probs: np.ndarray, top_p: float, temperature: float) -> int:
    """
    probs: softmax 확률 벡터 (V,)
    top_p: nucleus sampling threshold
    temperature: >0
    """
    probs = np.asarray(probs, dtype=np.float64)
    # temperature 적용을 log-space로(원본 그대로)
    p = np.log(np.maximum(probs, 1e-12)) / max(float(temperature), 1e-8)
    p = np.exp(p - p.max())
    p = p / np.sum(p)

    idx = np.argsort(p)[::-1]
    cdf = np.cumsum(p[idx])
    cut = int(np.searchsorted(cdf, float(top_p))) + 1
    cut = max(1, min(cut, len(idx)))

    keep = idx[:cut]
    kp = p[keep]
    kp = kp / np.sum(kp)

    return int(np.random.choice(keep, p=kp))

def generate_smiles_one(model, itos, stoi, cfg: CFG, max_len: int, top_p: float, temperature: float) -> str:
    """
    원본 로직:
      - 입력 x shape: (1, max_len-1)
      - seq는 [bos]로 시작
      - 매 step마다 model.predict(x)에서 t 위치 확률을 뽑아 top-p 샘플
      - eos 나오면 중단
      - bos/eos/pad 제외 후 join
    """
    pad_id = stoi[cfg.PAD]
    bos_id = stoi[cfg.BOS]
    eos_id = stoi[cfg.EOS]

    seq = [bos_id]
    for _ in range(max_len - 1):
        x = np.full((1, max_len - 1), pad_id, dtype=np.int32)
        t = min(len(seq) - 1, max_len - 2)
        x[0, :len(seq)] = np.asarray(seq[:max_len - 1], dtype=np.int32)

        y = model.predict(x, verbose=0)  # (1, T, V) with softmax
        probs = y[0, t, :]
        nxt = top_p_sample(probs, top_p=top_p, temperature=temperature)

        seq.append(nxt)
        if nxt == eos_id:
            break

    toks = [itos[i] for i in seq if i not in (bos_id, eos_id, pad_id)]
    return "".join(toks)

def generate_smiles(model, itos, stoi, cfg: CFG, gen_n: int, temperature: float, top_p: float) -> List[str]:
    """
    run_ablation_abc.py에서 쓰는 형태로 맞춘 wrapper.
    반환은 "토큰화된 SMILES 문자열"(special token 포함) 리스트.
    """
    out = []
    for _ in range(int(gen_n)):
        s = generate_smiles_one(
            model=model,
            itos=itos,
            stoi=stoi,
            cfg=cfg,
            max_len=cfg.MAX_LEN,
            top_p=float(top_p),
            temperature=float(temperature),
        )
        out.append(s)
    return out

def detokenize_generated(tok_smiles_list: List[str], cfg: CFG) -> List[str]:
    """TOK_SO3H/COOH를 원래 문자열로 복원(decode_special)."""
    return [decode_special(s, cfg) for s in tok_smiles_list]