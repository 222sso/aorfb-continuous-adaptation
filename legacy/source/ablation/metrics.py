# -*- coding: utf-8 -*-
from typing import Optional, Dict, List, Set
import numpy as np

# RDKit
try:
    from rdkit import Chem
    RDKit_OK = True
except Exception:
    RDKit_OK = False

def canonicalize_smiles(s: str) -> Optional[str]:
    if not RDKit_OK:
        return s
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None

def build_train_canon_set(train_plain_smiles: List[str]) -> Set[str]:
    """
    novelty reference set용 canonical set.
    train_plain_smiles는 decode_special까지 끝난 'plain SMILES'여야 함.
    """
    out = set()
    for s in train_plain_smiles:
        c = canonicalize_smiles(s)
        if c:
            out.add(c)
    return out

def eval_counts_canonical(generated: List[str], reference_can_set: Optional[set] = None) -> Dict[str, int]:
    can_list = []
    for s in generated:
        can = canonicalize_smiles(s)
        if can:
            can_list.append(can)

    valid = len(can_list)
    uniq = len(set(can_list))

    if reference_can_set is None:
        novel = 0
    else:
        novel = len([c for c in set(can_list) if c not in reference_can_set])

    return {"valid": valid, "unique": uniq, "novel": novel}

def rdkit_metrics(generated_plain: List[str], reference_can_set: Optional[set]) -> Dict[str, float]:
    """
    generated_plain: decode_special까지 된 plain smiles 리스트
    reference_can_set: train canonical set (novelty 기준)
    """
    n = int(len(generated_plain))
    cnt = eval_counts_canonical(generated_plain, reference_can_set=reference_can_set)

    valid = int(cnt["valid"])
    unique = int(cnt["unique"])
    novel = int(cnt["novel"])

    # sanitize fail(=MolFromSmiles 실패) 비율을 ring/branch 오류율로 사용
    sanitize_fail = n - valid

    # 비율 정의:
    # - valid_rate: valid / n
    # - unique_rate: unique / valid  (valid 내에서 unique)
    # - novelty_rate: novel / unique (unique 내에서 novel)
    valid_rate = (valid / n) if n > 0 else 0.0
    unique_rate = (unique / valid) if valid > 0 else 0.0
    novelty_rate = (novel / unique) if unique > 0 else 0.0
    sanitize_fail_rate = (sanitize_fail / n) if n > 0 else 0.0

    return {
        "n_gen": n,
        "valid": valid,
        "unique": unique,
        "novel": novel,
        "valid_rate": float(valid_rate),
        "unique_rate": float(unique_rate),
        "novelty_rate": float(novelty_rate),
        "sanitize_fail": int(sanitize_fail),
        "sanitize_fail_rate": float(sanitize_fail_rate),
    }