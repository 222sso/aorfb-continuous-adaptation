# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from .config import CFG
import os

# RDKit (가능한 경우)
try:
    from rdkit import Chem
    RDKit_OK = True
except Exception:
    RDKit_OK = False

# ============================================================
# 5) Token normalization (SO3H/COOH)  -- 원래 코드와 동일
# ============================================================
SO3H_EQUIV = re.compile(
    r"O=S\(=O\)\(O\)|"
    r"O=S\(O\)\(=O\)|"
    r"S\(=O\)\(O\)=O|"
    r"S\(O\)\(=O\)=O"
)
SO3H_CANON = re.compile(r"S\(=O\)\(=O\)O")

COOH_EQUIV = re.compile(r"O=C\(O\)")
COOH_CANON = re.compile(r"C\(=O\)O")

def encode_special(s: str, cfg: CFG) -> str:
    s = str(s).strip()
    s = SO3H_EQUIV.sub("S(=O)(=O)O", s)
    s = SO3H_CANON.sub(cfg.TOK_SO3H, s)
    s = COOH_EQUIV.sub("C(=O)O", s)
    s = COOH_CANON.sub(cfg.TOK_COOH, s)
    return s

def decode_special(s: str, cfg: CFG) -> str:
    s = str(s)
    s = s.replace(cfg.TOK_SO3H, "S(=O)(=O)O")
    s = s.replace(cfg.TOK_COOH, "C(=O)O")
    return s

def detokenize_to_plain_smiles(s_tok: str, cfg: CFG) -> str:
    # runner/metrics에서 쓰던 API 유지
    return decode_special(s_tok, cfg)

def check_token_stats(lst: List[str], tok: str, name: str):
    if not lst:
        print(f"{name}: empty list")
        return
    n = sum(tok in s for s in lst)
    print(f"{name}: {tok} in {n}/{len(lst)} ({n/len(lst):.3%})")

# ============================================================
# (B0) ChEMBL load + basic filtering + split
# ============================================================
def basic_len_filter(smiles_list: List[str], min_len: int, max_len: int) -> List[str]:
    out = []
    for s in smiles_list:
        if s is None:
            continue
        s = str(s).strip()
        if min_len <= len(s) <= (max_len - 2):  # -2 for BOS/EOS later
            out.append(s)
    return out

def load_smiles_column(csv_path: str, prefer_cols=("smiles","SMILES","canonical_smiles")) -> List[str]:
    df = pd.read_csv(csv_path)
    col = None
    for c in prefer_cols:
        if c in df.columns:
            col = c
            break
    if col is None:
        col = df.columns[0]
    return df[col].astype(str).tolist()

def split_list_train_valid(smiles_list: List[str], valid_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(smiles_list))
    rng.shuffle(idx)
    n_valid = max(1, int(len(idx) * valid_ratio))
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]
    train = [smiles_list[i] for i in train_idx]
    valid = [smiles_list[i] for i in valid_idx]
    return train, valid

def read_chembl_train_valid(cfg: CFG) -> Tuple[List[str], List[str]]:
    chembl_all = load_smiles_column(cfg.CHEMBL_CSV)
    chembl_all = basic_len_filter(chembl_all, cfg.MIN_LEN, cfg.MAX_LEN)

    chembl_train, chembl_valid = split_list_train_valid(chembl_all, cfg.VALID_RATIO_CHEMBL, cfg.SEED + 1)
    print("ChEMBL total/train/valid:", len(chembl_all), len(chembl_train), len(chembl_valid))

    # ✅ 원래 코드 순서: encode_special 적용
    chembl_train = [encode_special(s, cfg) for s in chembl_train]
    chembl_valid = [encode_special(s, cfg) for s in chembl_valid]
    return chembl_train, chembl_valid

# ============================================================
# SoftFT weights loading + RedDB attach weights
# ============================================================
def _find_inchikey_col(df: pd.DataFrame) -> str:
    cand_keys = ["inchiKey", "inchikey", "InChIKey", "INCHIKEY"]
    col = next((c for c in cand_keys if c in df.columns), None)
    if col is None:
        raise ValueError(f"inchiKey 컬럼 없음. cols={list(df.columns)[:60]}")
    return col

def _to_bool_series(x: pd.Series) -> pd.Series:
    """
    다양한 형태(True/False, 1/0, 'true'/'false', 'Y'/'N', NaN)을 안전하게 bool로 변환
    """
    # 이미 bool이면 그대로
    if pd.api.types.is_bool_dtype(x):
        return x.fillna(False)

    # 숫자면 0/1 기준
    if pd.api.types.is_numeric_dtype(x):
        return x.fillna(0).astype(float) != 0

    # 문자열 처리
    s = x.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y"}
    false_set = {"0", "false", "f", "no", "n", "nan", "none", ""}

    out = pd.Series(False, index=x.index, dtype=bool)
    out[s.isin(true_set)] = True
    out[s.isin(false_set)] = False
    # 그 외 값은 보수적으로 False
    return out


def load_softft_weight_map(cfg: CFG) -> Dict[str, float]:
    """
    cfg.WEIGHT_CSV에서 inchiKey -> weight(float) dict 로딩

    - inchiKey 컬럼명은 _find_inchikey_col(wdf)로 탐색
    - cfg.W_USE(use_for_ft) 컬럼이 있으면 True인 행만 사용
    - cfg.W_COL(w) 컬럼이 없으면 에러
    - 동일 key 중복은 weight 최대값으로 통합
    """
    if not getattr(cfg, "WEIGHT_CSV", None):
        raise ValueError("cfg.WEIGHT_CSV가 비어있음")

    if not os.path.exists(cfg.WEIGHT_CSV):
        raise FileNotFoundError(f"softft weight csv 없음: {cfg.WEIGHT_CSV}")

    wdf = pd.read_csv(cfg.WEIGHT_CSV)

    # --- find key column ---
    try:
        W_KEY = _find_inchikey_col(wdf)
    except Exception as e:
        raise ValueError(
            f"_find_inchikey_col 실패: {e}\n"
            f"Available cols(head): {list(wdf.columns)[:80]}"
        )

    if W_KEY not in wdf.columns:
        raise ValueError(
            f"InChIKey 컬럼 '{W_KEY}' 를 찾았다고 했지만 DF에 없음.\n"
            f"cols={list(wdf.columns)[:80]}"
        )

    # normalize key
    wdf[W_KEY] = wdf[W_KEY].astype(str).str.strip().str.upper()
    wdf = wdf[wdf[W_KEY].notna() & (wdf[W_KEY] != "")].copy()

    # --- optional use filter ---
    if getattr(cfg, "W_USE", None) and (cfg.W_USE in wdf.columns):
        before = len(wdf)
        use_mask = _to_bool_series(wdf[cfg.W_USE])
        wdf = wdf[use_mask].copy()
        print(f"[SoftFT] filter by {cfg.W_USE}: {before} -> {len(wdf)}")

    # --- weight col check ---
    if cfg.W_COL not in wdf.columns:
        raise ValueError(
            f"softft 파일에 weight 컬럼('{cfg.W_COL}') 없음.\n"
            f"cols(head)={list(wdf.columns)[:80]}"
        )

    wdf[cfg.W_COL] = pd.to_numeric(wdf[cfg.W_COL], errors="coerce")
    wdf = wdf.dropna(subset=[cfg.W_COL]).copy()

    # group by key, keep max weight
    wdf = wdf.groupby(W_KEY, as_index=False)[cfg.W_COL].max()

    # ensure python float
    keys = wdf[W_KEY].tolist()
    vals = [float(v) for v in wdf[cfg.W_COL].tolist()]
    w_map = dict(zip(keys, vals))

    if vals:
        print(
            "Loaded weights(unique keys):", len(w_map),
            "min/max =", float(np.min(vals)), float(np.max(vals)),
            "mean =", float(np.mean(vals)),
            "w>1 ratio =", float(np.mean(np.array(vals) > 1.0)),
        )
    else:
        print("Loaded weights(unique keys): 0 (empty after filtering)")

    return w_map

def load_reddb_df_with_weights(cfg: CFG, w_map: Dict[str, float]) -> Tuple[pd.DataFrame, str]:
    REDDB_DF = pd.read_csv(cfg.REDDB_CSV)
    SMI_COL = cfg.REDDB_SMILES_COL

    if SMI_COL not in REDDB_DF.columns:
        raise ValueError(f"RedDB CSV에 smiles 컬럼('{SMI_COL}')이 없음. cols={list(REDDB_DF.columns)[:60]}")

    IK_COL = _find_inchikey_col(REDDB_DF)

    REDDB_DF[IK_COL] = REDDB_DF[IK_COL].astype(str).str.strip().str.upper()
    REDDB_DF[SMI_COL] = REDDB_DF[SMI_COL].astype(str)

    # 길이 필터(토큰화 전 기준, BOS/EOS 고려하여 -2)
    REDDB_DF = REDDB_DF[REDDB_DF[SMI_COL].str.len().between(cfg.MIN_LEN, cfg.MAX_LEN - 2)].copy()

    # ✅ weight map 적용 (cfg.W_COL 사용)
    w_col = cfg.W_COL
    use_col = cfg.W_USE

    REDDB_DF[w_col] = REDDB_DF[IK_COL].map(w_map)

    # ✅ use_col을 항상 만들어 둠 (디버깅/추적 편함)
    # - 매칭되면 True, 아니면 False
    REDDB_DF[use_col] = REDDB_DF[w_col].notna()

    if cfg.USE_ONLY_SOFTFT:
        before = len(REDDB_DF)
        # 매칭된 것만 남김 (use_col True)
        REDDB_DF = REDDB_DF[REDDB_DF[use_col].astype(bool)].copy()
        print(f"[USE_ONLY_SOFTFT=True] RedDB filtered by {use_col}: {before} -> {len(REDDB_DF)}")
    else:
        # 매칭 안된 건 w=1로 사용
        REDDB_DF[w_col] = REDDB_DF[w_col].fillna(1.0)

    # float32로 정리
    REDDB_DF[w_col] = pd.to_numeric(REDDB_DF[w_col], errors="coerce").astype("float32")
    # (use_col은 bool로 확정)
    REDDB_DF[use_col] = REDDB_DF[use_col].astype(bool)

    # 에너지 컬럼 체크
    if cfg.REDDB_ENERGY_COL not in REDDB_DF.columns:
        raise ValueError(f"RedDB CSV에 에너지 컬럼('{cfg.REDDB_ENERGY_COL}')이 없음. cols={list(REDDB_DF.columns)[:60]}")

    REDDB_DF[cfg.REDDB_ENERGY_COL] = pd.to_numeric(REDDB_DF[cfg.REDDB_ENERGY_COL], errors="coerce")
    REDDB_DF = REDDB_DF.dropna(subset=[cfg.REDDB_ENERGY_COL]).copy()

    # 통계 출력
    if len(REDDB_DF):
        print(
            "RedDB DF:", REDDB_DF.shape,
            f"{w_col}>1 ratio:", float((REDDB_DF[w_col] > 1).mean()),
            f"{w_col} min/max:", float(REDDB_DF[w_col].min()), float(REDDB_DF[w_col].max())
        )
        print(
            f"{use_col} True ratio:", float(REDDB_DF[use_col].mean())
        )
    else:
        print("RedDB DF: (0, ?) after filtering")

    return REDDB_DF, IK_COL

# ============================================================
# (3) Split RedDB (smiles + w) AFTER weights attached
# ============================================================
def split_train_valid_with_w(df: pd.DataFrame, valid_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_valid = max(1, int(len(idx) * valid_ratio))
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    return df_train, df_valid

def select_fraction_by_energy(cfg: CFG, df: pd.DataFrame) -> pd.DataFrame:
    if not getattr(cfg, "REDDB_USE_FRAC_FILTER", False):
        return df

    df2 = df.sort_values(cfg.REDDB_ENERGY_COL, ascending=True)
    k = max(1, int(len(df2) * float(cfg.REDDB_FRAC)))
    if getattr(cfg, "REDDB_SELECT_LOWEST", True):
        return df2.head(k).copy()
    return df2.sample(k, random_state=cfg.SEED).copy()

def boost_weights_by_energy_fraction(cfg: CFG, df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ 데이터는 자르지 않고(keep all),
    에너지 기준 상위(=더 desirable) fraction에 대해서만 w를 추가 boost.
    - REDDB_SELECT_LOWEST=True: 에너지 낮을수록(desirable) → 하위 frac를 boost
    - REDDB_FRAC_BOOST=1.0 이면 아무 것도 안 함
    """
    frac = float(getattr(cfg, "REDDB_FRAC", 1.0))
    boost = float(getattr(cfg, "REDDB_FRAC_BOOST", 1.0))

    if frac >= 1.0 or boost <= 1.0:
        print(f"[RedDB frac-boost] skip (frac={frac}, boost={boost})")
        return df

    E = cfg.REDDB_ENERGY_COL
    w_col = cfg.W_COL
    use_col = cfg.W_USE

    if E not in df.columns:
        raise KeyError(f"[RedDB frac-boost] energy col '{E}' missing. cols={list(df.columns)[:60]}")
    if w_col not in df.columns:
        raise KeyError(f"[RedDB frac-boost] weight col '{w_col}' missing. cols={list(df.columns)[:60]}")

    df = df.copy()
    df[E] = pd.to_numeric(df[E], errors="coerce")
    df = df.dropna(subset=[E]).copy()

    # threshold by quantile
    if getattr(cfg, "REDDB_SELECT_LOWEST", True):
        thr = float(df[E].quantile(frac))
        mask = df[E] <= thr
        mode = "lowest"
    else:
        thr = float(df[E].quantile(1.0 - frac))
        mask = df[E] >= thr
        mode = "highest"

    before_mean = float(df[w_col].mean())
    before_hi_mean = float(df.loc[mask, w_col].mean()) if mask.any() else float("nan")

    df.loc[mask, w_col] = (df.loc[mask, w_col].astype(float) * boost).astype("float32")

    after_mean = float(df[w_col].mean())
    after_hi_mean = float(df.loc[mask, w_col].mean()) if mask.any() else float("nan")

    # (선택) use_for_ft는 "softft 매칭 여부"로 이미 정의돼 있음.
    # boost를 했다고 해서 use_for_ft True로 만들지는 않는 게 원칙적으로 맞음.
    # 다만 원하면 아래처럼 "boost 대상은 use_for_ft=True로 간주"도 가능.
    # if use_col in df.columns:
    #     df.loc[mask, use_col] = True

    print(
        f"[RedDB frac-boost] mode={mode} frac={frac} thr={thr:.6g} boost={boost} "
        f"boosted={int(mask.sum())}/{len(df)} "
        f"w_mean {before_mean:.3f}->{after_mean:.3f} "
        f"w_mean(boosted) {before_hi_mean:.3f}->{after_hi_mean:.3f}"
    )
    return df

def read_reddb_train_valid_with_weights(cfg: CFG) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    w_map = load_softft_weight_map(cfg)
    REDDB_DF, _ = load_reddb_df_with_weights(cfg, w_map)  # ✅ USE_ONLY_SOFTFT 처리됨

    # ============================================================
    # ✅ (중요) REDDB_FRAC: "필터"가 아니라 "가중치 부스팅" 용도
    #  - cfg.REDDB_USE_FRAC_FILTER=True 로 명시한 경우에만 필터를 허용
    #  - 기본은 keep all + boost
    # ============================================================
    if getattr(cfg, "REDDB_USE_FRAC_FILTER", False):
        before = len(REDDB_DF)
        REDDB_DF = select_fraction_by_energy(cfg, REDDB_DF)
        print(f"[RedDB frac-filter] {before} -> {len(REDDB_DF)} (frac={cfg.REDDB_FRAC}, lowest={cfg.REDDB_SELECT_LOWEST})")
    else:
        # keep all + boost
        REDDB_DF = boost_weights_by_energy_fraction(cfg, REDDB_DF)

    reddb_df_train, reddb_df_valid = split_train_valid_with_w(REDDB_DF, cfg.VALID_RATIO_REDDB, cfg.SEED + 2)

    SMI_COL = cfg.REDDB_SMILES_COL
    if SMI_COL not in reddb_df_train.columns:
        raise KeyError(f"Missing smiles column '{SMI_COL}' in reddb_df_train. cols={list(reddb_df_train.columns)[:60]}")

    if cfg.W_COL not in reddb_df_train.columns:
        raise KeyError(f"Missing weight column '{cfg.W_COL}' in reddb_df_train. cols={list(reddb_df_train.columns)[:60]}")

    reddb_train_base = reddb_df_train[SMI_COL].astype(str).tolist()

    reddb_train = reddb_df_train[SMI_COL].astype(str).tolist()
    reddb_valid = reddb_df_valid[SMI_COL].astype(str).tolist()
    reddb_train_w = reddb_df_train[cfg.W_COL].astype(np.float32).values
    reddb_valid_w = reddb_df_valid[cfg.W_COL].astype(np.float32).values

    print("RedDB total/train/valid:", len(REDDB_DF), len(reddb_train), len(reddb_valid))
    print("RedDB w>1(train):", float((np.array(reddb_train_w) > 1).mean()),
          "w range:", (float(np.min(reddb_train_w)), float(np.max(reddb_train_w))))

    reddb_train = [encode_special(s, cfg) for s in reddb_train]
    reddb_valid = [encode_special(s, cfg) for s in reddb_valid]
    reddb_train_base = [encode_special(s, cfg) for s in reddb_train_base]

    check_token_stats(reddb_train, cfg.TOK_SO3H, "RedDB train SO3H")
    check_token_stats(reddb_train, cfg.TOK_COOH, "RedDB train COOH")

    return reddb_train, reddb_valid, reddb_train_w, reddb_valid_w, reddb_train_base, REDDB_DF
# ============================================================
# (4) (Optional) Augmentation (keep weights aligned)
# ============================================================
def randomize_smiles(smi: str) -> Optional[str]:
    if not RDKit_OK:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, doRandom=True)
    except Exception:
        return None

def augment_train_smiles_with_w(smiles_list, w_list, cfg: CFG, n_aug_per_mol=1, aug_prob=0.5, seed=0):
    if (not RDKit_OK) or n_aug_per_mol <= 0 or aug_prob <= 0:
        return list(smiles_list), list(w_list)

    rng = np.random.default_rng(seed)
    out_smi = list(smiles_list)
    out_w   = list(w_list)

    for smi, wi in zip(smiles_list, w_list):
        if rng.random() > aug_prob:
            continue
        for _ in range(n_aug_per_mol):
            rs = randomize_smiles(smi)
            if rs is not None and (cfg.MIN_LEN <= len(rs) <= cfg.MAX_LEN - 2):
                out_smi.append(rs)
                out_w.append(float(wi))
    return out_smi, out_w

def augment_train_smiles(train_list: List[str], cfg: CFG, n_aug_per_mol: int = 1, aug_prob: float = 0.5, seed: int = 0) -> List[str]:
    rng = np.random.default_rng(seed)
    out = list(train_list)
    if (not RDKit_OK) or n_aug_per_mol <= 0 or aug_prob <= 0:
        return basic_len_filter(out, cfg.MIN_LEN, cfg.MAX_LEN)

    for smi in train_list:
        if rng.random() > aug_prob:
            continue
        for _ in range(n_aug_per_mol):
            rs = randomize_smiles(smi)
            if rs is not None:
                out.append(rs)
    return basic_len_filter(out, cfg.MIN_LEN, cfg.MAX_LEN)