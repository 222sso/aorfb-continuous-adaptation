# ablation/predict_generated.py
# ============================================
# After-generation scoring + predictor inference
#   - Input: cfg.OUTDIR/*_gen_*.txt (plain SMILES, one per line)
#   - Compute: validity/unique/novel per file
#   - Predict: ensemble mean±std using pred_outputs_multitask_scaffoldsplit_repeat (seed42/43/44)
#   - Save: summary_metrics.csv + samples_scored_all.csv
# ============================================

import os, re, json, glob, time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# quiet TF logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ----------------------------
# RDKit helpers
# ----------------------------
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    RDKit_OK = True
except Exception:
    RDKit_OK = False


def canon_smiles(s: str) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    if len(s) == 0:
        return None
    if not RDKit_OK:
        return s  # fallback
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)


def load_ref_set(path: str, smiles_col: Optional[str] = None) -> set:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Novelty reference not found: {path}")

    if path.lower().endswith(".smi"):
        lines = [l.strip() for l in open(path, "r", encoding="utf-8") if l.strip()]
        can = [canon_smiles(x) for x in lines]
        return set([c for c in can if c])

    df = pd.read_csv(path)
    if smiles_col is None:
        for c in ["smiles", "SMILES", "canonical_smiles", "smiles_main", "canon"]:
            if c in df.columns:
                smiles_col = c
                break
        if smiles_col is None:
            smiles_col = df.columns[0]
    can = [canon_smiles(x) for x in df[smiles_col].astype(str).tolist()]
    return set([c for c in can if c])


def validity_unique_novel(smiles_list: List[str], ref_set: set) -> Tuple[Dict[str, float], List[Optional[str]]]:
    can = [canon_smiles(s) for s in smiles_list]
    can_valid = [c for c in can if c]
    uniq = list(dict.fromkeys(can_valid))
    novel = [c for c in uniq if c not in ref_set]
    metrics = {
        "n_total": len(smiles_list),
        "n_valid": len(can_valid),
        "n_unique": len(set(can_valid)),
        "n_novel": len(novel),
        "valid_rate": (len(can_valid) / max(1, len(smiles_list))),
        "unique_rate": (len(set(can_valid)) / max(1, len(can_valid))),
        "novel_rate": (len(novel) / max(1, len(set(can_valid)))),
    }
    return metrics, can


# ----------------------------
# Predictor ensemble (seed42/43/44)
# ----------------------------
import tensorflow as tf
from tensorflow.keras import layers, Model


SO3H_EQUIV = re.compile(
    r"O=S\(=O\)\(O\)|"
    r"O=S\(O\)\(=O\)|"
    r"S\(=O\)\(O\)=O|"
    r"S\(O\)\(=O\)=O"
)
SO3H_CANON = re.compile(r"S\(=O\)\(=O\)O")
COOH_EQUIV = re.compile(r"O=C\(O\)")
COOH_CANON = re.compile(r"C\(=O\)O")

TOK_SO3H = "¤"
TOK_COOH = "§"


def encode_special_pred(s: str) -> str:
    s = SO3H_EQUIV.sub("S(=O)(=O)O", str(s))
    s = SO3H_CANON.sub(TOK_SO3H, s)
    s = COOH_EQUIV.sub("C(=O)O", s)
    s = COOH_CANON.sub(TOK_COOH, s)
    return s


@dataclass
class SeedArtifacts:
    seed: int
    weights_path: str
    vocab: List[str]
    stoi: Dict[str, int]
    PAD_ID: int
    UNK_ID: int
    MAX_LEN: int
    solv_mean: float
    solv_scale: float
    lumo_mean: float
    lumo_scale: float
    architecture: str
    head_units: int


def load_seed_artifacts(pred_dir: str, seed: int) -> SeedArtifacts:
    weights_path = os.path.join(pred_dir, f"seed{seed}_best_model.weights.h5")
    vocab_path   = os.path.join(pred_dir, f"seed{seed}_vocab.json")
    scalers_path = os.path.join(pred_dir, f"seed{seed}_scalers.json")
    for p in [weights_path, vocab_path, scalers_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing artifact: {p}")

    vj = json.load(open(vocab_path, "r", encoding="utf-8"))
    vocab = vj["vocab"]
    PAD_TOKEN = vj.get("PAD_TOKEN", "<PAD>")
    UNK_TOKEN = vj.get("UNK_TOKEN", "<UNK>")
    MAX_LEN = int(vj["MAX_LEN"])
    architecture = str(vj.get("architecture", "baseline"))
    head_units = int(vj.get("head_units", 128))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    if PAD_TOKEN not in stoi or UNK_TOKEN not in stoi:
        raise ValueError(f"PAD/UNK token not in vocab for seed {seed}. PAD={PAD_TOKEN} UNK={UNK_TOKEN}")
    PAD_ID = stoi[PAD_TOKEN]
    UNK_ID = stoi[UNK_TOKEN]

    sj = json.load(open(scalers_path, "r", encoding="utf-8"))
    return SeedArtifacts(
        seed=seed,
        weights_path=weights_path,
        vocab=vocab,
        stoi=stoi,
        PAD_ID=PAD_ID,
        UNK_ID=UNK_ID,
        MAX_LEN=MAX_LEN,
        solv_mean=float(sj["solv_mean"]),
        solv_scale=float(sj["solv_scale"]),
        lumo_mean=float(sj["lumo_mean"]),
        lumo_scale=float(sj["lumo_scale"]),
        architecture=architecture,
        head_units=head_units,
    )


def build_pred_model(vocab_size: int, max_len: int, dropout: float = 0.2,
                     architecture: str = "baseline", head_units: int = 128) -> Model:
    inp = layers.Input(shape=(max_len,), dtype="int32", name="inp")
    x = layers.Embedding(vocab_size, 128, mask_zero=True)(inp)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=False))(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    if architecture == "head_dense":
        solv_h = layers.Dense(head_units, activation="relu", name="solv_head_dense_1")(x)
        solv_h = layers.Dropout(dropout * 0.5, name="solv_head_dropout_1")(solv_h)
        lumo_h = layers.Dense(head_units, activation="relu", name="lumo_head_dense_1")(x)
        lumo_h = layers.Dropout(dropout * 0.5, name="lumo_head_dropout_1")(lumo_h)
        out_solv = layers.Dense(1, name="solv")(solv_h)
        out_lumo = layers.Dense(1, name="lumo")(lumo_h)
    else:
        out_solv = layers.Dense(1, name="solv")(x)
        out_lumo = layers.Dense(1, name="lumo")(x)

    return Model(inp, outputs={"solv": out_solv, "lumo": out_lumo})


def encode_one(tok: str, stoi: Dict[str, int], max_len: int, PAD_ID: int, UNK_ID: int):
    if len(tok) > max_len:
        return None
    ids = [stoi.get(ch, UNK_ID) for ch in tok]
    ids = ids + [PAD_ID] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)


_MODEL_CACHE = {}
def get_model(seed_art: SeedArtifacts, dropout=0.2):
    key = (
        seed_art.seed,
        seed_art.MAX_LEN,
        len(seed_art.vocab),
        dropout,
        seed_art.weights_path,
        seed_art.architecture,
        seed_art.head_units,
    )
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    m = build_pred_model(
        vocab_size=len(seed_art.vocab),
        max_len=seed_art.MAX_LEN,
        dropout=dropout,
        architecture=seed_art.architecture,
        head_units=seed_art.head_units,
    )
    m.load_weights(seed_art.weights_path)
    m.trainable = False
    _MODEL_CACHE[key] = m
    return m


def predict_seed(seed_art: SeedArtifacts, canon_list: List[str], batch_size=256) -> pd.DataFrame:
    t0 = time.time()
    print(f"[Predict][seed{seed_art.seed}] encode start: n={len(canon_list):,}", flush=True)
    tok_list = [encode_special_pred(s) for s in canon_list]

    X = []
    keep = []
    for i, t in enumerate(tok_list):
        arr = encode_one(t, seed_art.stoi, seed_art.MAX_LEN, seed_art.PAD_ID, seed_art.UNK_ID)
        if arr is None:
            continue
        X.append(arr); keep.append(i)

    if len(X) == 0:
        raise RuntimeError("No encodable rows (length/vocab).")

    X = np.stack(X, axis=0)
    print(f"[Predict][seed{seed_art.seed}] encode done: encodable={len(X):,}", flush=True)
    m = get_model(seed_art)
    print(f"[Predict][seed{seed_art.seed}] model loaded, predict start (batch_size={batch_size})", flush=True)
    pred = m.predict(X, batch_size=batch_size, verbose=0)
    print(f"[Predict][seed{seed_art.seed}] predict done in {(time.time() - t0):.1f}s", flush=True)

    solv_scaled = pred["solv"].reshape(-1)
    lumo_scaled = pred["lumo"].reshape(-1)

    solv = solv_scaled * seed_art.solv_scale + seed_art.solv_mean
    lumo = lumo_scaled * seed_art.lumo_scale + seed_art.lumo_mean

    return pd.DataFrame({
        "row_idx": keep,
        f"solv_pred_eV_seed{seed_art.seed}": solv,
        f"lumo_pred_eV_seed{seed_art.seed}": lumo,
    })


# ----------------------------
# Main scoring pipeline
# ----------------------------
def _read_smiles_txt(path: str, max_n: Optional[int] = None) -> List[str]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            lines.append(s)
            if max_n is not None and len(lines) >= max_n:
                break
    return lines


def _parse_case_from_filename(path: str) -> Dict[str, object]:
    """
    Parse generation case metadata from file name.
    Expected examples:
      - A_TCN_gen_n1000_T0p70_p0p75.txt
      - B_LSTM_gen_n500_T1p20_p0p95.txt
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    model = stem.split("_gen_")[0] if "_gen_" in stem else stem

    gen_n = np.nan
    temperature = np.nan
    top_p = np.nan

    m = re.search(r"_gen_n(\d+)_T([0-9p\-]+)_p([0-9p\-]+)$", stem)
    if m:
        gen_n = int(m.group(1))
        temperature = float(m.group(2).replace("p", "."))
        top_p = float(m.group(3).replace("p", "."))

    return {
        "case_id": stem,
        "model": model,
        "gen_n": gen_n,
        "temperature": temperature,
        "top_p": top_p,
    }


def score_and_predict_generated(
    gen_files: List[str],
    novelty_ref_csv: str,
    pred_dir: str,
    seeds: List[int],
    outdir: str,
    n_per_file: Optional[int] = None,
    smiles_col_out: str = "smiles_in",
    batch_size: int = 256,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t_all = time.time()
    os.makedirs(outdir, exist_ok=True)

    ref_set = load_ref_set(novelty_ref_csv)
    print("[Predict] novelty ref set:", len(ref_set), flush=True)

    # 1) Build samples table per gen file
    rng = np.random.default_rng(123)
    summary_rows = []
    sample_frames = []

    for fp in gen_files:
        case_info = _parse_case_from_filename(fp)
        m = str(case_info["model"])

        smi_all = _read_smiles_txt(fp, max_n=None)
        if len(smi_all) == 0:
            print(f"[Predict] skip empty file: {fp}")
            continue

        # sample
        if n_per_file is None or len(smi_all) <= n_per_file:
            take = smi_all
            if n_per_file is not None and len(smi_all) < n_per_file:
                print(f"[Predict] {m}: only {len(take)} rows available (requested {n_per_file})")
        else:
            idx = rng.choice(len(smi_all), size=n_per_file, replace=False)
            take = [smi_all[i] for i in idx]

        metrics, can_list = validity_unique_novel(take, ref_set)
        metrics.update({
            "case_id": case_info["case_id"],
            "model": m,
            "gen_n": case_info["gen_n"],
            "temperature": case_info["temperature"],
            "top_p": case_info["top_p"],
            "file": fp,
        })
        summary_rows.append(metrics)

        sample_frames.append(pd.DataFrame({
            "case_id": case_info["case_id"],
            "model": m,
            "gen_n": case_info["gen_n"],
            "temperature": case_info["temperature"],
            "top_p": case_info["top_p"],
            "file": fp,
            smiles_col_out: take,
            "canon": can_list,
            "rdkit_valid": [1 if c else 0 for c in can_list],
        }))

    if len(sample_frames) == 0:
        raise RuntimeError("No samples to score. Check gen_files paths.")

    summary_df = pd.DataFrame(summary_rows)[
        [
            "case_id", "model", "gen_n", "temperature", "top_p",
            "n_total", "n_valid", "n_unique", "n_novel",
            "valid_rate", "unique_rate", "novel_rate", "file"
        ]
    ].copy()
    samples_df = pd.concat(sample_frames, ignore_index=True)
    print("[Predict] samples_df:", samples_df.shape, flush=True)

    # 2) Predictor ensemble on valid rows only
    valid_mask = samples_df["rdkit_valid"].values.astype(bool)
    canon_valid = samples_df.loc[valid_mask, "canon"].astype(str).tolist()
    print(f"[Predict] valid rows for predictor: {len(canon_valid):,}", flush=True)

    # Save stage-1 outputs before heavy predictor inference.
    stage1_summary = os.path.join(outdir, "summary_metrics_stage1_pre_pred.csv")
    stage1_samples = os.path.join(outdir, "samples_stage1_pre_pred.csv")
    summary_df.to_csv(stage1_summary, index=False, encoding="utf-8")
    samples_df.to_csv(stage1_samples, index=False, encoding="utf-8")
    print(f"[Predict] Stage1 saved: {stage1_summary}", flush=True)
    print(f"[Predict] Stage1 saved: {stage1_samples}", flush=True)

    seed_preds = []
    for seed in seeds:
        t_seed = time.time()
        print(f"[Predict] seed {seed} start", flush=True)
        art = load_seed_artifacts(pred_dir, seed)
        dfp = predict_seed(art, canon_valid, batch_size=batch_size)
        seed_preds.append(dfp)
        seed_out = os.path.join(outdir, f"pred_seed{seed}.csv")
        dfp.to_csv(seed_out, index=False, encoding="utf-8")
        print(f"[Predict] seed {seed} saved: {seed_out} ({time.time() - t_seed:.1f}s)", flush=True)

    pred_merged = seed_preds[0]
    for dfp in seed_preds[1:]:
        pred_merged = pred_merged.merge(dfp, on="row_idx", how="inner")

    valid_df = samples_df.loc[valid_mask].reset_index(drop=True).copy()
    valid_df = valid_df.reset_index().rename(columns={"index":"row_idx"}).merge(pred_merged, on="row_idx", how="left")

    solv_cols = [c for c in valid_df.columns if c.startswith("solv_pred_eV_seed")]
    lumo_cols = [c for c in valid_df.columns if c.startswith("lumo_pred_eV_seed")]

    valid_df["solv_mean_eV"] = valid_df[solv_cols].mean(axis=1)
    valid_df["solv_std_eV"]  = valid_df[solv_cols].std(axis=1, ddof=0)
    valid_df["lumo_mean_eV"] = valid_df[lumo_cols].mean(axis=1)
    valid_df["lumo_std_eV"]  = valid_df[lumo_cols].std(axis=1, ddof=0)

    samples_out = samples_df.copy()
    for c in ["solv_mean_eV","solv_std_eV","lumo_mean_eV","lumo_std_eV"]:
        samples_out[c] = np.nan
    samples_out.loc[valid_mask, ["solv_mean_eV","solv_std_eV","lumo_mean_eV","lumo_std_eV"]] = \
        valid_df[["solv_mean_eV","solv_std_eV","lumo_mean_eV","lumo_std_eV"]].values

    # 3) property summary per model
    prop_rows = []
    for case_id in summary_df["case_id"].astype(str).unique().tolist():
        sub = samples_out[(samples_out["case_id"].astype(str) == case_id) & (samples_out["rdkit_valid"] == 1)].copy()
        model_name = str(sub["model"].iloc[0]) if len(sub) else str(summary_df.loc[summary_df["case_id"] == case_id, "model"].iloc[0])
        if len(sub) == 0:
            prop_rows.append({
                "case_id": case_id,
                "model": model_name,
                "n_pred": 0,
                "ΔG_solv_mean_eV": np.nan,
                "ΔG_solv_std_eV":  np.nan,
                "LUMO_mean_eV":    np.nan,
                "LUMO_std_eV":     np.nan,
                "ensemble_uncert_solv_mean": np.nan,
                "ensemble_uncert_lumo_mean": np.nan,
            })
            continue
        prop_rows.append({
            "case_id": case_id,
            "model": model_name,
            "n_pred": len(sub),
            "ΔG_solv_mean_eV": float(sub["solv_mean_eV"].mean()),
            "ΔG_solv_std_eV":  float(sub["solv_mean_eV"].std(ddof=0)),
            "LUMO_mean_eV":    float(sub["lumo_mean_eV"].mean()),
            "LUMO_std_eV":     float(sub["lumo_mean_eV"].std(ddof=0)),
            "ensemble_uncert_solv_mean": float(sub["solv_std_eV"].mean()),
            "ensemble_uncert_lumo_mean": float(sub["lumo_std_eV"].mean()),
        })

    prop_df = pd.DataFrame(prop_rows)
    final_summary = summary_df.merge(
        prop_df.drop(columns=["model"], errors="ignore"),
        on="case_id",
        how="left",
    )

    # 4) save
    out_summary = os.path.join(outdir, "summary_metrics.csv")
    out_all = os.path.join(outdir, "samples_scored_all.csv")
    final_summary.to_csv(out_summary, index=False, encoding="utf-8")
    samples_out.to_csv(out_all, index=False, encoding="utf-8")

    print("[Predict] Saved summary:", out_summary, flush=True)
    print("[Predict] Saved all samples:", out_all, flush=True)
    print(f"[Predict] total elapsed: {time.time() - t_all:.1f}s", flush=True)

    return final_summary, samples_out


def run_from_cfg(cfg):
    # run_ablation_abc.py에서 바로 호출하기 쉽게 cfg 기반 wrapper
    # gen files: cfg.OUTDIR/*_gen_*.txt
    gen_files = sorted(glob.glob(os.path.join(cfg.OUTDIR, "*_gen_*.txt")))
    if len(gen_files) == 0:
        raise FileNotFoundError(f"No gen txt files found in {cfg.OUTDIR} (pattern '*_gen_*.txt').")

    pred_dir = getattr(cfg, "PRED_DIR", "./pred_outputs_multitask_scaffoldsplit_repeat")
    seeds = getattr(cfg, "PRED_SEEDS", [42, 43, 44])
    novelty_ref = getattr(cfg, "NOVEL_REF_PATH", "dist_reports/merged_all_inchikey_v2.csv")

    outdir = os.path.join(cfg.OUTDIR, "pred_eval")

    # n_per_file: cfg.GEN_N과 동일하게(이미 그 개수만 생성했으면 None이어도 됨)
    n_per_file = getattr(cfg, "PRED_N_PER_FILE", None)
    batch_size = int(getattr(cfg, "PRED_BATCH_SIZE", 256))
    if n_per_file is not None:
        n_per_file = int(n_per_file)
    print(f"[Predict] config: seeds={list(seeds)}, n_per_file={n_per_file}, batch_size={batch_size}", flush=True)

    return score_and_predict_generated(
        gen_files=gen_files,
        novelty_ref_csv=novelty_ref,
        pred_dir=pred_dir,
        seeds=seeds,
        outdir=outdir,
        n_per_file=n_per_file,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    # CLI 실행도 가능하게
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gen-glob", default="", help="ex) outputs/*_gen_*.txt  (비우면 현재 디렉토리에서 탐색)")
    p.add_argument("--novel-ref", default="dist_reports/merged_all_inchikey_v2.csv")
    p.add_argument("--pred-dir", default="./pred_outputs_multitask_scaffoldsplit_repeat")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--outdir", default="./pred_eval")
    p.add_argument("--n-per-file", default="", help="ex) 500 (비우면 전체 사용)")
    args = p.parse_args()

    gen_glob = args.gen_glob.strip() or "*_gen_*.txt"
    gen_files = sorted(glob.glob(gen_glob))
    if len(gen_files) == 0:
        raise FileNotFoundError(f"No files for gen_glob: {gen_glob}")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    n_per = None
    if args.n_per_file.strip():
        n_per = int(args.n_per_file.strip())

    score_and_predict_generated(
        gen_files=gen_files,
        novelty_ref_csv=args.novel_ref,
        pred_dir=args.pred_dir,
        seeds=seeds,
        outdir=args.outdir,
        n_per_file=n_per,
        batch_size=256,
    )
