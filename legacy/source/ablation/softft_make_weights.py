# ablation/softft_make_weights.py
import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless에서도 저장 가능
import matplotlib.pyplot as plt


HARTREE_TO_EV = 27.211386245988


def missing_report(s: pd.Series):
    n = len(s)
    nn = int(s.isna().sum())
    return n, nn, nn / n * 100.0


def summarize(series: pd.Series, name: str, pcts=None) -> pd.DataFrame:
    if pcts is None:
        pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    s = series.dropna().astype(float)
    if len(s) == 0:
        return pd.DataFrame([{"col": name, "note": "ALL NaN"}])
    row = {
        "col": name,
        "count": len(s),
        "mean": s.mean(),
        "std": s.std(ddof=1) if len(s) > 1 else 0.0,
        "min": s.min(),
        "max": s.max(),
    }
    for p in pcts:
        row[f"p{p:02d}"] = float(np.percentile(s, p))
    return pd.DataFrame([row])


def plot_hist(series: pd.Series, title: str, path: str, bins: int = 60, xlabel: str = "value"):
    s = series.dropna().astype(float)
    if len(s) == 0:
        return
    plt.figure()
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_ecdf(series: pd.Series, title: str, path: str, xlabel: str = "value"):
    s = series.dropna().astype(float).values
    if len(s) == 0:
        return
    s = np.sort(s)
    y = np.arange(1, len(s) + 1) / len(s)
    plt.figure()
    plt.plot(s, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("ECDF")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def percentile_score(values) -> pd.Series:
    """0~1 분위수 점수. NaN은 NaN 유지"""
    s = pd.Series(values)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    m = s.notna()
    if m.sum() == 0:
        return out
    out.loc[m] = s.loc[m].rank(pct=True, method="average").astype(float)
    return out


def z_to_w(z, t: float, a: float, wmax: float):
    if pd.isna(z):
        return np.nan
    if z < t:
        w = 1.0
    else:
        w = 1.0 + a * (z - t) / (1.0 - t)
    return float(min(w, wmax))


def make_softft_weights(
    in_csv: str,
    outdir: str,
    key_col: str = "inchiKey",
    smiles_col: str = "smiles_main",
    solv_col: str = "Solvation_energy",
    orb_mode: str = "lumo",  # "lumo" | "homo"
    lumo_col: str = "single_point_lumo_solution",
    homo_col: str = "single_point_homo_solution",
    input_unit: str = "hartree",  # "hartree" | "ev"
    beta_solv: float = 1.0,
    beta_orb: float = 0.2,
    T: float = 0.90,
    A: float = 9.0,
    w_max: float = 10.0,
    hist_bins: int = 60,
    orb_lower_is_better: bool | None = None,
    topk: int = 200,
    random_state: int = 0,
):
    os.makedirs(outdir, exist_ok=True)

    orb_col_map = {"lumo": lumo_col, "homo": homo_col}
    if orb_mode not in orb_col_map:
        raise ValueError(f"orb_mode must be one of {list(orb_col_map.keys())}")

    orb_col = orb_col_map[orb_mode]

    # 기본 방향: lumo/homo 모두 "낮을수록 좋다"로 둠 (원하면 CLI에서 변경)
    if orb_lower_is_better is None:
        orb_lower_is_better = True

    df = pd.read_csv(in_csv)
    print("Loaded:", df.shape)

    need_cols = [key_col, smiles_col, solv_col, orb_col]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available cols(head): {list(df.columns)[:60]}"
        )

    # key standardize
    df[key_col] = df[key_col].astype(str).str.strip().str.upper()
    df[smiles_col] = df[smiles_col].astype(str)

    # numeric
    df[solv_col] = pd.to_numeric(df[solv_col], errors="coerce")
    df[orb_col] = pd.to_numeric(df[orb_col], errors="coerce")

    # unit convert -> eV (해석/일관성)
    iu = input_unit.lower()
    if iu == "hartree":
        df[solv_col] = df[solv_col] * HARTREE_TO_EV
        df[orb_col] = df[orb_col] * HARTREE_TO_EV
    elif iu == "ev":
        pass
    else:
        raise ValueError("input_unit must be 'hartree' or 'ev'")

    # Missingness
    print("\n[Missingness]")
    for c in [solv_col, orb_col]:
        n, nn, rr = missing_report(df[c])
        print(f"- {c}: NaN {nn}/{n} ({rr:.2f}%)")

    # Summary stats
    summary = pd.concat(
        [
            summarize(df[solv_col], solv_col),
            summarize(df[orb_col], orb_col),
        ],
        ignore_index=True,
    )
    summary_path = os.path.join(outdir, "summary_stats.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)

    # Plots (raw)
    plot_hist(df[solv_col], f"{solv_col} histogram", os.path.join(outdir, "hist_solv.png"),
              bins=hist_bins, xlabel="value (eV)")
    plot_ecdf(df[solv_col], f"{solv_col} ECDF", os.path.join(outdir, "ecdf_solv.png"),
              xlabel="value (eV)")

    plot_hist(df[orb_col], f"{orb_col} histogram", os.path.join(outdir, f"hist_{orb_mode}.png"),
              bins=hist_bins, xlabel="value (eV)")
    plot_ecdf(df[orb_col], f"{orb_col} ECDF", os.path.join(outdir, f"ecdf_{orb_mode}.png"),
              xlabel="value (eV)")

    # SoftFT scores
    # solv: 더 음수일수록 좋음 -> -solv가 클수록 좋음
    df["q_solv"] = percentile_score(-df[solv_col])

    # orb: 방향 맞춰서 "클수록 좋게" 변환 후 percentile
    if orb_lower_is_better:
        df["q_orb"] = percentile_score(-df[orb_col])
    else:
        df["q_orb"] = percentile_score(df[orb_col])

    # 결합 점수 z
    df["z"] = (beta_solv * df["q_solv"] + beta_orb * df["q_orb"]) / (beta_solv + beta_orb)

    df["w"] = df["z"].apply(lambda zz: z_to_w(zz, t=T, a=A, wmax=w_max))
    df["use_for_ft"] = df["w"].notna()

    # Save outputs
    out_cols = [key_col, smiles_col, solv_col, orb_col, "q_solv", "q_orb", "z", "w", "use_for_ft"]
    out_path = os.path.join(outdir, "softft_scores_weights.csv")
    df[out_cols].to_csv(out_path, index=False)
    print("Saved:", out_path)

    top_path = os.path.join(outdir, "top_weight_samples.csv")
    df[df["use_for_ft"]].sort_values("w", ascending=False).head(topk)[out_cols].to_csv(top_path, index=False)
    print("Saved:", top_path)

    # Plots (z/w)
    plot_hist(df["z"], "Combined score z histogram", os.path.join(outdir, "hist_z.png"),
              bins=hist_bins, xlabel="z")
    plot_ecdf(df["z"], "Combined score z ECDF", os.path.join(outdir, "ecdf_z.png"),
              xlabel="z")
    plot_hist(df["w"], "Sample weight w histogram", os.path.join(outdir, "hist_w.png"),
              bins=hist_bins, xlabel="w")
    plot_ecdf(df["w"], "Sample weight w ECDF", os.path.join(outdir, "ecdf_w.png"),
              xlabel="w")

    # Quick checks
    print("\nDone.")
    print("z percentiles:", df["z"].quantile([0.5, 0.7, 0.8, 0.9, 0.95, 0.99]))
    valid = df["z"].notna()

    wgt_gt1_ratio = float((df.loc[valid, "w"] > 1.0).mean()) if valid.any() else float("nan")
    print("w>1 ratio:", wgt_gt1_ratio)
    print("w mean:", float(df.loc[valid, "w"].mean()) if valid.any() else float("nan"),
          "w max:", float(df.loc[valid, "w"].max()) if valid.any() else float("nan"))

    # 1) q_solv는 solv와 단조 반비례(스피어만 ~ -1)
    sp1 = df[[solv_col, "q_solv"]].dropna().corr(method="spearman").iloc[0, 1]
    print("spearman(solv, q_solv) =", sp1)

    # 2) q_orb는 orb_col과 단조 반비례/비례 (설정에 따라)
    sp2 = df[[orb_col, "q_orb"]].dropna().corr(method="spearman").iloc[0, 1]
    print(f"spearman({orb_mode}, q_orb) =", sp2)

    # 3) w>1 비율은 z>=T 비율과 같아야 함(유효 샘플 기준)
    if valid.any():
        print("ratio(z>=T) =", float((df.loc[valid, "z"] >= T).mean()))
        print("ratio(w>1)  =", float((df.loc[valid, "w"] > 1.0).mean()))

        # 4) formula spot-check
        tmp = df.loc[valid, ["z", "w"]].sample(min(5, valid.sum()), random_state=random_state)
        tmp2 = tmp.assign(
            w_formula=tmp["z"].apply(lambda z: 1.0 if z < T else min(1.0 + A * (z - T) / (1.0 - T), w_max))
        )
        print("\n[Spot-check z->w]")
        print(tmp2)

        # 5) top-weight sanity check
        top = df.loc[valid].nlargest(min(50, valid.sum()), "w")
        allv = df.loc[valid]
        print(f"\nTOP mean solv, {orb_mode}:", float(top[solv_col].mean()), float(top[orb_col].mean()))
        print(f"ALL mean solv, {orb_mode}:", float(allv[solv_col].mean()), float(allv[orb_col].mean()))

    return out_path  # 가장 중요한 산출물 경로


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=".", help="프로젝트 루트(상대경로 해석 기준)")
    p.add_argument("--in-csv", default="dist_reports/merged_all_inchikey_v2.csv")
    p.add_argument("--outdir", default="dist_reports/softft_v3")

    p.add_argument("--key-col", default="inchiKey")
    p.add_argument("--smiles-col", default="smiles_main")
    p.add_argument("--solv-col", default="Solvation_energy")
    p.add_argument("--orb-mode", default="lumo", choices=["lumo", "homo"])
    p.add_argument("--lumo-col", default="single_point_lumo_solution")
    p.add_argument("--homo-col", default="single_point_homo_solution")

    p.add_argument("--input-unit", default="hartree", choices=["hartree", "ev"])
    p.add_argument("--beta-solv", type=float, default=1.0)
    p.add_argument("--beta-orb", type=float, default=0.2)

    p.add_argument("--T", type=float, default=0.90)
    p.add_argument("--A", type=float, default=9.0)
    p.add_argument("--w-max", type=float, default=10.0)

    p.add_argument("--hist-bins", type=int, default=60)
    p.add_argument("--orb-lower-is-better", type=int, default=1, help="1이면 낮을수록 좋음, 0이면 높을수록 좋음")
    p.add_argument("--topk", type=int, default=200)
    p.add_argument("--random-state", type=int, default=0)

    args = p.parse_args()

    root = os.path.abspath(args.project_root)
    in_csv = os.path.join(root, args.in_csv)
    outdir = os.path.join(root, args.outdir)

    out_path = make_softft_weights(
        in_csv=in_csv,
        outdir=outdir,
        key_col=args.key_col,
        smiles_col=args.smiles_col,
        solv_col=args.solv_col,
        orb_mode=args.orb_mode,
        lumo_col=args.lumo_col,
        homo_col=args.homo_col,
        input_unit=args.input_unit,
        beta_solv=args.beta_solv,
        beta_orb=args.beta_orb,
        T=args.T,
        A=args.A,
        w_max=args.w_max,
        hist_bins=args.hist_bins,
        orb_lower_is_better=bool(args.orb_lower_is_better),
        topk=args.topk,
        random_state=args.random_state,
    )

    # 실행 코드가 받아쓰기 쉽게 "마지막 줄에 결과 경로만" 출력
    print("\nRESULT_SOFTFT_WEIGHTS_CSV=", out_path)


if __name__ == "__main__":
    _cli()