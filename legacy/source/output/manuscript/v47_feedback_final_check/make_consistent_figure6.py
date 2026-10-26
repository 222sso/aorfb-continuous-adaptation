#!/usr/bin/env python3
"""Redraw only the current Figure 6 from its frozen likelihood and motif records."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
from scipy import stats

RDLogger.DisableLog("rdApp.*")

import argparse
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'redrawn_figure6');args=ap.parse_args()
OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'ablation').is_dir())
ST = ROOT / "q1_comparison_experiments_260806" / "results" / "structure_transfer_260812"
NLL_MOL, NLL_SUM = ST / "heldout_molecule_likelihood.csv", ST / "heldout_likelihood_summary.csv"
TOKEN = ST / "heldout_target_token_nll.csv"
MOTIF = ST / "equal_controlled_motif_alignment.csv"

COLORS = {"down": "#3F78B5", "up": "#E69F00", "text": "#20242A",
          "muted": "#68727D", "grid": "#E3E7EB"}
SHORT = {"carboxylic_acid": "carboxylic acid", "sulfonic_acid": "sulfonic acid",
         "carbonyl": "carbonyl", "phenolic_hydroxyl": "phenolic OH",
         "aliphatic_hydroxyl": "aliphatic OH",
         "primary_or_secondary_amine": "1°/2° amine",
         "aromatic_nitrogen": "aromatic N", "nitrile": "nitrile", "halogen": "halogen"}
# 라벨 위치(pt).  원점 부근에 네 점이 몰려 있어 자동 배치로는 겹친다.  LEADER 는 지시선.
PLACE = {"carboxylic_acid": (-9, 13, "right"), "carbonyl": (-9, 11, "right"),
         "nitrile": (9, 6, "left"), "halogen": (2, -15, "center"),
         "aliphatic_hydroxyl": (0, 14, "center"),
         "phenolic_hydroxyl": (-38, 30, "right"),
         "primary_or_secondary_amine": (-30, -30, "right"),
         "sulfonic_acid": (-6, 28, "right"), "aromatic_nitrogen": (30, -30, "left")}
LEADER = {"phenolic_hydroxyl", "primary_or_secondary_amine", "sulfonic_acid",
          "aromatic_nitrogen", "carboxylic_acid", "carbonyl"}


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.edgecolor": COLORS["text"], "axes.linewidth": 0.8,
        "text.color": COLORS["text"], "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"], "ytick.color": COLORS["text"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def fragment_image(smiles: str, w: int = 210, h: int = 150) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 3
    opts.fixedFontSize = 34
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")


def main() -> None:
    set_style()
    mol = pd.read_csv(NLL_MOL)
    summ = pd.read_csv(NLL_SUM).set_index("subset")
    motif = pd.read_csv(MOTIF)
    tgt = mol[mol.is_target.astype(bool)].copy()
    assert len(mol) == 311 and len(tgt) == 8

    fig = plt.figure(figsize=(7.6, 3.3), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.50], wspace=0.60,
                          left=0.092, right=0.958, top=0.885, bottom=0.235)
    ax_slope, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    axes = [ax_slope, ax_b]

    # ---------- (a) Gardner-Altman paired estimation plot ----------
    # 세 열을 한 축 위에 둔다: Equal, Continuous, 그리고 차이 열.  차이 열의 눈금은 오른쪽
    # 축이 지고, 두 축은 "Equal 평균 = 차이 0" 으로 묶인다.  그 묶임을 수평 기준선 두 개가
    # 눈으로 보여 준다 -- 아래 선이 차이 0, 위 선이 점추정 높이다.  기준선이 없으면 오른쪽
    # 눈금이 어디에 걸린 값인지 읽을 수 없다.
    ax = axes[0]
    XEQ, XCO, XD, WID = 0.0, 1.0, 2.08, 0.52
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    tgt["delta"] = tgt.nll_per_token_soft - tgt.nll_per_token_equal
    # 부트스트랩은 CSV 원래 순서로 돌려야 원 스크립트와 같은 난수 스트림이 된다.
    # rng.choice 는 인덱스를 뽑으므로 정렬하면 다른 값을 집어 CI 가 달라진다.
    deltas = tgt.delta.to_numpy()
    tgt = tgt.sort_values("delta").reset_index(drop=True)
    n_down = int((tgt.delta < 0).sum())
    assert n_down == 7, n_down

    # --- 부트스트랩 재생성: 원 스크립트와 같은 조건(5,000 회, 시드 260812) ---
    rng = np.random.default_rng(260812)
    boot = np.array([rng.choice(deltas, size=len(deltas), replace=True).mean()
                     for _ in range(5000)])
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    t = summ.loc["target"]
    assert abs(deltas.mean() - t.paired_delta_nll_soft_minus_equal_mean) < 1e-12
    assert abs(lo - t.paired_delta_nll_bootstrap_95ci_low) < 1e-12
    assert abs(hi - t.paired_delta_nll_bootstrap_95ci_high) < 1e-12

    m_eq = float(tgt.nll_per_token_equal.mean())
    m_co = float(tgt.nll_per_token_soft.mean())
    span = np.concatenate([tgt.nll_per_token_equal.to_numpy(),
                           tgt.nll_per_token_soft.to_numpy(), m_eq + boot])
    pad = 0.09 * (span.max() - span.min())
    ax.set_xlim(-0.34, XD + WID + 0.30)
    ax.set_ylim(span.min() - pad, span.max() + pad)

    # 두 기준선: 아래 = Equal 평균(차이 0), 위 = Continuous 평균(점추정).
    for level in (m_eq, m_co):
        ax.axhline(level, color=COLORS["text"], lw=0.8, zorder=2)

    for row in tgt.itertuples():
        colour = COLORS["down"] if row.delta < 0 else COLORS["up"]
        ax.plot([XEQ, XCO], [row.nll_per_token_equal, row.nll_per_token_soft],
                color=colour, lw=1.4, alpha=0.85, zorder=4, solid_capstyle="round")
        ax.scatter([XEQ, XCO], [row.nll_per_token_equal, row.nll_per_token_soft],
                   s=24, color=colour, edgecolor="white", lw=0.5, zorder=5)

    # 차이 열: 반쪽 바이올린 + 점추정 + 95% 구간.  세로 위치는 m_eq + delta 로 왼쪽
    # 눈금에 그대로 얹으므로 오른쪽 축과 어긋날 수 없다.
    grid = np.linspace(boot.min(), boot.max(), 240)
    dens = stats.gaussian_kde(boot)(grid)
    dens = WID * dens / dens.max()
    ax.fill_betweenx(m_eq + grid, XD, XD + dens, color=COLORS["down"],
                     alpha=0.30, lw=0, zorder=3)
    ax.plot(XD + dens, m_eq + grid, color=COLORS["down"], lw=0.9, zorder=4)
    ax.plot([XD, XD], [m_eq + lo, m_eq + hi], color=COLORS["text"], lw=2.0,
            solid_capstyle="butt", zorder=5)
    ax.scatter([XD], [m_co], s=34, color=COLORS["text"], edgecolor="white",
               lw=0.6, zorder=6)

    ax.set_xticks([XEQ, XCO, XD])
    ax.set_xticklabels(["Equal\nweighting\nN = 8", "Continuous\nweighting\nN = 8",
                        "Continuous\n− Equal"])
    ax.tick_params(axis="x", length=0, pad=4, labelsize=8.2)
    # 아래 축은 끊어서 그린다.  왼쪽 두 열은 원자료(NLL), 오른쪽 한 열은 그 차이라서
    # 눈금이 다른 양이다.  선을 이어 버리면 셋이 같은 축 위의 값처럼 읽힌다.
    x0, x1 = ax.get_xlim()
    for seg in ((x0, XCO + 0.38), (XD - 0.38, x1)):
        ax.plot(seg, [0, 0], transform=ax.get_xaxis_transform(), clip_on=False,
                color=COLORS["text"], lw=0.8, zorder=7, solid_capstyle="butt")
    ax.set_ylabel("per-token NLL of held-out target molecules\n(lower = more probable)",
                  fontsize=8.6)

    # 오른쪽 축: 같은 세로 눈금을 차이로 읽는다(원점을 m_eq 만큼 옮긴 것).
    axd = ax.twinx()
    axd.set_ylim(ax.get_ylim()[0] - m_eq, ax.get_ylim()[1] - m_eq)
    for side in ("top", "left", "bottom"):
        axd.spines[side].set_visible(False)
    axd.set_ylabel("paired mean difference", fontsize=8.4, labelpad=3)
    axd.tick_params(axis="y", labelsize=8)

    handles = [plt.Line2D([], [], color=COLORS["down"], lw=1.6, marker="o",
                          markeredgecolor="white", markersize=5,
                          label=f"more probable ({n_down})"),
               plt.Line2D([], [], color=COLORS["up"], lw=1.6, marker="o",
                          markeredgecolor="white", markersize=5,
                          label=f"less probable ({len(tgt) - n_down})")]
    ax.legend(handles=handles, frameon=False, fontsize=7.4, loc="upper right",
              handletextpad=0.5, borderaxespad=0.15, labelspacing=0.28)

    # ---------- (b) motif log-odds, structures instead of names ----------
    ax = axes[1]
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color=COLORS["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    x, y = "dft_target_vs_non_target_log_odds", "soft_vs_equal_log_odds"
    inf = motif[motif.informative_for_direction == 1]
    exc = motif[motif.informative_for_direction != 1]
    r = stats.pearsonr(inf[x], inf[y])
    assert abs(r[0] - 0.944) < 0.001, r[0]

    ax.axhline(0, color=COLORS["text"], lw=0.9, zorder=2, linestyle=(0, (4, 3)))
    ax.axvline(0, color=COLORS["text"], lw=0.9, zorder=2, linestyle=(0, (4, 3)))
    # 회귀선과 신뢰 밴드.  n = 6 이므로 t 분포로 평균 응답의 95% 구간을 그린다.
    fit = np.polyfit(inf[x], inf[y], 1)
    span = np.linspace(inf[x].min() - 0.8, inf[x].max() + 0.8, 120)
    pred = np.polyval(fit, span)
    resid = inf[y] - np.polyval(fit, inf[x])
    dof = len(inf) - 2
    se = np.sqrt((resid ** 2).sum() / dof)
    xbar, sxx = inf[x].mean(), ((inf[x] - inf[x].mean()) ** 2).sum()
    half = stats.t.ppf(0.975, dof) * se * np.sqrt(1 / len(inf) + (span - xbar) ** 2 / sxx)
    ax.fill_between(span, pred - half, pred + half, color=COLORS["muted"],
                    alpha=0.18, linewidth=0, zorder=2)
    ax.plot(span, pred, color=COLORS["muted"], lw=1.1, zorder=3)
    ax.scatter(exc[x], exc[y], s=38, facecolor="white", edgecolor=COLORS["muted"],
               lw=1.0, zorder=5)
    ax.scatter(inf[x], inf[y], s=42, color=COLORS["down"], edgecolor="white",
               lw=0.6, zorder=6)
    for row in motif.itertuples():
        dx, dy, ha = PLACE[row.motif]
        ax.annotate(SHORT[row.motif], (getattr(row, x), getattr(row, y)),
                    textcoords="offset points", xytext=(dx, dy), fontsize=7.4,
                    color=COLORS["text"], ha=ha, va="center", zorder=7,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=COLORS["muted"],
                                    shrinkA=0, shrinkB=3)
                    if row.motif in LEADER else None)
    ax.set_xlim(-8.8, 6.2)
    ax.set_ylim(-2.9, 3.5)
    # 축 라벨이 계산식과 방향을 모두 말하게 한다.  "log-odds" 만 적으면 아는 사람만 읽는다.
    # 축에는 기호만 둔다.  식과 출처(Haldane-Anscombe 보정)는 캡션이 진다.
    ax.set_xlabel("ln OR, target vs remaining records (RedDB)", fontsize=8.6)
    ax.set_ylabel("ln OR,  continuous vs equal  (generated)", fontsize=8.6)
    handles = [plt.Line2D([], [], marker="o", color=COLORS["down"], linestyle="none",
                          markersize=6, markeredgecolor="white", label="resolved"),
               plt.Line2D([], [], marker="o", color="white", linestyle="none", markersize=6,
                          markeredgecolor=COLORS["muted"], label="unresolved")]
    ax.legend(handles=handles, frameon=False, fontsize=7.6, loc="lower right",
              handletextpad=0.3, borderaxespad=0.3, labelspacing=0.3)

    for letter, xf in (("a", 0.004), ("b", 0.505)):
        fig.text(xf, 0.982, letter, fontsize=12, fontweight="bold",
                 ha="left", va="bottom")

    stem = OUT / "Figure6_predictor_independent_consistent"
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(stem.with_suffix(ext), dpi=600, bbox_inches="tight", facecolor="white")

    (OUT / "Figure6_unchanged_statistics_audit.json").write_text(json.dumps({
        "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (NLL_MOL, NLL_SUM, MOTIF)},
        "panel_a": {"n": int(len(tgt)), "n_lower": n_down,
                    "mean_delta": float(t.paired_delta_nll_soft_minus_equal_mean),
                    "ci95": [float(t.paired_delta_nll_bootstrap_95ci_low),
                             float(t.paired_delta_nll_bootstrap_95ci_high)]},
        "panel_b": {"n_resolved": int(len(inf)), "pearson_r": float(r[0]),
                    "spearman_rho": float(stats.spearmanr(inf[x], inf[y])[0]),
                    "fit_slope": float(fit[0]), "fit_intercept": float(fit[1]),
                    "labels": SHORT},
    }, indent=2), encoding="utf-8")

    print(f"(a) {n_down}/8 lower; Δ = {t.paired_delta_nll_soft_minus_equal_mean:+.4f} "
          f"[{t.paired_delta_nll_bootstrap_95ci_low:+.4f}, "
          f"{t.paired_delta_nll_bootstrap_95ci_high:+.4f}]")
    print(f"(b) {len(inf)} resolved motifs labelled in text; fit slope {fit[0]:.3f}; "
          f"Pearson {r[0]:.4f} (not printed on the figure)")
    print(f"\n저장: {stem.name}.{{png,pdf,svg}} + audit.json")


if __name__ == "__main__":
    main()
