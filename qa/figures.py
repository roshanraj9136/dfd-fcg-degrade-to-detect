"""
Figures for the README / report.

    python -m qa.figures         # writes results/figures/*.png
"""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from qa.config import ROOT  # noqa: E402

OUT = ROOT / "results" / "figures"
ORDER = ["clean", "RE_0", "CS_3", "CC_3", "BW_3", "GNC_3", "GB_1", "GB_3", "GB_5",
         "JPEG_1", "JPEG_3", "JPEG_5", "VC_1", "VC_3", "VC_5"]
COL = {"original": "#d6336c", "platt": "#7950f2", "qshift": "#f08c00", "en": "#2b8a3e",
       "en-face": "#1098ad", "en-proj": "#adb5bd", "refit": "#495057"}
LABEL = {"original": "DFD-FCG (released)", "platt": "Platt recalibration", "qshift": "quality shift (face crop)",
         "en": "Evidence Nulling (ours)", "en-face": "EN, face-crop quality", "en-proj": "attribution projection",
         "refit": "augmented head re-fit"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def f1_collapse(M):
    pc = M["per_condition"]
    conds = [c for c in ORDER if c in pc["original"]["all"]]
    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(10, 4))
    o = pc["original"]["all"]
    n = pc["en"]["lopo"]
    ax.plot(x, [o[c]["auroc"] for c in conds], "-o", color="#1971c2", label="AUROC (released)")
    ax.plot(x, [o[c]["fpr@0.5"] for c in conds], "-o", color=COL["original"], label="real videos called FAKE (released)")
    ax.plot(x, [n[c]["fpr@0.5"] for c in conds], "-o", color=COL["en"],
            label="real videos called FAKE (Evidence Nulling, type unseen)")
    ax.plot(x, [o[c]["tpr@0.5"] for c in conds], "--", color=COL["original"], alpha=.6, label="fakes caught (released)")
    ax.plot(x, [n[c]["tpr@0.5"] for c in conds], "--", color=COL["en"], alpha=.6, label="fakes caught (Evidence Nulling)")
    ax.set_xticks(x, conds, rotation=45)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("rate")
    ax.set_title("AUROC vs false alarms at the model's threshold (0.5), FF++ test split")
    ax.legend(fontsize=8, ncol=2, loc="center left")
    fig.tight_layout()
    fig.savefig(OUT / "f1_collapse.png", dpi=160)
    plt.close(fig)


def f2_methods(M):
    pc = M["per_condition"]
    conds = [c for c in ORDER if c in pc["original"]["all"] and c != "clean"]
    methods = [m for m in COL if m in pc]
    w = 0.8 / len(methods)
    x = np.arange(len(conds))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for k, m in enumerate(methods):
        src = pc[m]["all"] if m == "original" else pc[m]["lopo"]
        axes[0].bar(x + k * w, [src[c]["fpr@0.5"] for c in conds], w, color=COL[m], label=LABEL[m])
        axes[1].bar(x + k * w, [src[c]["tpr@0.5"] for c in conds], w, color=COL[m])
    axes[0].set_ylabel("false-alarm rate")
    axes[1].set_ylabel("detection rate")
    axes[1].set_xticks(x + 0.4 - w / 2, conds, rotation=45)
    axes[0].legend(fontsize=8, ncol=4)
    axes[0].set_title("Every correction tested on a degradation type it never saw (threshold 0.5)")
    fig.tight_layout()
    fig.savefig(OUT / "f2_methods.png", dpi=160)
    plt.close(fig)


def f3_drift(arr):
    conds = [c for c in ["GB_5", "JPEG_5", "VC_5", "GNC_3"] if f"drift_s_{c}" in arr.files]
    fig, axes = plt.subplots(len(conds), 2, figsize=(14, 1.7 * len(conds) + 0.8), sharex=True)
    axes = np.atleast_2d(axes)
    for row, c in zip(axes, conds):
        Ms = [np.vstack([arr[f"drift_s_{c}"].T, arr[f"drift_t_{c}"][None]])]
        if f"pred_s_{c}" in arr.files:
            Ms.append(np.vstack([arr[f"pred_s_{c}"].T, arr[f"pred_t_{c}"][None]]))
        v = max(np.abs(M).max() for M in Ms)
        for ax, M, t in zip(row, Ms, ["measured", "predicted from background quality"]):
            ax.imshow(M, cmap="PiYG_r", vmin=-v, vmax=v, aspect="auto")
            ax.set_yticks(range(5), ["S lips", "S skin", "S eyes", "S nose", "temporal"], fontsize=8)
            ax.set_title(f"{c}: {t}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xticks(range(24), range(1, 25), fontsize=7)
        ax.set_xlabel("CLIP layer")
    fig.suptitle("Change of each exact term on real test videos (degraded - clean); pink = towards FAKE")
    fig.tight_layout()
    fig.savefig(OUT / "f3_drift_anatomy.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    M = json.load(open(ROOT / "results" / "metrics.json"))
    arr = np.load(ROOT / "results" / "arrays.npz")
    f1_collapse(M)
    f2_methods(M)
    f3_drift(arr)
    print("figures in", OUT)


if __name__ == "__main__":
    main()
