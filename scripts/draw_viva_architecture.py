"""Render precise architecture diagrams as PNG/SVG and a two-page PDF.

Dimensions are checked against qa/engine.py, the released upstream svl.py,
service/app/gend_live.py, and the stored read-outs. No model inference needed.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/viva/public/architecture"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
INK, MUTED, BLUE, TEAL = "#0f172a", "#475569", "#4338ca", "#0f766e"


def canvas(title, sub, height):
    fig, ax = plt.subplots(figsize=(16, height / 100))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 1600), ylim=(height, 0))
    ax.axis("off")
    ax.text(60, 60, title, fontsize=25, weight="bold", color=INK, va="top")
    ax.text(60, 112, sub, fontsize=12.5, color=MUTED, va="top")
    return fig, ax


def box(ax, x, y, w, h, title, lines, color=BLUE, fill="#eef2ff", size=13, title_size=15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14",
                              facecolor=fill, edgecolor=color, linewidth=1.5))
    ax.text(x + 20, y + 20, title, va="top", fontsize=title_size, weight="bold", color=color)
    for i, line in enumerate(lines):
        ax.text(x + 20, y + 57 + 29 * i, line, va="top", fontsize=size, color=INK)


def arrow(ax, a, b, color="#64748b"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=18,
                                linewidth=1.7, color=color))


def note(ax, x, y, text, size=12, color=MUTED):
    ax.text(x, y, text, fontsize=size, color=color, va="top", linespacing=1.6)


def save(fig, name, pdf):
    fig.savefig(OUT / f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(OUT / f"{name}.svg", facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def dfd(pdf):
    fig, ax = canvas("1. Inside DFD-FCG: every important dimension",
                     "Blue = the authors' detector. Teal = our fitted self-calibration. Shapes use channel-first tensors after preprocessing.", 1770)
    box(ax, 60, 170, 430, 140, "Aligned face crops", ["RGB array: [N, 150, 150, 3]", "Select 10 frames per clip."])
    box(ax, 540, 170, 480, 140, "Resize + normalize", ["Model input: [B, 10, 3, 224, 224]", "B = clips in one forward-pass batch"])
    box(ax, 1070, 170, 470, 140, "Frozen CLIP ViT-L/14", ["24 layers; width 1024; 16 heads", "14 × 14 pixel patches"])
    arrow(ax, (490, 240), (535, 240)); arrow(ax, (1020, 240), (1065, 240))
    note(ax, 60, 335, "224 / 14 = 16 patches per side → 256 patch tokens + 1 CLS token → [B, 10, 257, 1024]. Side branches exclude CLS.")
    box(ax, 60, 390, 720, 265, "Spatial path: what does the face look like?", [
        "4 learned face-part queries: [4, 1024]",
        "Attention over patches: [B, 10, 4, 256]",
        "Weighted patch features: [B, 10, 4, 1024]",
        "Mean over 10 frames and 4 queries → [B, 1024]",
        "Sum the 24 layer outputs; do not concatenate them."])
    box(ax, 820, 390, 720, 265, "Temporal path: how do patches change?", [
        "Frame-to-frame affinities from Q, K and V",
        "Per-patch temporal matrices: 10 × 10",
        "Temporal convolution / projection + spatial convolution",
        "One value per spatial patch: [B, 1, 16, 16]",
        "Flatten → [B, 256]; sum the 24 layer outputs."])
    arrow(ax, (420, 655), (420, 685)); arrow(ax, (1180, 655), (1180, 685))
    box(ax, 60, 690, 720, 95, "Spatial vector y_s", ["[B, 1024] — 1024 appearance-feature values"])
    box(ax, 820, 690, 720, 95, "Temporal vector y_t", ["[B, 256] — 16 × 16 patch-response values"])
    note(ax, 60, 810, "Both vectors are used in TWO different places below. Same length (1280), different normalization.", size=14, color=INK)
    box(ax, 60, 855, 720, 300, "A. The authors' classification heads", [
        "s = LayerNorm(y_s): [B, 1024]",
        "t = LayerNorm(y_t): [B, 256]",
        "Spatial head: Linear(1024 → 2)",
        "Temporal head: Linear(256 → 2)",
        "Fusion input: concat(s, t) → [B, 1280]",
        "Fusion head: Linear(1280 → 2)",
        "Each head outputs [REAL logit, FAKE logit]."], size=12.5)
    box(ax, 820, 855, 720, 300, "B. Our quality-shift read-out", [
        "u = concat(y_s, y_t) → [B, 1280]",
        "Join RAW features first (before the head LayerNorms).",
        "Standardize with fitted mean and standard deviation:",
        "u_std = (u − mean) / std   → [B, 1280]",
        "G = u_std @ W.T + b      → [B, 3]",
        "W: [3, 1280]; b: [3]",
        "Three predicted shifts: spatial, temporal, fusion."], color=TEAL, fill="#f0fdfa", size=12.5)
    arrow(ax, (420, 1155), (420, 1200)); arrow(ax, (1180, 1155), (1180, 1200), TEAL)
    box(ax, 60, 1205, 1480, 185, "Correct the logits, then make ONE DFD score", [
        "For each head: d = FAKE logit − REAL logit. Stack the three differences → D: [B, 3].",
        "D_corrected = D − G  → [B, 3]     → sigmoid per head → mean of 3 probabilities → [B].",
        "Mean over the face's clips → p_dfd: ONE scalar. Live default: 8 clips per face, processed in batches.",
        "The live combiner receives this corrected score. The blur-probe strict score is not a combiner input."], color=TEAL, fill="#f0fdfa", size=13)
    box(ax, 60, 1430, 1480, 130, "Concatenate means append the numbers", [
        "Example: [s1, s2, s3] and [t1, t2] → [s1, s2, s3, t1, t2]. It is not addition or averaging.",
        "Here: 1024 + 256 = 1280. Four queries are pooled, not concatenated into a 4096-vector."], fill="#f8fafc", color=MUTED)
    note(ax, 60, 1600, "Papers: DFD-FCG — Han et al., CVPR 2025; CLIP — Radford et al., ICML 2021.")
    note(ax, 60, 1635, "Exact implementation: upstream src/model/clip/svl.py; qa/engine.py; service/app/pipeline.py; service/app/model.json.")
    note(ax, 60, 1670, "Important: 256 is the patch-grid size, not the number of frames. CLIP's final text-aligned projection is not used here.")
    note(ax, 60, 1710, "Roshan Raj · DFD-FCG extension · Dimensions checked against the implementation", size=11)
    save(fig, "dfd-dimensions", pdf)


def ensemble(pdf):
    fig, ax = canvas("2. GenD and the final ensemble: five scores, one answer",
                     "The final fusion combines scores. It does not concatenate DFD and GenD feature embeddings.", 1640)
    box(ax, 60, 165, 1480, 135, "Same face crops → sample up to 24 available frames", [
        "One GenD input batch: [M, 3, 224, 224], where M ≤ 24. Each network handles individual frames.",
        "We run two GenD-PE network instances: our fine-tuned checkpoint and the released checkpoint."], fill="#f8fafc", color=MUTED)
    for x, title, origin in [(60, "Fine-tuned GenD-PE", "We updated LayerNorm parameters + classifier head."),
                             (820, "Released GenD-PE", "Authors' released weights; unchanged by us.")]:
        box(ax, x, 345, 720, 225, title, [
            origin,
            "Perception Encoder L → f: [M, 1024]",
            "L2 normalize each frame's vector → f_norm: [M, 1024]",
            "Linear(1024 → 2) → logits: [M, 2]",
            "d = logit_FAKE − logit_REAL → [M]"], size=12.5)
        arrow(ax, (x + 180, 570), (x + 180, 610))
        arrow(ax, (x + 540, 570), (x + 540, 610), TEAL)
        box(ax, x, 615, 345, 245, "Raw score", ["d: [M]", "sigmoid(d) → [M]", "Mean over frames", "→ one probability"], size=12)
        box(ax, x + 375, 615, 345, 245, "Corrected score", ["Standardize f_norm", "Read-out: 1024 → 1", "g: [M]; use d − g", "sigmoid, then mean", "→ one probability"], color=TEAL, fill="#f0fdfa", size=12)
    note(ax, 60, 895, "Each GenD network runs once for both of its scores. Each corrected path has its own fitted W: [1, 1024], b: [1].")
    f = json.loads((ROOT / "service/app/fusion.json").read_text())
    names = [("DFD corrected", "p_dfd"), ("Tuned PE raw", "p_pe_ft"), ("Tuned PE corrected", "p_pe_ft_sc"),
             ("Released PE raw", "p_pe"), ("Released PE corrected", "p_pe_sc")]
    for i, (label, ident) in enumerate(names):
        x = 60 + i * 300
        box(ax, x, 955, 280, 100, label, [f"{ident}: scalar"], size=12, title_size=11.5,
            color=TEAL if i in (0, 2, 4) else BLUE, fill="#f0fdfa" if i in (0, 2, 4) else "#eef2ff")
        arrow(ax, (x + 140, 1055), (x + 140, 1095))
    weights = ", ".join(f"{w:.3f}" for w in f["coef"])
    box(ax, 60, 1100, 1480, 190, "Stack FIVE score logits → logistic regression", [
        "x = [logit(p_dfd), logit(p_pe_ft), logit(p_pe_ft_sc), logit(p_pe), logit(p_pe_sc)] → [5] per face",
        f"Weights w: [5] = [{weights}]; intercept b = {f['intercept']:.3f}",
        "z = w · x + b → scalar;  p_face = sigmoid(z) → scalar between 0 and 1",
        "logit(p) = ln(p / (1 − p)). Weights were fitted on validation videos; this is not a majority vote."], color=TEAL, fill="#f0fdfa", size=12.5)
    arrow(ax, (800, 1290), (800, 1325))
    box(ax, 60, 1330, 1480, 130, "One video verdict", [
        "Track up to 2 faces → take max(p_face). If p ≥ 0.5: FAKE; otherwise: REAL.",
        "Confidence = max(p, 1 − p). Invalid scores return an analysis error. No weights are learned from an upload."], fill="#f8fafc", color=MUTED)
    note(ax, 60, 1500, "Papers: GenD — Yermakov et al., WACV 2026; Perception Encoder — Bolya et al., 2025 (arXiv:2504.13181).")
    note(ax, 60, 1535, "Our additions: fitted quality read-outs, the GenD fine-tune, learned score fusion, evaluation and the deployed checker.")
    note(ax, 60, 1570, "Five separate score inputs from three networks; raw and corrected GenD scores share a forward pass.", size=11)
    save(fig, "ensemble-dimensions", pdf)


if __name__ == "__main__":
    with PdfPages(OUT / "architecture-explained.pdf") as pdf:
        dfd(pdf)
        ensemble(pdf)
    print("Wrote two PNGs, two SVGs and architecture-explained.pdf")
