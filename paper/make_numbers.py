"""Every number in the paper comes from the result files through this script.
    python paper/make_numbers.py      -> paper/numbers.tex (LaTeX macros) and paper/tables.tex (tables)
Sources: results/metrics.json, results/analysis.json (round 1, DFD-FCG + self-calibration),
results/ensemble/results.json and all_options_round2.json (round 2, GenD and the fused checker)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
M = json.load(open(ROOT / "results" / "metrics.json", encoding="utf-8"))
A = json.load(open(ROOT / "results" / "analysis.json", encoding="utf-8"))
E = json.load(open(ROOT / "results" / "ensemble" / "results.json", encoding="utf-8"))
O2 = json.load(open(ROOT / "results" / "ensemble" / "selfcal_transfer.json", encoding="utf-8"))

S = {(r["method"], r["regime"]): r for r in M["summary"]}
PC = M["per_condition"]
pc = lambda v: f"{100 * v:.1f}\\%"  # noqa: E731
p0 = lambda v: f"{100 * v:.0f}\\%"  # noqa: E731
f3 = lambda v: f"{v:.3f}"  # noqa: E731
macros = {}


DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six",
          "7": "Seven", "8": "Eight", "9": "Nine"}


def mac(name, val):
    """LaTeX macro names cannot contain digits: GB5 -> GBFive."""
    macros["".join(DIGITS.get(ch, ch) for ch in name)] = val


# ---------------------------------------------------------------- round 1: decision collapse and self-calibration
for key, meth in (("Orig", "original"), ("Platt", "platt"), ("SC", "selfcal"), ("Strict", "selfcal+d2d"), ("Qs", "qshift"),
                  ("Refit", "refit")):
    r = S[(meth, "lopo")]
    mac(f"{key}DegAuroc", f3(r["deg_mean_auroc"]))
    mac(f"{key}FA", pc(r["deg_mean_fpr@0.5"]))
    mac(f"{key}WorstFA", p0(r["deg_worst_fpr@0.5"]))
    mac(f"{key}TPR", pc(r["deg_mean_tpr@0.5"]))
    mac(f"{key}Bacc", pc(r["deg_mean_bacc@0.5"]))
    mac(f"{key}Pooled", f3(S[(meth, "all")]["pooled_auroc"]))   # pooled AUROC exists for the seen-type protocol only
    mac(f"{key}TprFive", pc(r["tpr@mfpr05"]))
    mac(f"{key}TprTen", pc(r["tpr@mfpr10"]))
    mac(f"{key}CleanFA", pc(r["clean_fpr05"]))
    mac(f"{key}ConFA", pc(S[(meth, "all")]["deg_mean_fpr@0.5"]))
    mac(f"{key}CvFA", pc(S[(meth, "cv_lopo")]["deg_mean_fpr@0.5"]))
for cond in ("GB_5", "VC_5", "JPEG_5", "clean"):
    k = cond.replace("_", "")
    mac(f"OrigFA{k}", p0(PC["original"]["lopo"][cond]["fpr@0.5"]))
    mac(f"OrigAuroc{k}", f3(PC["original"]["lopo"][cond]["auroc"]))
    mac(f"SCFA{k}", p0(PC["selfcal"]["lopo"][cond]["fpr@0.5"]))
    mac(f"SCTPR{k}", p0(PC["selfcal"]["lopo"][cond]["tpr@0.5"]))
    mac(f"OrigTPR{k}", p0(PC["original"]["lopo"][cond]["tpr@0.5"]))
fr = A["fragility"]["pB2"]["by_input"]["clean"]
mac("ProbeReal", f"${fr['real_mean'][2]:+.2f}$")
mac("ProbeFake", f"${fr['fake_mean'][2]:+.2f}$")
mac("ProbeAuroc", f3(fr["auroc_minus_response_fusion"]))
fp = A["fingerprints"]
mac("FpAcc", pc(fp["accuracy_explanation"]))
mac("FpLogit", pc(fp["accuracy_logits_only"]))
mac("FpN", str(fp["n_videos"]))

# ---------------------------------------------------------------- round 2: transfer to GenD, fused checker
for key, src, raw, cal in (("Clip", O2, "clip", "clip_sc"), ("Pe", O2, "pe", "pe_sc")):
    for g in ("ffpp", "dfdc"):
        for c in ("clean", "GB_5", "VC_5"):
            k = f"{g}:{c}"
            t = g.capitalize() + c.replace("_", "")
            mac(f"{key}RawFA{t}", p0(src[raw][k]["false_alarm"]))
            mac(f"{key}CalFA{t}", p0(src[cal][k]["false_alarm"]))
            mac(f"{key}RawBacc{t}", pc(src[raw][k]["bacc"]))
            mac(f"{key}CalBacc{t}", pc(src[cal][k]["bacc"]))
            mac(f"{key}RawAuroc{t}", f3(src[raw][k]["auroc"]))
            mac(f"{key}CalAuroc{t}", f3(src[cal][k]["auroc"]))
o, ch, d, ci, ck = E["options"], E["chosen"], E["decision"], E["ci95"], E["checker"]
mac("NVal", str(E["n_val"]))
mac("NewBaseBacc", pc(o["dfd"]["mean_new_bacc"]))
mac("NewFinBacc", pc(o[ch]["mean_new_bacc"]))
mac("NewGain", f"${100 * d['gain_new_sets']:+.1f}$")
mac("NewCiLo", f"${100 * ci['new_sets'][0]:+.1f}$")
mac("NewCiHi", f"${100 * ci['new_sets'][1]:+.1f}$")
mac("FfppBase", pc(o["dfd"]["ffpp"]["bacc"]))
mac("FfppFin", pc(o[ch]["ffpp"]["bacc"]))
mac("FfppChange", f"${100 * d['change_ffpp']:+.1f}$")
mac("DfdcBase", pc(o["dfd"]["dfdc"]["bacc"]))
mac("DfdcFin", pc(o[ch]["dfdc"]["bacc"]))
mac("DfdcChange", f"${100 * d['change_dfdc']:+.1f}$")
mac("FfppCleanFAFin", pc(o[ch]["ffpp:clean"]["false_alarm"]))
mac("FfppCleanFABase", pc(o["dfd"]["ffpp:clean"]["false_alarm"]))
mac("AllAcc", pc(ck["overall_acc"]))
mac("AllAccBase", pc(ck["overall_acc_original"]))
mac("AllN", str(ck["overall_videos"]))
for i, b in enumerate(ck["bands"]):
    n = "ABC"[i]
    mac(f"Band{n}Acc", pc(b["acc"]))
    mac(f"Band{n}Share", pc(b["share"]))
mac("PeftValBase", "0.719")      # train_gend.py validation score at step 0 and best (D:/dfd-data/ens/ft_pe/train_log.jsonl)
mac("PeftValBest", "0.919")
(OUT / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()), encoding="utf-8")

# ---------------------------------------------------------------- tables
NAMES = {"dfd": "DFD-FCG + SC", "pe": "GenD-PE", "pe_ft": "GenD-PE ft", "pe_ft_sc": "GenD-PE ft + SC", "pe_sc": "GenD-PE + SC"}
G = [("celebdf", "Celeb-DF v2 (2020)", "bacc"), ("social", "Social-media deepfakes (2026)", "bacc"),
     ("mavos", "MAVOS-DD, seen methods (2025)", "bacc"), ("mavos_open", "MAVOS-DD, unseen methods (2025)", "bacc"),
     ("sf", "SynthForensics$^\\dagger$ (2026)", "caught"), ("synth", "AI talking heads$^\\dagger$ (2026)", "caught"),
     ("ffpp", "FF++ test$^\\ast$", "bacc"), ("fsh", "FaceShifter$^\\ast$", "bacc"), ("dfdc", "DFDC part 0$^\\ast$", "bacc")]
cols = ["dfd", "pe", "pe_ft"]
rows = []
for g, name, k in G:
    vals = [o[m][g].get(k) for m in cols] + [o[ch][g].get(k)]
    best = max(v for v in vals if v is not None)
    cells = [("\\textbf{" + pc(v) + "}") if v == best else pc(v) for v in vals]
    n = o[ch][g]["n"] if "n" in o[ch][g] else o[ch][f"{g}:clean"]["n"]
    rows.append(f"{name} & {n} & " + " & ".join(cells) + " \\\\")
t_fusion = ("\\begin{tabular}{lrcccc}\n\\toprule\nHeld-out test set & $n$ & " + " & ".join(NAMES[m] for m in cols) +
            " & Fused (live) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
for c, cn in (("clean", "as uploaded"), ("GB_5", "strong blur"), ("VC_5", "heavy H.264")):
    for g, gn in (("ffpp", "FF++"), ("dfdc", "DFDC")):
        k = f"{g}:{c}"
        r = []
        for raw, cal in (("clip", "clip_sc"), ("pe", "pe_sc")):
            r += [f"{p0(O2[raw][k]['false_alarm'])} $\\rightarrow$ \\textbf{{{p0(O2[cal][k]['false_alarm'])}}}",
                  f"{f3(O2[raw][k]['auroc'])} $\\rightarrow$ {f3(O2[cal][k]['auroc'])}"]
        rows.append(f"{gn}, {cn} & " + " & ".join(r) + " \\\\")
t_transfer = ("\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{GenD-CLIP} & \\multicolumn{2}{c}{GenD-PE} \\\\\n"
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nTest videos & false alarms & AUROC & false alarms & AUROC \\\\\n"
              "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
for key, meth in (("Released DFD-FCG", "original"), ("Platt recalibration", "platt"), ("Quality-regression shift", "qshift"),
                  ("Quality-augmented head re-fit", "refit"), ("Self-calibration (ours)", "selfcal"),
                  ("Self-calibration + blur probe", "selfcal+d2d")):
    r = S[(meth, "lopo")]
    rows.append(f"{key} & {f3(r['deg_mean_auroc'])} & {pc(r['deg_mean_fpr@0.5'])} & {p0(r['deg_worst_fpr@0.5'])} & "
                f"{pc(r['deg_mean_tpr@0.5'])} & {pc(r['deg_mean_bacc@0.5'])} & {pc(r['tpr@mfpr05'])} & "
                f"{f3(S[(meth, 'all')]['pooled_auroc'])} \\\\")
t_round1 = ("\\begin{tabular}{lccccccc}\n\\toprule\nMethod & AUROC & FA@0.5 & worst FA & TPR@0.5 & bal.\\ acc. & TPR@5\\%FA & "
            "pooled AUROC \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
(OUT / "tables.tex").write_text("\\newcommand{\\TabRoundOne}{" + t_round1 + "}\n\\newcommand{\\TabTransfer}{" + t_transfer +
                                "}\n\\newcommand{\\TabFusion}{" + t_fusion + "}\n", encoding="utf-8")
print(len(macros), "macros; tables written")
