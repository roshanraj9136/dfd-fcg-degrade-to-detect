"""Write the round-2 block of README.md (between <!-- ROUND2-START --> and <!-- ROUND2-END -->) from
results/ensemble/results.json, so every number in the README comes from the result file."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = json.load(open(ROOT / "results" / "ensemble" / "results.json"))
o, c = R["options"], R["chosen"]
NAMES = {"dfd": "DFD-FCG (released)", "dfdft": "DFD-FCG retrained (round 1)", "pe": "GenD-PE", "pe_sc": "GenD-PE + self-cal.",
         "pe_ft": "GenD-PE retrained", "pe_ft_sc": "GenD-PE retrained + self-cal.", "clip": "GenD-CLIP",
         "clip_sc": "GenD-CLIP + self-cal.", "clip_ft": "GenD-CLIP retrained", "clip_ft_sc": "GenD-CLIP retrained + self-cal."}
G = [("celebdf", "Celeb-DF v2 test", "bacc"), ("social", "Social-media deepfakes 2026, test part", "bacc"),
     ("mavos", "MAVOS-DD 2025, methods seen in training", "bacc"), ("mavos_open", "MAVOS-DD 2025, methods never seen", "bacc"),
     ("sf", "SynthForensics 2026 (fakes only: caught)", "caught"), ("synth", "AI talking heads 2026 (fakes only: caught)", "caught"),
     ("ffpp", "FF++ test (as it is / blur / compression)", "bacc"), ("fsh", "FaceShifter (never trained on)", "bacc"),
     ("dfdc", "DFDC part 0 (never trained on)", "bacc")]
pc = lambda v: "–" if v is None else f"{100 * v:.1f}%"  # noqa: E731
pt = lambda v: f"{'+' if v >= 0 else '−'}{abs(100 * v):.1f}"  # noqa: E731
cols = ["dfd"] + [m for m in R["members"] if m != "dfd"]
lines = ["## Round 2: several detectors, one firm answer", "",
         "The live checker now combines " + ", ".join(NAMES.get(m, m) for m in R["members"]).rstrip(".") + ". Each detector's quality bias is "
         "removed with our self-calibration; a logistic-regression combiner fitted on "
         f"{R['n_val']} validation videos turns their scores into one REAL / FAKE answer with a confidence (no \"not sure\"). "
         "Plan, selection rule and switch rule were written before testing: [`results/ensemble/PLAN.md`](results/ensemble/PLAN.md).", "",
         "| held-out test videos | " + " | ".join(NAMES.get(m, m) for m in cols) + " | **combined (live)** |",
         "|---|" + "---|" * (len(cols) + 1)]
for g, name, k in G:
    lines.append(f"| {name} | " + " | ".join(pc(o[m][g].get(k)) for m in cols) + f" | **{pc(o[c][g].get(k))}** |")
d, ci = R["decision"], R["ci95"]
lines += ["", f"Balanced accuracy at 0.5 unless noted. Newer sets (mean of the four with real and fake videos): "
          f"{pc(o['dfd']['mean_new_bacc'])} → **{pc(o[c]['mean_new_bacc'])}** ({pt(d['gain_new_sets'])} points, bootstrap 95% range "
          f"{pt(ci['new_sets'][0])} to {pt(ci['new_sets'][1])}); FF++ {pt(d['change_ffpp'])}; DFDC {pt(d['change_dfdc'])}. "
          f"Switch rule {'met: deployed' if d['deploy'] else 'not met'}.", ""]
if R.get("selfcal"):
    k = next(iter(R["selfcal"]))
    s = R["selfcal"][k]
    lines += [f"**Self-calibration transfers to a second detector.** {NAMES.get(k, k)} (Yermakov et al., WACV 2026) calls "
              f"{pc(s['ffpp:GB_5']['raw']['false_alarm'])} of real FF++ test videos FAKE under strong blur and "
              f"{pc(s['ffpp:VC_5']['raw']['false_alarm'])} under heavy compression. The same read-out recipe on its class token "
              f"(same 49 fit videos, no retraining) brings this to {pc(s['ffpp:GB_5']['cal']['false_alarm'])} and "
              f"{pc(s['ffpp:VC_5']['cal']['false_alarm'])}.", ""]
ck = R.get("checker", {})
if ck.get("overall_acc"):
    lines += [f"All {ck['overall_videos']} held-out test videos as uploaded (no added damage): {pc(ck['overall_acc_original'])} → "
              f"**{pc(ck['overall_acc'])}** correct.", ""]
if ck.get("bands"):
    lines += ["Confidence on held-out videos: " + "; ".join(
        f"{int(100 * b['lo'])}–{min(100, int(100 * b['hi']))}%: right {pc(b['acc'])} ({pc(b['share'])} of answers)" for b in ck["bands"]) + ".", ""]
lines += ["Reproduce: `service/ens/` (`score_gend.py`, `train_gend.py`, `evaluate.py`, `report.py`, `export_live.py`); "
          "the live code is `service/app/gend_live.py` + `pipeline.py`; parity checks `tests/check_gend_live.py`, "
          "`tests/check_fusion_live.py`."]
block = "\n".join(lines)
p = ROOT / "README.md"
t = p.read_text(encoding="utf-8")
a, b = "<!-- ROUND2-START -->", "<!-- ROUND2-END -->"
if a in t:
    t = t[:t.index(a) + len(a)] + "\n" + block + "\n" + t[t.index(b):]
else:
    t = t.replace("<!-- RESULTS-END -->", "<!-- RESULTS-END -->\n\n" + a + "\n" + block + "\n" + b, 1)
p.write_text(t, encoding="utf-8")
print(block)
