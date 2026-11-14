"""Self-calibration carried over, unchanged, to GenD (CPU only, from stored scores).
For each GenD network: raw vs self-calibrated, per reported test condition (FF++ / FaceShifter / DFDC x clean / GB_5 /
VC_5) and per newer test set, at the 0.5 line (no refitting of any threshold).
    python service/ens/selfcal_transfer.py     -> results/ensemble/selfcal_transfer.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import evaluate as Ev  # noqa: E402

NETS = ["clip", "pe", "pe_ft"]
names = [n for b in NETS for n in (b, b + "_sc")]
new, rep = Ev.load_members(["dfd"] + names)
out = {n: Ev.table(new, rep, [n]) for n in ["dfd"] + names if n in new}
p = ROOT / "results" / "ensemble" / "selfcal_transfer.json"
p.write_text(json.dumps(out, indent=1))
for b in NETS:
    r, c = out[b], out[b + "_sc"]
    print(b, " | ".join(f"{k}: FA {100 * r[k]['false_alarm']:.0f}->{100 * c[k]['false_alarm']:.0f}"
                         for k in ("ffpp:clean", "ffpp:GB_5", "ffpp:VC_5", "dfdc:GB_5")))
print("wrote", p)
