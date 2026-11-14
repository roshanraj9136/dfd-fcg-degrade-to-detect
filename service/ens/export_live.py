"""Write the live checker's fusion files for one option (e.g. dfd+dfdft+pe_ft_sc) into service/app/:
  fusion.json        members, combiner weights (refitted on the validation videos exactly as evaluate.py), and per
                     member: type, weights / read-out files, GenD backbone, fine-tuned tensors, self-calibration read-out
  model_ft_v1.json   round-1 fine-tuned DFD-FCG read-out (if dfdft is a member)
  gend_<name>.pt     fine-tuned GenD LayerNorm + head tensors (if a *_ft member is used)
    python service/ens/export_live.py <option>
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import data as Dt  # noqa: E402
import evaluate as Ev  # noqa: E402

OPT = sys.argv[1]
APP = ROOT / "service" / "app"
FT = {"pe_ft": Dt.cache_file("ft_pe/best.pt"), "clip_ft": Dt.cache_file("ft_clip/best.pt")}
NAMES = {"dfd": "DFD-FCG", "dfdft": "DFD-FCG (retrained)", "pe": "GenD-PE", "pe_ft": "GenD-PE (retrained)",
         "clip": "GenD-CLIP", "clip_ft": "GenD-CLIP (retrained)"}


def main():
    mem = OPT.split("+")
    new, _ = Ev.load_members(mem)
    lr, nval = Ev.fit_fusion(new, mem)
    det = {}
    for m in mem:
        if m == "dfd":
            continue
        if m == "dfdft":
            shutil.copy(Dt.D / "readout_new.json", APP / "model_ft_v1.json")
            det[m] = dict(type="dfd", weights="weights_ft_v1.ckpt", model_json="model_ft_v1.json")
            continue
        base = m[:-3] if m.endswith("_sc") else m
        kind = base.split("_")[0]
        cfg = dict(type="gend", backbone=kind, frames=24)
        if base.endswith("_ft"):
            dst = APP / f"gend_{base}.pt"
            shutil.copy(FT[base], dst)
            cfg["ln"] = dst.name
        if m.endswith("_sc"):
            cal = Dt.GenDSelfCal(base.replace("_ft", "ft"))
            cfg["selfcal"] = dict(mean=cal.sc.mean_.tolist(), scale=cal.sc.scale_.tolist(),
                                  coef=[cal.reg.coef_.tolist()], intercept=[float(cal.reg.intercept_)])
        det[m] = cfg
    out = dict(members=mem, coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]), detectors=det,
               names={m: NAMES.get(m.replace("_sc", ""), m) + (", quality-checked" if m.endswith("_sc") and m != "dfd" else "")
                      for m in mem}, fitted_on_validation_videos=nval, val_conds=list(Ev.VAL_CONDS),
               note="Combiner fitted on the validation splits only (service/ens/evaluate.py fit_fusion).")
    (APP / "fusion.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", APP / "fusion.json", {k: v for k, v in out.items() if k not in ("detectors",)})


if __name__ == "__main__":
    main()
