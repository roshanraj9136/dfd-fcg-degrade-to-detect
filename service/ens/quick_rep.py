"""Quick look: a GenD model vs DFD-FCG on the 225 reported test videos (threshold 0.5, no fitting)."""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import numpy as np  # noqa: E402

import data as Dt  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "clip"
dfd, meta = Dt.dfd_rep()
L, _ = Dt.gend(name, "rep")
for cond in ("clean", "GB_5", "VC_5"):
    for g in ("ffpp", "fsh", "dfdc"):
        ys, sd, sg = [], [], []
        for vid, (lab, meth) in meta.items():
            gg = Dt.rep_group(vid, meth)
            if not (gg == g or (g == "fsh" and gg == "ffpp" and lab == 0)):
                continue
            k = (vid, cond, 0)
            if k not in L or k not in dfd:
                continue
            ys.append(lab)
            sd.append(float(Dt.logit(dfd[k])))
            sg.append(Dt.video_score(L[k]))
        a, b = Dt.metrics(ys, sd), Dt.metrics(ys, sg)
        print(f"{cond:5s} {g:5s} n={len(ys):3d}  DFD-FCG bacc {100 * a['bacc']:5.1f} auc {100 * a['auroc']:5.1f} "
              f"fa {100 * a['false_alarm']:4.0f} | GenD-{name} bacc {100 * b['bacc']:5.1f} auc {100 * b['auroc']:5.1f} "
              f"fa {100 * b['false_alarm']:4.0f} caught {100 * b['caught']:4.0f}")
