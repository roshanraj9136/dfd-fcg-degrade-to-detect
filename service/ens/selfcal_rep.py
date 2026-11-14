"""Self-calibration carried over to a GenD model: raw vs corrected on the reported test videos.
    python service/ens/selfcal_rep.py clip pe ..."""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import data as Dt  # noqa: E402

_, meta = Dt.dfd_rep()
for name in sys.argv[1:]:
    cal = Dt.GenDSelfCal(name)
    L, F = Dt.gend(name, "rep")
    print(name, "targets", {k: round(v, 2) for k, v in cal.targets.items() if k.endswith("_4")})
    for cond in ("clean", "GB_5", "VC_5"):
        for g in ("ffpp", "dfdc"):
            ys, a, b = [], [], []
            for vid, (lab, meth) in meta.items():
                if Dt.rep_group(vid, meth) != g:
                    continue
                k = (vid, cond, 0)
                ys.append(lab)
                a.append(Dt.video_score(L[k]))
                b.append(Dt.video_score(cal.correct(L[k], F[k])))
            m1, m2 = Dt.metrics(ys, a), Dt.metrics(ys, b)
            print(f"  {cond:5s} {g:5s} raw: fa {100 * m1['false_alarm']:3.0f} caught {100 * m1['caught']:3.0f} "
                  f"bacc {100 * m1['bacc']:5.1f} auc {100 * m1['auroc']:5.1f} | self-cal: fa {100 * m2['false_alarm']:3.0f} "
                  f"caught {100 * m2['caught']:3.0f} bacc {100 * m2['bacc']:5.1f} auc {100 * m2['auroc']:5.1f}")
