"""Per-condition numbers on the reported test videos from eval.json:  python service/ens/show_rep.py dfd clip clip_sc"""
import json
import sys

r = json.load(open(r"D:\dfd-data\ens\eval.json"))
for m in sys.argv[1:]:
    for g in ("ffpp", "fsh", "dfdc"):
        cells = []
        for c in ("clean", "GB_5", "VC_5"):
            x = r[m][f"{g}:{c}"]
            cells.append(f"{c:5s} bacc {100 * x['bacc']:5.1f} fa {100 * x['false_alarm']:3.0f} "
                         f"caught {100 * x['caught']:3.0f} auc {100 * x['auroc']:5.1f}")
        print(f"{m:<14} {g:<5}", " | ".join(cells))
