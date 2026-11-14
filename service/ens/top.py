"""Top options of an eval.json by validation CV score:  python service/ens/top.py <eval.json> [n=12]"""
import json
import sys

r = json.load(open(sys.argv[1]))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
f = {k: v for k, v in r.items() if isinstance(v, dict) and "cv_val_bacc" in v}
for k in sorted(f, key=lambda k: -f[k]["cv_val_bacc"])[:n]:
    v = f[k]
    print(f"{k:<38} cv {100 * v['cv_val_bacc']:5.1f} | new {100 * v['mean_new_bacc']:5.1f} ffpp {100 * v['ffpp']['bacc']:5.1f} "
          f"fsh {100 * v['fsh']['bacc']:5.1f} dfdc {100 * v['dfdc']['bacc']:5.1f} | sf {100 * v['sf']['caught']:5.1f} "
          f"synth {100 * v['synth']['caught']:5.1f}")
d = r["dfd"]
print(f"{'dfd (live today)':<38}          | new {100 * d['mean_new_bacc']:5.1f} ffpp {100 * d['ffpp']['bacc']:5.1f} "
      f"fsh {100 * d['fsh']['bacc']:5.1f} dfdc {100 * d['dfdc']['bacc']:5.1f}")
