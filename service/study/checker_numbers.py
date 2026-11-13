"""The few accuracy numbers the public checker shows, taken from results/accuracy_study/upgrade.json.
python service/study/checker_numbers.py   ->  web/checker/src/accuracy.json"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
U = json.load(open(ROOT / "results" / "accuracy_study" / "upgrade.json"))
g, s = U["groups"], U["sure"]
out = dict(
    videos=U["videos"],
    ffpp_clean_acc=g["ffpp"]["clean"]["v2"]["acc"],
    ffpp_blur_acc=g["ffpp"]["GB_5"]["v2"]["acc"],
    ffpp_blur_acc_original=g["ffpp"]["GB_5"]["orig"]["acc"],
    dfdc_clean_acc=g["dfdc"]["clean"]["v2"]["acc"],
    sure_acc=s["all"]["acc_sure"], sure_acc_ffpp=s["ffpp"]["acc_sure"], unsure_acc=s["all"]["acc_unsure"],
    sure_share=s["all"]["share"],
)
(ROOT / "web" / "checker" / "src" / "accuracy.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(out)
