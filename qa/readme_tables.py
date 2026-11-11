"""
Print the Markdown result tables used in README.md from results/*.json (keeps the README honest).

    python -m qa.readme_tables > results/tables.md
"""
import json

from qa.config import ROOT

LABEL = {"original": "DFD-FCG (released)", "platt": "Platt recalibration", "qshift": "quality-regression shift (face crop)",
         "en": "background-quality nulling", "refit": "quality-augmented head re-fit",
         "d2d-pB2": "blur-probe contrast", "d2d-pJ50": "JPEG-probe contrast", "d2d-lr": "probe responses, learned",
         "selfcal": "**self-calibrated (ours)**", "selfcal+d2d": "**self-calibrated + blur probe, strict (ours)**"}
ORDER = ["original", "platt", "qshift", "en", "refit", "d2d-pB2", "d2d-lr", "selfcal", "selfcal+d2d"]


def p(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}%"


def f(x):
    return "–" if x is None else f"{x:.3f}"


def main():
    M = json.load(open(ROOT / "results" / "metrics.json"))
    A = json.load(open(ROOT / "results" / "analysis.json"))
    rows = {(r["method"], r["regime"]): r for r in M["summary"]}
    pc = M["per_condition"]
    for regime, title in [("all", "train split → test split, degradation types seen at other severities"),
                          ("lopo", "train split → test split, tested degradation type unseen"),
                          ("cv_lopo", "5-fold CV over test identity pairs, tested type unseen")]:
        print(f"\n**{title}**\n")
        print("| method | clean AUROC | degraded AUROC | false alarms @0.5 | worst | fakes caught @0.5 | balanced acc | "
              "detect @5% FA | @10% | @20% | mixed-pool AUROC |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for m in ORDER:
            r = rows.get((m, regime))
            if not r:
                continue
            ref = rows.get((m, "cv_all" if regime.startswith("cv") else "all"), {})
            print(f"| {LABEL[m]} | {f(r['clean_auroc'])} | {f(r['deg_mean_auroc'])} | {p(r['deg_mean_fpr@0.5'])} | "
                  f"{p(r['deg_worst_fpr@0.5'], 0)} | {p(r['deg_mean_tpr@0.5'])} | {p(r['deg_mean_bacc@0.5'])} | "
                  f"{p(r.get('tpr@mfpr05'))} | {p(r.get('tpr@mfpr10'))} | {p(r.get('tpr@mfpr20'))} | {f(ref.get('pooled_auroc'))} |")
    print("\n**Per condition, released vs self-calibrated (type unseen), threshold 0.5: false alarms / fakes caught / AUROC**\n")
    print("| condition | released | self-calibrated |")
    print("|---|---|---|")
    for c in ["clean", "RE_0", "CS_3", "CC_3", "BW_3", "GNC_3", "GB_1", "GB_3", "GB_5", "JPEG_1", "JPEG_3", "JPEG_5",
              "VC_1", "VC_3", "VC_5"]:
        o, s = pc["original"]["all"].get(c), pc["selfcal"]["lopo"].get(c)
        if o and s:
            print(f"| {c} | {p(o['fpr@0.5'], 0)} / {p(o['tpr@0.5'], 0)} / {f(o['auroc'])} | "
                  f"{p(s['fpr@0.5'], 0)} / {p(s['tpr@0.5'], 0)} / {f(s['auroc'])} |")
    print("\n**Unseen manipulation and unseen dataset (clean videos)**\n")
    print("| method | FaceShifter AUROC | FaceShifter caught @0.5 | DFDC AUROC | DFDC balanced acc @0.5 | DFDC false alarms | DFDC caught |")
    print("|---|---|---|---|---|---|---|")
    for m in ["original", "selfcal", "selfcal+d2d", "d2d-pB2"]:
        r = pc.get(m, {})
        fs, dc = r.get("fsh", {}), r.get("dfdc", {})
        print(f"| {LABEL[m]} | {f(fs.get('auroc'))} | {p(fs.get('tpr@0.5'))} | {f(dc.get('auroc'))} | {p(dc.get('bacc@0.5'))} | "
              f"{p(dc.get('fpr@0.5'))} | {p(dc.get('tpr@0.5'))} |")
    F = A["fragility"]["pB2"]
    print("\n**Response of each head to a 2-px blur of the aligned face (clean inputs, mean logit change)**\n")
    print("| group | spatial | temporal | fusion |")
    print("|---|---|---|---|")
    for g in ["REAL", "DF", "F2F", "FS", "NT", "FSh", "DFDC_R", "DFDC_F"]:
        if g in F:
            v = F[g]["mean_response"]
            print(f"| {g} | {v[0]:+.2f} | {v[1]:+.2f} | {v[2]:+.2f} |")
    print("\n| input condition | real (fusion) | fake (fusion) | AUROC of the response alone |")
    print("|---|---|---|---|")
    for c, v in F["by_input"].items():
        print(f"| {c} | {v['real_mean'][2]:+.2f} | {v['fake_mean'][2]:+.2f} | {v['auroc_minus_response_fusion']:.3f} |")
    if "dfdc" in F:
        v = F["dfdc"]
        print(f"| DFDC clean | {v['real_mean'][2]:+.2f} | {v['fake_mean'][2]:+.2f} | {v['auroc_minus_response_fusion']:.3f} |")
    print("\n**Earlier two-video findings re-tested (clean FF++ test videos)**\n")
    R = A["replication"]
    for g in ["REAL", "FAKE", "DF", "F2F", "FS", "NT", "FSh"]:
        if g in R:
            r = R[g]
            top_q = max(r["query_share"], key=r["query_share"].get)
            top_s = max(r["region_spatial_share"], key=r["region_spatial_share"].get)
            top_t = max(r["region_temporal_share"], key=r["region_temporal_share"].get)
            print(f"- {g} (n={r['n_videos']}): accuracy {p(r['accuracy'], 0)}, spatial-only {p(r['acc_spatial_only'], 0)}, "
                  f"temporal-only {p(r['acc_temporal_only'], 0)}; spatial share (Shapley) {p(r['spatial_share_shapley'], 0)}; "
                  f"top query {top_q} {p(r['query_share'][top_q], 0)}; top spatial region {top_s} "
                  f"{p(r['region_spatial_share'][top_s], 0)}; top temporal region {top_t} {p(r['region_temporal_share'][top_t], 0)}; "
                  f"query signed {r['query_signed']}; top spatial layers {r['top_layers_spatial']}, temporal {r['top_layers_temporal']}")
    fp = A["fingerprints"]
    print(f"\nFingerprints: {fp['n_videos']} fake videos, method identified from the explanation {p(fp['accuracy_explanation'], 0)} "
          f"vs {p(fp['accuracy_logits_only'], 0)} from the 3 logits (chance 25%). Confusion {fp['confusion']}")
    cf = A["counterfactual"]
    print(f"\nEN counterfactual: {cf}")
    print(f"\nGeometry: { {h: A['geometry'][h] for h in 'sta'} }")


if __name__ == "__main__":
    main()
