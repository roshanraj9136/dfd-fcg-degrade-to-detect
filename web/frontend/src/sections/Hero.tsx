import { ArrowDown, GitBranch } from "lucide-react";
import type { Conditions, Summary } from "../data";
import { pct, sgn } from "../data";
import type { Analysis } from "./Anatomy";
import { Kpi } from "../components/ui";

export const REPO = "https://github.com/roshanraj9136/dfd-fcg-degrade-to-detect";

export function Hero({ cond, summary, analysis }: { cond: Conditions; summary: Summary; analysis: Analysis }) {
  const o = cond.original.all;
  const deg = Object.keys(o).filter((c) => c !== "clean" && c !== "RE_0");
  const worst = deg.reduce((a, b) => (o[b]["fpr@0.5"] > o[a]["fpr@0.5"] ? b : a), deg[0]);
  const meanAuc = deg.reduce((s, c) => s + (o[c].auroc ?? 0), 0) / deg.length;
  const main = cond[summary.main];
  const ours = main?.lopo?.[worst] ?? main?.all?.[worst];
  const fr = analysis.fragility?.pB2?.by_input?.clean;
  const row = (m: string) => summary.table.find((r) => r.method === m && r.regime === "all");
  return (
    <header className="glow relative overflow-hidden border-b border-edge">
      <div className="mx-auto max-w-6xl px-5 pb-16 pt-24">
        <div className="flex flex-wrap gap-2">
          <span className="chip">Extension of DFD-FCG · CVPR 2025</span>
          <span className="chip">{summary.n.test_real} real + {summary.n.test_fake} fake FF++ test videos · 15 conditions</span>
          <span className="chip">B.Tech project · Roshan Raj</span>
        </div>
        <h1 className="mt-6 max-w-5xl text-4xl font-semibold leading-tight tracking-tight text-white md:text-6xl">
          Blur a real face and the detector grows <span className="text-fake">suspicious</span>.
          Blur a fake one and it <span className="text-real">barely reacts</span>.
        </h1>
        <p className="mt-6 max-w-3xl text-lg leading-relaxed text-slate-300">
          DFD-FCG reports robustness as AUROC, which stays at <b className="text-white">{meanAuc.toFixed(3)}</b> under
          blur, resampling and compression. At its own threshold the same degradations make it call up
          to <b className="text-fake">{pct(o[worst]["fpr@0.5"], 0)}</b> of authentic videos fake. Reading the model's exact
          explanation shows why: much of its evidence for REAL is fine natural detail, so removing detail adds FAKE evidence
          to authentic videos, while the fragile evidence on fakes is lost. The model's own features also say <i>how</i> degraded a face is, and
          a linear read-out of them, subtracted from each head, restores the threshold.
        </p>
        <div className="mt-10 grid gap-4 md:grid-cols-4">
          <Kpi label="AUROC under degradation (released)" value={meanAuc.toFixed(3)} sub="looks robust" tone="accent" />
          <Kpi label={`Real videos called FAKE, ${worst}`} value={pct(o[worst]["fpr@0.5"], 0)} sub="released model, threshold 0.5" tone="fake" />
          <Kpi label="Same videos, self-calibrated" value={pct(ours?.["fpr@0.5"], 0)}
            sub={`fakes caught ${pct(ours?.["tpr@0.5"], 0)} (released ${pct(o[worst]["tpr@0.5"], 0)}); this degradation type was never seen in fitting`} tone="real" />
          <Kpi label="Mixed-quality AUROC" value={`${Number(row("original")?.pooled_auroc ?? 0).toFixed(3)} → ${Number(row(summary.main)?.pooled_auroc ?? 0).toFixed(3)}`}
            sub="all test videos of all conditions ranked together" tone="white" />
        </div>
        {fr && (
          <p className="mt-6 text-sm muted">Response of the fusion logit to a 2-pixel blur on clean video: real {sgn(fr.real_mean[2])},
            fake {sgn(fr.fake_mean[2])}; the direction alone separates them with AUROC {fr.auroc_minus_response_fusion.toFixed(3)}.</p>
        )}
        <div className="mt-8 flex flex-wrap gap-3">
          <a href="#collapse" className="btn btn-on flex items-center gap-2">See the evidence <ArrowDown size={16} /></a>
          <a href={REPO} className="btn flex items-center gap-2" target="_blank" rel="noreferrer"><GitBranch size={16} /> Code &amp; data</a>
        </div>
      </div>
    </header>
  );
}
