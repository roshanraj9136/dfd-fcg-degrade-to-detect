import { ArrowDown, GitBranch } from "lucide-react";
import type { Conditions, Summary } from "../data";
import { pct } from "../data";
import { Kpi } from "../components/ui";

const REPO = "https://github.com/roshanraj9136/dfd-fcg-evidence-nulling";

export function Hero({ cond, summary }: { cond: Conditions; summary: Summary }) {
  const o = cond.original.all;
  const deg = Object.keys(o).filter((c) => c !== "clean" && c !== "RE_0");
  const worst = deg.reduce((a, b) => (o[b]["fpr@0.5"] > o[a]["fpr@0.5"] ? b : a), deg[0]);
  const meanAuc = deg.reduce((s, c) => s + (o[c].auroc ?? 0), 0) / deg.length;
  const ours = cond.en?.lopo?.[worst] ?? cond.en?.all?.[worst];
  return (
    <header className="glow relative overflow-hidden border-b border-edge">
      <div className="mx-auto max-w-6xl px-5 pb-16 pt-24">
        <div className="flex flex-wrap gap-2">
          <span className="chip">Extension of DFD-FCG · CVPR 2025</span>
          <span className="chip">{summary.n.test_real} real + {summary.n.test_fake} fake FF++ test videos</span>
          <span className="chip">B.Tech project · Roshan Raj</span>
        </div>
        <h1 className="mt-6 max-w-4xl text-4xl font-semibold leading-tight tracking-tight text-white md:text-6xl">
          Fake, or just <span className="text-accent">compressed</span>?
        </h1>
        <p className="mt-6 max-w-3xl text-lg leading-relaxed text-slate-300">
          DFD-FCG's robustness is reported as AUROC, and under blur, resampling and H.264 its AUROC stays at{" "}
          <b className="text-white">{meanAuc.toFixed(3)}</b> on average. At the model's own decision threshold the same
          degradations make it call up to <b className="text-fake">{pct(o[worst]["fpr@0.5"], 0)}</b> of{" "}
          <i>authentic</i> videos fake. Using the model's exact additive explanation we locate the false evidence and
          remove it with <b className="text-real">Evidence Nulling</b>, a correction learned from real videos only.
        </p>
        <div className="mt-10 grid gap-4 md:grid-cols-4">
          <Kpi label="AUROC, degraded (released)" value={meanAuc.toFixed(3)} sub="looks robust" tone="accent" />
          <Kpi label={`False alarms, ${summary.conditions[worst] ?? worst}`} value={pct(o[worst]["fpr@0.5"], 0)}
               sub={`real videos called FAKE (${worst})`} tone="fake" />
          <Kpi label="Same condition, Evidence Nulling" value={pct(ours?.["fpr@0.5"], 0)}
               sub="degradation type unseen during fitting" tone="real" />
          <Kpi label="Fakes still caught there" value={pct(ours?.["tpr@0.5"], 0)}
               sub={`released model: ${pct(o[worst]["tpr@0.5"], 0)}`} />
        </div>
        <div className="mt-10 flex flex-wrap gap-3">
          <a href="#collapse" className="btn btn-on flex items-center gap-2">See the evidence <ArrowDown size={16} /></a>
          <a href={REPO} className="btn flex items-center gap-2" target="_blank" rel="noreferrer"><GitBranch size={16} /> Code &amp; data</a>
        </div>
      </div>
    </header>
  );
}
