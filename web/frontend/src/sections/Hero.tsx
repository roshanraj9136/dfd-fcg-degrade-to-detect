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
  return (
    <header className="glow relative overflow-hidden border-b border-edge">
      <div className="mx-auto max-w-6xl px-5 pb-16 pt-24">
        <div className="flex flex-wrap gap-2">
          <span className="chip">Extension of DFD-FCG · CVPR 2025</span>
          <span className="chip">{summary.n.test_real} real + {summary.n.test_fake} fake FF++ test videos</span>
          <span className="chip">B.Tech project · Roshan Raj</span>
        </div>
        <h1 className="mt-6 max-w-5xl text-4xl font-semibold leading-tight tracking-tight text-white md:text-6xl">
          Blur a real face and the detector gets <span className="text-fake">suspicious</span>.
          Blur a fake one and it <span className="text-real">relaxes</span>.
        </h1>
        <p className="mt-6 max-w-3xl text-lg leading-relaxed text-slate-300">
          DFD-FCG reports robustness as AUROC, which stays at <b className="text-white">{meanAuc.toFixed(3)}</b> under
          blur, resampling and compression. At its own threshold the same degradations make it call up
          to <b className="text-fake">{pct(o[worst]["fpr@0.5"], 0)}</b> of authentic videos fake. Using the model's
          exact explanation we found why: its FAKE evidence is fragile and its REAL evidence is fine detail, so
          degradation pushes both classes towards each other. We turn that into a fix, <b className="text-real">Degrade-to-Detect</b>:
          blur the face once more yourself and watch which way the evidence moves.
        </p>
        <div className="mt-10 grid gap-4 md:grid-cols-4">
          <Kpi label="AUROC under degradation (released)" value={meanAuc.toFixed(3)} sub="looks robust" tone="accent" />
          <Kpi label={`Real videos called FAKE, ${worst}`} value={pct(o[worst]["fpr@0.5"], 0)} sub="released model, threshold 0.5" tone="fake" />
          <Kpi label="Same videos, Degrade-to-Detect" value={pct(ours?.["fpr@0.5"], 0)}
            sub={`fakes caught ${pct(ours?.["tpr@0.5"], 0)} (released ${pct(o[worst]["tpr@0.5"], 0)}); this degradation type was never seen`} tone="real" />
          <Kpi label="Response to a 2-px blur, clean videos" value={fr ? `${sgn(fr.real_mean[2])} / ${sgn(fr.fake_mean[2])}` : "–"}
            sub={fr ? `fusion logit, real / fake; separates them with AUROC ${fr.auroc_minus_response_fusion.toFixed(3)} on its own` : ""} />
        </div>
        <div className="mt-10 flex flex-wrap gap-3">
          <a href="#collapse" className="btn btn-on flex items-center gap-2">See the evidence <ArrowDown size={16} /></a>
          <a href={REPO} className="btn flex items-center gap-2" target="_blank" rel="noreferrer"><GitBranch size={16} /> Code &amp; data</a>
        </div>
      </div>
    </header>
  );
}
