import type { Analysis } from "./Anatomy";
import type { Conditions, Summary } from "../data";
import { pct } from "../data";
import { Section } from "../components/ui";

export function Method({ analysis, summary, cond }: { analysis: Analysis; summary: Summary; cond: Conditions }) {
  const tr = analysis.geometry.transfer_unseen;
  const g = analysis.geometry as unknown as Record<string, { cos_mean_drift_vs_forgery: number; drift_energy_top: number[] }>;
  const proj = cond["en-proj"];
  const enf = cond["en-face"];
  const deg = Object.keys(cond.original.all).filter((c) => c !== "clean" && c !== "RE_0");
  const mean = (m: Conditions[string] | undefined, k: "fpr@0.5" | "tpr@0.5") =>
    m ? deg.reduce((s, c) => s + (m.lopo[c]?.[k] ?? 0), 0) / deg.length : null;
  return (
    <Section id="method" kicker="Contribution" title="Evidence Nulling: subtract what the video's quality adds"
      lead={<>Every head logit of DFD-FCG is an exact sum of terms. Degradation shifts those terms, and so does a
        forgery, often in the <i>same</i> direction, so the attribution alone cannot tell them apart. The pixels
        around the face can: a face manipulation never touches them, but blur, resampling and compression do. We learn,
        from authentic videos only, how background quality predicts every term's shift, and remove that prediction.</>}>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="card space-y-4">
          <h3 className="text-lg font-semibold text-white">1 · Exact attribution (our earlier work)</h3>
          <div className="eq">d_h = Σ_k c_h[k] + const_h,&nbsp;&nbsp; P = ⅓ Σ_h σ(d_h),&nbsp;&nbsp; h ∈ {"{spatial, temporal, fusion}"}</div>
          <p className="text-sm muted">96 spatial terms (24 CLIP layers × 4 facial-component queries), 24 temporal, 120
            fusion. The split reproduces the model's logits to 2·10⁻⁷.</p>
          <h3 className="pt-2 text-lg font-semibold text-white">2 · A forgery-free quality reference</h3>
          <p className="text-sm muted">Eight blur, noise, blocking, sharpness and colour descriptors measured on the pixels
            outside a dilated face hull (about 60% of the region around the face). A real video and its FF++ fake share
            these pixels, so the descriptors carry quality and no forgery evidence by construction.</p>
          <h3 className="pt-2 text-lg font-semibold text-white">3 · Learn the quality → evidence map from real videos</h3>
          <div className="eq">c_h(degraded) − c_h(clean) ≈ W_h [ φ(q_bg,degraded) − φ(q_bg,clean) ]</div>
          <p className="text-sm muted">Ridge regression on {summary.en_pairs.toLocaleString()} paired clips of{" "}
            {summary.n.fit_real} real FF++ <i>train-split</i> videos, 7 degradation types at severities 2 and 4.</p>
          <h3 className="pt-2 text-lg font-semibold text-white">4 · Null the predicted evidence at test time</h3>
          <div className="eq">c′_h = c_h − W_h [ φ(q_bg) − φ̄_clean-real ],&nbsp;&nbsp; d′_h = Σ_k c′_h[k] + const_h</div>
          <p className="text-sm muted">No fake video, no retraining, no clean copy of the test video. The removed amount is
            reported per verdict as its <b>quality share</b>, a new kind of explanation: how much of “FAKE” came from the
            video's quality, and from which layer and facial component.</p>
        </div>
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-lg font-semibold text-white">Why the background is needed</h3>
            <p className="mt-2 text-sm muted">Cosine between the mean degradation drift and the mean forgery direction in
              term space: spatial {g.s?.cos_mean_drift_vs_forgery?.toFixed(2)}, temporal{" "}
              {g.t?.cos_mean_drift_vs_forgery?.toFixed(2)}, fusion {g.a?.cos_mean_drift_vs_forgery?.toFixed(2)}.
              Removing the drift direction from the attribution itself (ablation) also removes forgery evidence:</p>
            <table className="grid-table mt-3">
              <thead><tr><th>variant (degradation type unseen)</th><th>false alarms</th><th>fakes caught</th></tr></thead>
              <tbody>
                <tr><td>attribution projection only</td><td>{pct(mean(proj, "fpr@0.5"))}</td><td>{pct(mean(proj, "tpr@0.5"))}</td></tr>
                <tr><td>quality from the face crop</td><td>{pct(mean(enf, "fpr@0.5"))}</td><td>{pct(mean(enf, "tpr@0.5"))}</td></tr>
                <tr className="bg-real/5"><td className="text-real">quality from the background (ours)</td><td>{pct(mean(cond.en, "fpr@0.5"))}</td><td>{pct(mean(cond.en, "tpr@0.5"))}</td></tr>
                <tr><td>released model</td><td>{pct(mean(cond.original, "fpr@0.5"))}</td><td>{pct(mean(cond.original, "tpr@0.5"))}</td></tr>
              </tbody>
            </table>
          </div>
          <div className="card">
            <h3 className="text-lg font-semibold text-white">It transfers to unseen degradations</h3>
            <p className="mt-2 text-sm muted">Fitted without one degradation type, the correction's prediction of that type's
              logit shift on real test videos (R², paired):</p>
            <table className="grid-table mt-3">
              <thead><tr><th>unseen type</th><th>spatial</th><th>temporal</th><th>fusion</th><th>mean fusion shift → after</th></tr></thead>
              <tbody>
                {Object.keys(tr).map((k) => (
                  <tr key={k}><td className="font-mono">{k}</td>
                    {(["s", "t", "a"] as const).map((h) => <td key={h}>{tr[k][h].logit_drift_explained.toFixed(2)}</td>)}
                    <td className="font-mono">{tr[k].a.mean_logit_drift.toFixed(2)} → {tr[k].a.mean_logit_drift_after.toFixed(2)}</td></tr>
                ))}
              </tbody>
            </table>
            {analysis.counterfactual.n_false_alarms > 0 && (
              <p className="mt-4 text-sm muted">Of the {analysis.counterfactual.n_false_alarms} false alarms on degraded real
                test videos, Evidence Nulling reverses <b className="text-real">{analysis.counterfactual.false_alarms_fixed}</b>.</p>
            )}
          </div>
        </div>
      </div>
    </Section>
  );
}
