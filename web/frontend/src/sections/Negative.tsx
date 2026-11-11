import type { Analysis } from "./Anatomy";
import type { Conditions } from "../data";
import { pct } from "../data";
import { Section } from "../components/ui";

export function Negative({ analysis, cond }: { analysis: Analysis; cond: Conditions }) {
  const tr = analysis.geometry.transfer_unseen;
  const deg = Object.keys(cond.original.all).filter((c) => c !== "clean" && c !== "RE_0");
  const mean = (m: string, k: "fpr@0.5" | "tpr@0.5") => cond[m] ? deg.reduce((s, c) => s + (cond[m].lopo[c]?.[k] ?? 0), 0) / deg.length : NaN;
  const g = analysis.geometry as unknown as Record<string, { cos_mean_drift_vs_forgery: number }>;
  return (
    <Section id="negative" kicker="What did not work" title="Correcting with an image-quality estimate fails"
      lead={<>Our first idea was to estimate image quality from the pixels around the face (a face manipulation never
        changes them) and subtract the evidence that quality predicts. It does not work, and the reason is informative.</>}>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="card text-sm muted space-y-3">
          <p>Eight blur, noise, blocking and colour descriptors of the background, regressed on the paired change of
            every exact term of {analysis.en?.n_pairs?.toLocaleString()} real training clips. On real test videos the
            prediction explains only this share of the logit change of a degradation type it was not fitted on (R²):</p>
          <table className="grid-table">
            <thead><tr><th>unseen type</th><th>spatial</th><th>temporal</th><th>fusion</th></tr></thead>
            <tbody>{Object.keys(tr).map((k) => (
              <tr key={k}><td className="font-mono">{k}</td>{(["s", "t", "a"] as const).map((h) => <td key={h}>{tr[k][h].logit_drift_explained.toFixed(2)}</td>)}</tr>
            ))}</tbody>
          </table>
          <p>Background sharpness varies far more between videos (focus, depth of field, camera) than within a video under
            compression, so it does not tell how much detail the face has lost.</p>
        </div>
        <div className="card text-sm muted space-y-3">
          <p>Removing the degradation direction from the attribution itself also fails, because degradation and
            forgery push the terms in similar directions: cosine between the mean degradation drift and the mean forgery
            direction is spatial {g.s?.cos_mean_drift_vs_forgery?.toFixed(2)}, temporal {g.t?.cos_mean_drift_vs_forgery?.toFixed(2)},
            fusion {g.a?.cos_mean_drift_vs_forgery?.toFixed(2)}.</p>
          <p>Averaged over degraded test conditions (type unseen): background-quality nulling gives false alarms{" "}
            {pct(mean("en", "fpr@0.5"))} and catches {pct(mean("en", "tpr@0.5"))} of fakes, against{" "}
            {pct(mean("original", "fpr@0.5"))} / {pct(mean("original", "tpr@0.5"))} for the released model.</p>
          <p className="text-slate-300">What does separate the two is not <i>how degraded</i> a video is but <i>how its
            evidence reacts</i> to further degradation. That is the idea behind Degrade-to-Detect.</p>
        </div>
      </div>
    </Section>
  );
}
