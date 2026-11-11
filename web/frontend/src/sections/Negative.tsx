import type { Analysis } from "./Anatomy";
import type { Summary } from "../data";
import { pct } from "../data";
import { Section } from "../components/ui";

export function Negative({ analysis, summary }: { analysis: Analysis; summary: Summary }) {
  const tr = analysis.geometry.transfer_unseen;
  const row = (m: string, reg = "lopo") => summary.table.find((r) => r.method === m && r.regime === reg);
  const cmp = (m: string) => [row(m)?.["tpr@mfpr05"], row(m)?.["tpr@mfpr10"], row(m)?.["tpr@mfpr20"]] as number[];
  const o = cmp("original"), d = cmp("d2d-pB2"), s = cmp(summary.main);
  return (
    <Section id="negative" kicker="What did not work" title="Two ideas that failed, and why"
      lead={<>Both are reported because the reasons explain the result that did work.</>}>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="card space-y-3 text-sm muted">
          <h3 className="text-lg font-semibold text-white">1 · Quality from the background</h3>
          <p>A face manipulation never touches the pixels around the face, so we measured blur, noise, blocking and colour
            there and regressed every exact term's shift on them ({analysis.en?.n_pairs?.toLocaleString()} paired real
            training clips). For a degradation type it was not fitted on, the prediction explains this share (R²) of the
            logit shift on real test videos; “&lt; 0” means worse than predicting the average shift:</p>
          <table className="grid-table">
            <thead><tr><th>unseen type</th><th>spatial</th><th>temporal</th><th>fusion</th></tr></thead>
            <tbody>{Object.keys(tr).map((k) => (
              <tr key={k}><td className="font-mono">{k}</td>{(["s", "t", "a"] as const).map((h) => { const x = tr[k][h].logit_drift_explained; return <td key={h}>{x < 0 ? "< 0" : x.toFixed(2)}</td>; })}</tr>
            ))}</tbody>
          </table>
          <p>Background sharpness varies far more between videos (focus, depth of field, camera) than within a video
            under compression. The model's own features, which see the face, do much better.</p>
        </div>
        <div className="card space-y-3 text-sm muted">
          <h3 className="text-lg font-semibold text-white">2 · The blur probe as the fix</h3>
          <p>On clean video the probe response separates real from fake almost perfectly, so subtracting it looked like a
            free correction. At the 0.5 threshold it does cut false alarms, but at an equal false-alarm budget it detects no
            more fakes than simply moving the released threshold:</p>
          <table className="grid-table">
            <thead><tr><th>detection at mean false alarms ≤ (type unseen)</th><th>5%</th><th>10%</th><th>20%</th></tr></thead>
            <tbody>
              <tr><td>released, threshold moved</td>{o.map((x, i) => <td key={i}>{pct(x)}</td>)}</tr>
              <tr><td>blur probe contrast</td>{d.map((x, i) => <td key={i}>{pct(x)}</td>)}</tr>
              <tr className="bg-real/5"><td className="text-real">self-calibrated</td>{s.map((x, i) => <td key={i}>{pct(x)}</td>)}</tr>
            </tbody>
          </table>
          <p>Once a video is degraded, much of the fine detail is already gone: real videos respond less to the probe and
            fake videos respond more, so the gap that made the probe so informative on clean video shrinks. The probe stays
            useful as an explanation of each verdict and as the optional strict mode.</p>
        </div>
      </div>
    </Section>
  );
}
