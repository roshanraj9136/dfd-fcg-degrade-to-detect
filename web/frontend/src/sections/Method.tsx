import type { Conditions, Summary } from "../data";
import { f3, pct } from "../data";
import { Section } from "../components/ui";

export function Method({ summary, cond }: { summary: Summary; cond: Conditions }) {
  const main = cond[summary.main];
  const strict = cond[summary.strict];
  const row = (m: string, reg = "all") => summary.table.find((r) => r.method === m && r.regime === reg);
  const targets = Object.entries(summary.drift_targets ?? {}).filter(([c]) => c !== "clean").sort((a, b) => b[1][2] - a[1][2]);
  return (
    <Section id="method" kicker="Contribution" title="Self-calibrated DFD-FCG: let the model measure its own degradation"
      lead={<>The two LayerNorm heads of DFD-FCG read a 1024-d spatial and a 256-d temporal feature built from the frozen
        CLIP encoder. Those same features encode blur, resampling and compression. We fit one linear read-out that
        predicts, from them, how much spurious FAKE evidence the video's degradation has added to each head, and
        subtract it. The released checkpoint is untouched.</>}>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="card space-y-4">
          <h3 className="text-lg font-semibold text-white">The rule</h3>
          <div className="eq">ĝ = W · standardise([y_s, y_t]) + b&nbsp;&nbsp;&nbsp; d′_h = d_h − ĝ_h&nbsp;&nbsp;&nbsp; P = ⅓ Σ_h σ(d′_h)</div>
          <p className="text-sm muted">d_h is the logit difference (FAKE − REAL) of the spatial, temporal and fusion heads,
            P is the paper's own fusion, so the threshold stays at 0.5.</p>
          <h3 className="pt-2 text-lg font-semibold text-white">What it is trained to predict</h3>
          <p className="text-sm muted">For a training clip under condition c, the target is the average shift that c
            causes on <i>real</i> training videos (paired: degraded minus clean), 0 for clean, and it is the same target
            for real and fake clips. The read-out therefore learns “which degradation is this, how strong”, never “is this
            fake”. Ridge regression, α = {summary.alpha}, fitted on {summary.n.fit_real} real and {summary.n.fit_fake} fake
            FF++ <i>train-split</i> videos at severities 2 and 4 (tests use 1, 3, 5).</p>
          <table className="grid-table">
            <thead><tr><th>condition (training)</th><th>spatial</th><th>temporal</th><th>fusion</th></tr></thead>
            <tbody>{targets.slice(0, 8).map(([c, v]) => (
              <tr key={c}><td className="font-mono">{c}</td>{v.map((x, i) => <td key={i}>{x >= 0 ? "+" : ""}{x.toFixed(2)}</td>)}</tr>
            ))}</tbody>
          </table>
          <p className="text-xs muted">Largest targets: the spurious evidence each condition adds to an authentic video, in logits.</p>
        </div>
        <div className="space-y-6">
          <div className="card space-y-3">
            <h3 className="text-lg font-semibold text-white">Strict mode: add the blur probe</h3>
            <div className="eq">d″_h = d′_h − {summary.strict_lam} · ( d_h(blur(x)) − d_h(x) )</div>
            <p className="text-sm muted">Finding 2 as a decision aid: one extra forward pass on the face blurred by 2 px. It
              lowers false alarms further at a cost in detection, so it is an operating mode, not a free gain.
              {strict && row(summary.strict) && <> Degraded conditions: false alarms {pct(row(summary.strict)?.["deg_mean_fpr@0.5"] as number)},
                fakes caught {pct(row(summary.strict)?.["deg_mean_tpr@0.5"] as number)}.</>}</p>
          </div>
          <div className="card space-y-3">
            <h3 className="text-lg font-semibold text-white">Why it is an extension of DFD-FCG</h3>
            <ul className="list-disc space-y-2 pl-5 text-sm muted">
              <li>It keeps the released checkpoint, its three heads and its fusion rule.</li>
              <li>It was found through the exact additive explanation of those heads (our earlier work), which showed
                where degradation enters the decision.</li>
              <li>It answers the gap in the paper's robustness study, which reports AUROC only: the threshold becomes usable
                under degradation, not only the ranking.</li>
              <li>Every verdict now comes with an estimate of how much of it is due to video quality.</li>
            </ul>
            {main && (
              <div className="rounded-xl border border-real/30 bg-real/5 p-4 text-sm">
                Clean test videos: AUROC {f3(main.all.clean.auroc)} (released {f3(cond.original.all.clean.auroc)}),
                false alarms {pct(main.all.clean["fpr@0.5"])}, fakes caught {pct(main.all.clean["tpr@0.5"])}.
                {main.fsh?.auroc != null && <> FaceShifter, never seen by DFD-FCG: AUROC {f3(main.fsh.auroc)} (released {f3(cond.original.fsh.auroc)}).</>}
                {main.dfdc?.auroc != null && <> DFDC, another dataset: AUROC {f3(main.dfdc.auroc)} (released {f3(cond.original.dfdc?.auroc)}).</>}
              </div>
            )}
          </div>
        </div>
      </div>
    </Section>
  );
}
