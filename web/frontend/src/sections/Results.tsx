import { useState } from "react";
import type { Conditions, Summary } from "../data";
import { METHOD_LABEL, NEEDS, OURS, f3, pct } from "../data";
import { Section, Toggle } from "../components/ui";

type Regime = "lopo" | "all" | "cv_lopo" | "cv_all";
const REGIMES: { value: Regime; label: string }[] = [
  { value: "lopo", label: "train split → test, degradation type unseen" },
  { value: "all", label: "train split → test, all types (other severities)" },
  { value: "cv_lopo", label: "5-fold CV on test pairs, type unseen" },
  { value: "cv_all", label: "5-fold CV on test pairs" },
];

export function Results({ cond, summary }: { cond: Conditions; summary: Summary }) {
  const [regime, setRegime] = useState<Regime>("lopo");
  const rows = summary.table.filter((r) => r.regime === regime && r.method !== "d2d-pJ50");   // its lambda was selected as 0 (= released)
  const ref = Object.fromEntries(summary.table.filter((r) => r.regime === (regime.startsWith("cv") ? "cv_all" : "all"))
    .map((r) => [r.method, r]));
  return (
    <Section id="results" kicker="Evaluation" title="Every method, same videos, same threshold"
      lead={<>Rules are fitted on FF++ <i>train-split</i> videos at severities 2 and 4 and tested on the disjoint
        <i> test split</i> at severities 1, 3 and 5; a second protocol cross-validates over test identity pairs (videos
        DFD-FCG never saw). “Type unseen” hides the tested degradation type from fitting. The last three columns use
        one global threshold swept on the test data: the best mean detection rate at a mean false-alarm budget of
        5, 10 or 20%, which compares methods independently of where they put the threshold.</>}>
      <div className="mb-4"><Toggle value={regime} onChange={setRegime} options={REGIMES} /></div>
      <div className="card overflow-x-auto">
        <table className="grid-table min-w-[1100px]">
          <thead>
            <tr>
              <th>method</th><th>fitted with</th><th>clean AUROC</th><th>degraded AUROC</th>
              <th>false alarms @0.5</th><th>worst</th><th>fakes caught @0.5</th><th>balanced acc @0.5</th>
              <th>mixed-pool AUROC</th><th>detect @5% FA</th><th>@10%</th><th>@20%</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const m = String(r.method);
              const ours = OURS.has(m);
              return (
                <tr key={m + r.regime} className={ours ? "bg-real/5" : ""}>
                  <td className={ours ? "font-semibold text-real" : "text-white"}>{METHOD_LABEL[m] ?? m}</td>
                  <td className="max-w-[220px] text-xs muted">{NEEDS[m]}</td>
                  <td>{f3(r.clean_auroc as number)}</td>
                  <td>{f3(r.deg_mean_auroc as number)}</td>
                  <td>{pct(r["deg_mean_fpr@0.5"] as number)}</td>
                  <td>{pct(r["deg_worst_fpr@0.5"] as number)}</td>
                  <td>{pct(r["deg_mean_tpr@0.5"] as number)}</td>
                  <td>{pct(r["deg_mean_bacc@0.5"] as number)}</td>
                  <td>{f3(ref[m]?.pooled_auroc as number)}</td>
                  <td>{pct(r["tpr@mfpr05"] as number)}</td>
                  <td>{pct(r["tpr@mfpr10"] as number)}</td>
                  <td>{pct(r["tpr@mfpr20"] as number)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-4 text-xs muted">Means over the degraded test conditions. Unseen manipulation FaceShifter (clean):
          AUROC {f3(cond.original.fsh?.auroc)} released, {f3(cond[summary.main]?.fsh?.auroc)} self-calibrated.
          {cond.original.dfdc && <> Unseen dataset DFDC ({summary.n.dfdc_real} real + {summary.n.dfdc_fake} fake, clean):
            AUROC {f3(cond.original.dfdc.auroc)} released, {f3(cond[summary.main]?.dfdc?.auroc)} self-calibrated;
            balanced accuracy {pct(cond.original.dfdc["bacc@0.5"])} vs {pct(cond[summary.main]?.dfdc?.["bacc@0.5"])}.</>}</p>
      </div>
    </Section>
  );
}
