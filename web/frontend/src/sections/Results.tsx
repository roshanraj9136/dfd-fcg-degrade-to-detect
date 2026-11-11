import { useState } from "react";
import type { Conditions, Summary } from "../data";
import { METHOD_LABEL, NEEDS, OURS, f3, pct } from "../data";
import { Section, Toggle } from "../components/ui";

export function Results({ cond, summary }: { cond: Conditions; summary: Summary }) {
  const [regime, setRegime] = useState<"lopo" | "all">("lopo");
  const rows = summary.table.filter((r) => r.regime === regime || (r.method === "original" && r.regime === "all"));
  const pooled = Object.fromEntries(summary.table.filter((r) => r.regime === "all").map((r) => [r.method, r]));
  return (
    <Section id="results" kicker="Evaluation" title="Every method, same videos, same threshold"
      lead={<>Rules are fitted on FF++ <i>train-split</i> videos at severities 2 and 4 and tested on the disjoint
        <i> test split</i> at severities 1, 3 and 5. “Degradation type unseen” additionally hides the tested type
        from fitting. τ₅ is the threshold giving 5% false alarms on clean training reals. “Mixed pool” ranks all test
        videos of all conditions together, as a real deployment would see them.</>}>
      <div className="mb-4"><Toggle value={regime} onChange={setRegime} options={[
        { value: "lopo", label: "degradation type unseen" }, { value: "all", label: "all types seen (other severities)" }]} /></div>
      <div className="card overflow-x-auto">
        <table className="grid-table min-w-[1000px]">
          <thead>
            <tr>
              <th>method</th><th>fitted with</th><th>clean AUROC</th><th>degraded AUROC</th>
              <th>false alarms @0.5</th><th>worst</th><th>fakes caught @0.5</th>
              <th>balanced acc @0.5</th><th>mixed-pool AUROC</th><th>false alarms @τ₅</th><th>fakes caught @τ₅</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const m = String(r.method);
              const ours = OURS.has(m);
              return (
                <tr key={m + r.regime} className={ours ? "bg-real/5" : ""}>
                  <td className={ours ? "font-semibold text-real" : "text-white"}>{METHOD_LABEL[m] ?? m}</td>
                  <td className="muted">{NEEDS[m]}</td>
                  <td>{f3(r.clean_auroc as number)}</td>
                  <td>{f3(r.deg_mean_auroc as number)}</td>
                  <td>{pct(r["deg_mean_fpr@0.5"] as number)}</td>
                  <td>{pct(r["deg_worst_fpr@0.5"] as number)}</td>
                  <td>{pct(r["deg_mean_tpr@0.5"] as number)}</td>
                  <td>{pct(r["deg_mean_bacc@0.5"] as number)}</td>
                  <td>{f3(pooled[m]?.pooled_auroc as number)}</td>
                  <td>{pct(r["deg_mean_fpr@tau5"] as number)}</td>
                  <td>{pct(r["deg_mean_tpr@tau5"] as number)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-4 text-xs muted">Means over the degraded test conditions. Unseen FaceShifter (clean):
          AUROC {f3(cond.original.fsh.auroc)} released, {f3(cond[summary.main]?.fsh?.auroc)} Degrade-to-Detect;
          fakes caught {pct(cond.original.fsh["tpr@0.5"])} vs {pct(cond[summary.main]?.fsh?.["tpr@0.5"])}.</p>
      </div>
    </Section>
  );
}
