import { useState } from "react";
import type { Conditions, Summary } from "../data";
import { METHOD_LABEL, f3, pct } from "../data";
import { Section, Toggle } from "../components/ui";

const NEEDS: Record<string, string> = {
  original: "–", platt: "fakes + reals (clean)", qshift: "reals only", en: "reals only",
  "en-face": "reals only", "en-proj": "reals only", refit: "fakes + reals, all degradations",
};

export function Results({ cond, summary }: { cond: Conditions; summary: Summary }) {
  const [regime, setRegime] = useState<"lopo" | "all">("lopo");
  const rows = summary.table.filter((r) => r.regime === regime || (r.method === "original" && r.regime === "all"));
  return (
    <Section id="results" kicker="Evaluation" title="Every method, same videos, same thresholds"
      lead={<>Corrections are fitted on FF++ <i>train-split</i> videos at severities 2 and 4 and tested on disjoint
        <i> test-split</i> identities at severities 1, 3 and 5. “Unseen type” additionally hides the tested
        degradation type from the fit. τ₅ is the threshold that gives 5% false alarms on clean training reals.</>}>
      <div className="mb-4"><Toggle value={regime} onChange={setRegime} options={[
        { value: "lopo", label: "degradation type unseen" }, { value: "all", label: "all types seen (other severities)" }]} /></div>
      <div className="card overflow-x-auto">
        <table className="grid-table min-w-[900px]">
          <thead>
            <tr>
              <th>method</th><th>fitted with</th><th>clean AUROC</th><th>degraded AUROC</th>
              <th>false alarms @0.5</th><th>worst false alarms</th><th>fakes caught @0.5</th>
              <th>balanced acc @0.5</th><th>false alarms @τ₅</th><th>fakes caught @τ₅</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const m = String(r.method);
              const ours = m === "en";
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
                  <td>{pct(r["deg_mean_fpr@tau5"] as number)}</td>
                  <td>{pct(r["deg_mean_tpr@tau5"] as number)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-4 text-xs muted">Means over the {Object.keys(cond.original.all).length - 2} degraded test
          conditions. Unseen FaceShifter (not in DFD-FCG's training data), clean: AUROC {f3(cond.original.fsh.auroc)} released,{" "}
          {f3(cond.en?.fsh?.auroc)} with Evidence Nulling.</p>
      </div>
    </Section>
  );
}
