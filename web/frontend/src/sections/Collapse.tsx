import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Conditions, Summary } from "../data";
import { METHOD_COLOR, METHOD_LABEL, pct } from "../data";
import { Section, Toggle } from "../components/ui";

const ORDER = ["clean", "RE_0", "CS_3", "CC_3", "BW_3", "GNC_3", "GB_1", "GB_3", "GB_5", "JPEG_1", "JPEG_3", "JPEG_5", "VC_1", "VC_3", "VC_5"];

export function Collapse({ cond, summary }: { cond: Conditions; summary: Summary }) {
  const [regime, setRegime] = useState<"lopo" | "all">("lopo");
  const conds = ORDER.filter((c) => c in cond.original.all);
  const ours = cond[summary.main];
  const data = useMemo(() => conds.map((c) => ({
    cond: c,
    auroc: cond.original.all[c].auroc,
    fpr: cond.original.all[c]["fpr@0.5"],
    fpr_null: ours?.[regime]?.[c]?.["fpr@0.5"],
    tpr: cond.original.all[c]["tpr@0.5"],
    tpr_null: ours?.[regime]?.[c]?.["tpr@0.5"],
  })), [cond, conds, regime, ours]);
  const methods = ["platt", "qshift", "d2d-pB2", "selfcal", "selfcal+d2d"].filter((m) => m in cond);
  const bars = conds.filter((c) => c !== "clean").map((c) => {
    const row: Record<string, number | string> = { cond: c };
    for (const m of ["original", ...methods]) row[m] = (m === "original" ? cond[m].all : cond[m][regime])?.[c]?.["fpr@0.5"] ?? 0;
    return row;
  });
  return (
    <Section id="collapse" kicker="Finding 1" title="AUROC hides a calibration collapse"
      lead={<>The paper measures robustness with AUROC, which only asks whether fakes still score above reals. A deployed
        detector uses a threshold. Below, the released model is evaluated on {summary.n.test_real} real and{" "}
        {summary.n.test_fake} fake FF++ <i>test</i> videos under the authors' own perturbation protocol: AUROC barely
        moves while the share of real videos called FAKE climbs with every blur, resampling or compression step.</>}>
      <div className="card">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div className="text-sm muted">Released model (pink/blue) and self-calibrated (green), threshold 0.5</div>
          <Toggle value={regime} onChange={setRegime} options={[
            { value: "lopo", label: "rule never saw this degradation type" },
            { value: "all", label: "rule fitted on all types (other severities)" }]} />
        </div>
        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={data} margin={{ left: 0, right: 20, top: 10, bottom: 10 }}>
            <CartesianGrid stroke="#1d2742" />
            <XAxis dataKey="cond" stroke="#94a3b8" fontSize={12} />
            <YAxis domain={[0, 1]} stroke="#94a3b8" tickFormatter={(v) => `${Math.round(v * 100)}%`} />
            <Tooltip contentStyle={{ background: "#0e1528", border: "1px solid #1d2742" }}
              formatter={(v) => pct(Number(v))} />
            <Legend />
            <ReferenceLine y={0.05} stroke="#fbbf24" strokeDasharray="4 4" />
            <Line isAnimationActive={false} dataKey="auroc" name="AUROC (released)" stroke="#38bdf8" strokeWidth={2.5} dot />
            <Line isAnimationActive={false} dataKey="fpr" name="Real videos called FAKE (released)" stroke="#f472b6" strokeWidth={2.5} dot />
            <Line isAnimationActive={false} dataKey="fpr_null" name="Real videos called FAKE (self-calibrated)" stroke="#34d399" strokeWidth={2.5} dot />
            <Line isAnimationActive={false} dataKey="tpr_null" name="Fakes caught (self-calibrated)" stroke="#34d399" strokeDasharray="5 4" dot={false} />
            <Line isAnimationActive={false} dataKey="tpr" name="Fakes caught (released)" stroke="#f472b6" strokeDasharray="5 4" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="card mt-6">
        <div className="mb-3 text-sm muted">False-alarm rate per condition, every method ({regime === "lopo" ? "degradation type held out" : "all types, other severities"})</div>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={bars} margin={{ left: 0, right: 20, top: 10, bottom: 10 }}>
            <CartesianGrid stroke="#1d2742" />
            <XAxis dataKey="cond" stroke="#94a3b8" fontSize={12} />
            <YAxis domain={[0, 1]} stroke="#94a3b8" tickFormatter={(v) => `${Math.round(v * 100)}%`} />
            <Tooltip contentStyle={{ background: "#0e1528", border: "1px solid #1d2742" }} formatter={(v) => pct(Number(v))} />
            <Legend />
            {["original", ...methods].map((m) => (
              <Bar isAnimationActive={false} key={m} dataKey={m} name={METHOD_LABEL[m] ?? m} fill={METHOD_COLOR[m] ?? "#888"} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Section>
  );
}
