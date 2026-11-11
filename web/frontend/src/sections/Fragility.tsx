import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Analysis, DriftLayers } from "./Anatomy";
import { Heat } from "./Anatomy";
import { FAKE_METHODS, pct, sgn } from "../data";
import { Section, Toggle } from "../components/ui";

const PROBE_LABEL: Record<string, string> = { pB2: "2-px blur", pJ50: "JPEG q50" };

export function Fragility({ analysis, layers }: { analysis: Analysis; layers: DriftLayers }) {
  const probes = Object.keys(analysis.fragility ?? {});
  const [pr, setPr] = useState(probes[0] ?? "pB2");
  const F = analysis.fragility?.[pr];
  if (!F) return null;
  const inputs = Object.keys(F.by_input);
  const data = inputs.map((c) => ({ cond: c, real: F.by_input[c].real_mean[2], fake: F.by_input[c].fake_mean[2],
    auc: F.by_input[c].auroc_minus_response_fusion }));
  const groups = ["REAL", "DF", "F2F", "FS", "NT", "FSh"].filter((g) => g in F);
  const per = groups.map((g) => ({ g: g === "REAL" ? "real" : g, s: F[g].mean_response[0], t: F[g].mean_response[1], a: F[g].mean_response[2] }));
  const LR = layers[`frag:${pr}_REAL`], LF = layers[`frag:${pr}_FAKE`];
  const c0 = F.by_input.clean;
  return (
    <Section id="fragility" kicker="Finding 2" title="Real and fake videos move in opposite directions"
      lead={<>We applied one mild, known probe to the aligned face and measured how each head's evidence changed.
        On clean videos the fake evidence collapses ({c0 ? `${pct(c0.frac_fakes_drop, 0)} of fakes drop` : ""}) while
        authentic videos keep or gain FAKE evidence ({c0 ? `${pct(c0.frac_reals_rise, 0)} of reals rise` : ""}). The direction
        of the response alone separates real from fake with AUROC {c0?.auroc_minus_response_fusion.toFixed(3)}. The same
        mechanism explains the false alarms: the model has learned “fine natural detail = real”, and degradation removes it.</>}>
      <div className="mb-4"><Toggle value={pr} onChange={setPr} options={probes.map((p) => ({ value: p, label: PROBE_LABEL[p] ?? p }))} /></div>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="card">
          <div className="mb-3 text-sm muted">Mean change of the fusion logit caused by the probe, per input condition</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={data} margin={{ left: 0, right: 10 }}>
              <CartesianGrid stroke="#1d2742" />
              <XAxis dataKey="cond" stroke="#94a3b8" fontSize={11} angle={-35} textAnchor="end" height={50} />
              <YAxis stroke="#94a3b8" />
              <ReferenceLine y={0} stroke="#64748b" />
              <Tooltip contentStyle={{ background: "#0e1528", border: "1px solid #1d2742" }} formatter={(v) => Number(v).toFixed(3)} />
              <Legend />
              <Bar dataKey="real" name="real videos" fill="#34d399" />
              <Bar dataKey="fake" name="fake videos" fill="#f472b6" />
            </BarChart>
          </ResponsiveContainer>
          <p className="mt-2 text-xs muted">The gap shrinks as the input is already degraded: once the fragile evidence is
            gone, there is less left to collapse. AUROC of the response alone: {data.map((d) => `${d.cond} ${d.auc.toFixed(2)}`).join(" · ")}</p>
        </div>
        <div className="card">
          <div className="mb-3 text-sm muted">Mean response per head on clean videos, by manipulation method</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={per}>
              <CartesianGrid stroke="#1d2742" />
              <XAxis dataKey="g" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <ReferenceLine y={0} stroke="#64748b" />
              <Tooltip contentStyle={{ background: "#0e1528", border: "1px solid #1d2742" }} formatter={(v) => Number(v).toFixed(3)} />
              <Legend />
              <Bar dataKey="s" name="spatial" fill="#38bdf8" />
              <Bar dataKey="t" name="temporal" fill="#a78bfa" />
              <Bar dataKey="a" name="fusion" fill="#fbbf24" />
            </BarChart>
          </ResponsiveContainer>
          <p className="mt-2 text-xs muted">FaceShifter (FSh) was never seen by DFD-FCG in training. {groups.filter((g) => g !== "REAL").map((g) =>
            `${FAKE_METHODS[g] ?? g}: spatial ${sgn(F[g].mean_response[0])}, temporal ${sgn(F[g].mean_response[1])}`).join("; ")}.</p>
        </div>
      </div>
      {LR && LF && (
        <div className="card mt-6 grid gap-6 lg:grid-cols-2">
          <div>
            <div className="mb-2 text-sm muted">Where the probe changes the evidence of <b className="text-real">real</b> videos (exact terms)</div>
            <Heat s={LR.s} t={LR.t} />
          </div>
          <div>
            <div className="mb-2 text-sm muted">… and of <b className="text-fake">fake</b> videos. Green cells: FAKE evidence removed.</div>
            <Heat s={LF.s} t={LF.t} />
          </div>
        </div>
      )}
    </Section>
  );
}
