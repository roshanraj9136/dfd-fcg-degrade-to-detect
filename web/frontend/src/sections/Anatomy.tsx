import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell } from "recharts";
import { Section, Toggle } from "../components/ui";

export type DriftLayers = Record<string, { s: number[][]; t: number[] }>;
export type Analysis = {
  replication: Record<string, Record<string, unknown>>;
  drift: Record<string, {
    delta_d: Record<string, number>; delta_p: number;
    delta_spatial_query: Record<string, number>; delta_spatial_region: Record<string, number>;
    delta_temporal_region: Record<string, number>; shallow_vs_deep_temporal: number[]; shallow_vs_deep_spatial: number[];
  }>;
  geometry: Record<string, unknown> & {
    transfer_unseen: Record<string, Record<string, { energy_captured: number; logit_drift_explained: number; mean_logit_drift: number; mean_logit_drift_after: number }>>;
  };
  counterfactual: { n_false_alarms: number; false_alarms_fixed: number; mean_nuisance_share_in_false_alarms: number | null };
  fingerprints: { n_videos: number; classes: string[]; chance: number; accuracy_explanation: number; accuracy_logits_only: number; confusion: number[][] };
  fragility: Record<string, Fragility>;
  en: { n_pairs: number; descriptors: string[] };
};
export type FragGroup = {
  n_clips: number; mean_response: number[]; query: Record<string, number>; region_spatial: Record<string, number>;
  region_temporal: Record<string, number>; shallow_deep_spatial: number[]; shallow_deep_temporal: number[];
};
export type Fragility = Record<string, FragGroup> & {
  by_input: Record<string, { real_mean: number[]; fake_mean: number[]; auroc_minus_response_fusion: number;
    auroc_minus_response_mean: number; frac_fakes_drop: number; frac_reals_rise: number }>;
};

const PARTS = ["lips", "skin", "eyes", "nose"];

export function Heat({ s, t }: { s: number[][]; t: number[] }) {
  const all = [...s.flat(), ...t].map(Math.abs);
  const m = Math.max(1e-6, ...all);
  const cell = (v: number) => {
    const a = Math.min(1, Math.abs(v) / m);
    return v >= 0 ? `rgba(244,114,182,${a})` : `rgba(52,211,153,${a})`;
  };
  return (
    <div className="overflow-x-auto">
      <table className="text-[10px]">
        <thead>
          <tr><th className="pr-2 text-right font-normal muted">layer</th>
            {s.map((_, l) => <th key={l} className="w-6 font-normal muted">{l + 1}</th>)}</tr>
        </thead>
        <tbody>
          {PARTS.map((p, q) => (
            <tr key={p}><td className="pr-2 text-right muted">spatial · {p}</td>
              {s.map((row, l) => <td key={l} title={row[q].toFixed(4)} style={{ background: cell(row[q]) }} className="h-6 w-6 border border-ink" />)}</tr>
          ))}
          <tr><td className="pr-2 text-right muted">temporal</td>
            {t.map((v, l) => <td key={l} title={v.toFixed(4)} style={{ background: cell(v) }} className="h-6 w-6 border border-ink" />)}</tr>
        </tbody>
      </table>
    </div>
  );
}

export function Anatomy({ analysis, layers }: { analysis: Analysis; layers: DriftLayers }) {
  const opts = ["GB_5", "JPEG_5", "VC_5", "GB_3", "JPEG_3", "VC_3", "CS_3", "GNC_3", "RE_0"].filter((c) => c in analysis.drift);
  const [c, setC] = useState(opts[0]);
  const d = analysis.drift[c];
  const L = layers[c];
  const regions = Object.keys(d.delta_spatial_region).map((r) => ({
    region: r, spatial: d.delta_spatial_region[r], temporal: d.delta_temporal_region[r],
  }));
  const heads = Object.entries(d.delta_d).map(([k, v]) => ({ head: k, v }));
  return (
    <Section id="anatomy" kicker="Finding 3" title="Where the false evidence enters the model"
      lead={<>DFD-FCG's decision is an exact sum of per-layer, per-component and per-patch contributions (our earlier
        work). Comparing each real test video with its own degraded copy shows exactly which terms the degradation
        changes. Pink cells add FAKE evidence to an authentic video.</>}>
      <div className="mb-4"><Toggle value={c} onChange={setC} options={opts.map((o) => ({ value: o, label: o }))} /></div>
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="card lg:col-span-2">
          <div className="mb-3 text-sm muted">Mean change of each exact term (degraded − clean), real test videos</div>
          {L ? <Heat s={L.s} t={L.t} /> : <div className="muted">no layer data</div>}
          <p className="mt-4 text-sm muted">
            Mean change of P(FAKE): <b className="text-white">{d.delta_p >= 0 ? "+" : ""}{(100 * d.delta_p).toFixed(1)} points</b>.
            Temporal layers 1–12 vs 13–24: {d.shallow_vs_deep_temporal.map((x) => x.toFixed(3)).join(" vs ")};
            spatial: {d.shallow_vs_deep_spatial.map((x) => x.toFixed(3)).join(" vs ")}.
          </p>
        </div>
        <div className="card">
          <div className="mb-3 text-sm muted">Change of each head's logit</div>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={heads}>
              <CartesianGrid stroke="#1d2742" />
              <XAxis dataKey="head" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ background: "#0e1528", border: "1px solid #1d2742" }} formatter={(v) => Number(v).toFixed(3)} />
              <Bar dataKey="v">{heads.map((h) => <Cell key={h.head} fill={h.v >= 0 ? "#f472b6" : "#34d399"} />)}</Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="mb-2 mt-4 text-sm muted">Change by face region (patch maps)</div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={regions}>
              <CartesianGrid stroke="#1d2742" />
              <XAxis dataKey="region" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ background: "#0e1528", border: "1px solid #1d2742" }} formatter={(v) => Number(v).toFixed(3)} />
              <Bar dataKey="spatial" fill="#38bdf8" />
              <Bar dataKey="temporal" fill="#a78bfa" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </Section>
  );
}
