import type { Analysis } from "./Anatomy";
import { FAKE_METHODS, pct } from "../data";
import { Section } from "../components/ui";

type Rep = {
  n_videos: number; accuracy: number; acc_spatial_only: number; acc_temporal_only: number;
  query_share: Record<string, number>; region_spatial_share: Record<string, number>;
  region_temporal_share: Record<string, number>; spatial_share_shapley: number;
  top_layers_spatial: number[]; top_layers_temporal: number[]; region_spatial_signed: Record<string, number>;
  query_signed: Record<string, number>;
};

const argmax = (o: Record<string, number>) => Object.entries(o).reduce((a, b) => (b[1] > a[1] ? b : a))[0];

export function Replication({ analysis, earlier }: { analysis: Analysis; earlier: { claim: string; old: string; test: (r: Record<string, Rep>) => { now: string; holds: boolean | null } }[] }) {
  const R = analysis.replication as unknown as Record<string, Rep>;
  const groups = ["REAL", "DF", "F2F", "FS", "NT", "FSh"].filter((g) => g in R);
  const fp = analysis.fingerprints;
  return (
    <Section id="replication" kicker="Finding 4" title="Our earlier two-video findings, re-tested at scale"
      lead={<>The first explainability study had one real and one fake video. Here every claim is re-measured on the
        FF++ test split. Some hold, some do not; both outcomes are reported. A pattern the two videos could not show:
        identity swaps (Deepfakes, FaceSwap) are caught on the eyes, expression re-enactments (Face2Face,
        NeuralTextures) on the skin.</>}>
      <div className="card overflow-x-auto">
        <table className="grid-table min-w-[800px]">
          <thead><tr><th>earlier claim (2 videos)</th><th>earlier number</th><th>now (test split)</th><th>verdict</th></tr></thead>
          <tbody>
            {earlier.map((e) => {
              const r = e.test(R);
              return (
                <tr key={e.claim}>
                  <td className="text-white">{e.claim}</td><td className="muted">{e.old}</td><td>{r.now}</td>
                  <td>{r.holds == null ? <span className="chip">mixed</span> : r.holds
                    ? <span className="chip border-real/40 text-real">holds</span>
                    : <span className="chip border-fake/40 text-fake">does not hold</span>}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="card mt-6 overflow-x-auto">
        <div className="mb-3 text-sm muted">Clean test videos per group: accuracy at 0.5, single-branch accuracy, and where the evidence sits</div>
        <table className="grid-table min-w-[900px]">
          <thead><tr><th>group</th><th>n</th><th>accuracy</th><th>spatial only</th><th>temporal only</th>
            <th>spatial share (Shapley)</th><th>top query</th><th>top spatial region</th><th>top temporal region</th><th>deepest spatial layers</th></tr></thead>
          <tbody>
            {groups.map((g) => {
              const r = R[g];
              return (
                <tr key={g}>
                  <td className="text-white">{g === "REAL" ? "Real" : FAKE_METHODS[g]}</td><td>{r.n_videos}</td>
                  <td>{pct(r.accuracy, 0)}</td><td>{pct(r.acc_spatial_only, 0)}</td><td>{pct(r.acc_temporal_only, 0)}</td>
                  <td>{pct(r.spatial_share_shapley, 0)}</td>
                  <td>{argmax(r.query_share)} ({pct(r.query_share[argmax(r.query_share)], 0)})</td>
                  <td>{argmax(r.region_spatial_share)} ({pct(r.region_spatial_share[argmax(r.region_spatial_share)], 0)})</td>
                  <td>{argmax(r.region_temporal_share)} ({pct(r.region_temporal_share[argmax(r.region_temporal_share)], 0)})</td>
                  <td className="font-mono">{r.top_layers_spatial.join(", ")}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div className="card">
          <h3 className="text-lg font-semibold text-white">The explanation reveals how the fake was made</h3>
          <p className="mt-2 text-sm muted">A linear classifier on the attribution of clean fakes only (no pixels), five-fold
            and grouped by identity, names the manipulation method of {fp.n_videos} fake videos:</p>
          <div className="mt-4 flex gap-6">
            <div><div className="kpi text-real">{pct(fp.accuracy_explanation, 0)}</div><div className="text-xs muted">from the explanation</div></div>
            <div><div className="kpi text-slate-400">{pct(fp.accuracy_logits_only, 0)}</div><div className="text-xs muted">from the 3 head logits</div></div>
            <div><div className="kpi text-slate-500">{pct(fp.chance, 0)}</div><div className="text-xs muted">chance</div></div>
          </div>
        </div>
        <div className="card">
          <div className="mb-2 text-sm muted">Confusion (rows = true method)</div>
          <table className="text-sm">
            <thead><tr><th />{fp.classes.map((c) => <th key={c} className="px-3 font-normal muted">{c}</th>)}</tr></thead>
            <tbody>
              {fp.confusion.map((row, i) => {
                const tot = row.reduce((a, b) => a + b, 0) || 1;
                return (
                  <tr key={i}><td className="pr-3 muted">{fp.classes[i]}</td>
                    {row.map((v, j) => <td key={j} className="px-3 py-2 text-center font-mono"
                      style={{ background: `rgba(56,189,248,${(0.7 * v) / tot})` }}>{v}</td>)}</tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </Section>
  );
}
