import { useEffect, useState } from "react";
import type { Conditions, Summary, VideoIndex } from "./data";
import { getStatic, pct } from "./data";
import { Hero } from "./sections/Hero";
import { Collapse } from "./sections/Collapse";
import type { Analysis, DriftLayers } from "./sections/Anatomy";
import { Anatomy } from "./sections/Anatomy";
import { Method } from "./sections/Method";
import { Results } from "./sections/Results";
import { Explorer } from "./sections/Explorer";
import { Replication } from "./sections/Replication";
import { Playground } from "./sections/Playground";
import { Fragility } from "./sections/Fragility";
import { Negative } from "./sections/Negative";
import { REPO } from "./sections/Hero";
import { Loading } from "./components/ui";

type Rep = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
const top = (o: Record<string, number>) => Object.entries(o).reduce((a, b) => (b[1] > a[1] ? b : a));
const bottom = (o: Record<string, number>) => Object.entries(o).reduce((a, b) => (b[1] < a[1] ? b : a));

const EARLIER = [
  { claim: "The eyes query carries the FAKE verdict", old: "eyes 45.6% of spatial evidence",
    test: (R: Rep) => { const [k, v] = top(R.FAKE.query_share); return { now: `top query on fakes: ${k} ${pct(v, 0)}`, holds: k === "eyes" }; } },
  { claim: "The nose query is the strongest REAL vote", old: "nose −0.601 on the real video",
    test: (R: Rep) => { const [k, v] = bottom(R.REAL.query_signed); return { now: `most negative query on reals: ${k} (${v.toFixed(3)})`, holds: k === "nose" }; } },
  { claim: "REAL verdicts come mostly from the temporal branch", old: "70.9% temporal",
    test: (R: Rep) => ({ now: `temporal share on reals: ${pct(1 - R.REAL.spatial_share_shapley, 0)}`, holds: R.REAL.spatial_share_shapley < 0.5 }) },
  { claim: "FAKE verdicts split evenly between branches", old: "50.4% spatial",
    test: (R: Rep) => { const s = R.FAKE.spatial_share_shapley; return { now: `spatial share on fakes: ${pct(s, 0)}`, holds: s > 0.4 && s < 0.6 }; } },
  { claim: "Spatial evidence on fakes sits on the eye region", old: "eyes region 47.7%",
    test: (R: Rep) => { const [k, v] = top(R.FAKE.region_spatial_share); return { now: `top region: ${k} ${pct(v, 0)}`, holds: k === "eyes" }; } },
  { claim: "Temporal evidence sits on the skin", old: "skin 41.9% (fake), 66.6% (real)",
    test: (R: Rep) => { const [a] = top(R.FAKE.region_temporal_share); const [b] = top(R.REAL.region_temporal_share);
      return { now: `top temporal region: fakes ${a}, reals ${b}`, holds: a === "skin" && b === "skin" ? true : a === "skin" || b === "skin" ? null : false }; } },
  { claim: "The spatial branch reads deep layers", old: "layers 19, 17, 20",
    test: (R: Rep) => { const l: number[] = R.FAKE.top_layers_spatial; return { now: `top layers on fakes: ${l.join(", ")}`, holds: l.every((x) => x >= 13) }; } },
  { claim: "Either branch alone already gets clean video right", old: "spatial-only 0.912, temporal-only 0.908 (fake)",
    test: (R: Rep) => { const f = R.FAKE, r = R.REAL;
      return { now: `fakes: S ${pct(f.acc_spatial_only, 0)} / T ${pct(f.acc_temporal_only, 0)}; reals: S ${pct(r.acc_spatial_only, 0)} / T ${pct(r.acc_temporal_only, 0)}`,
        holds: Math.min(f.acc_spatial_only, f.acc_temporal_only, r.acc_spatial_only, r.acc_temporal_only) > 0.8 ? true : null }; } },
];

const NAV = [["collapse", "Collapse"], ["fragility", "Fragility"], ["anatomy", "Anatomy"], ["method", "Method"],
  ["results", "Results"], ["explorer", "Explorer"], ["replication", "Replication"], ["negative", "Negative result"], ["api", "API"]];

export default function App() {
  const [d, setD] = useState<{ cond: Conditions; summary: Summary; analysis: Analysis; layers: DriftLayers; videos: VideoIndex[] } | null>(null);
  const [err, setErr] = useState<string>("");
  useEffect(() => {
    Promise.all([getStatic<Conditions>("conditions.json"), getStatic<Summary>("summary.json"),
      getStatic<Analysis>("analysis.json"), getStatic<DriftLayers>("drift_layers.json"), getStatic<VideoIndex[]>("videos.json")])
      .then(([cond, summary, analysis, layers, videos]) => setD({ cond, summary, analysis, layers, videos }))
      .catch((e) => setErr(String(e)));
  }, []);
  return (
    <div>
      <nav className="fixed inset-x-0 top-0 z-50 border-b border-edge bg-ink/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3">
          <a href="#" className="font-semibold text-white">Degrade-to-Detect <span className="muted font-normal">for DFD-FCG</span></a>
          <div className="hidden gap-4 text-sm md:flex">
            {NAV.map(([id, label]) => <a key={id} href={`#${id}`} className="muted hover:text-white">{label}</a>)}
          </div>
        </div>
      </nav>
      {err && <div className="mx-auto max-w-6xl p-24 text-fake">Could not load results: {err}</div>}
      {!d && !err && <div className="mx-auto max-w-6xl p-24"><Loading /></div>}
      {d && (
        <>
          <Hero cond={d.cond} summary={d.summary} analysis={d.analysis} />
          <Collapse cond={d.cond} summary={d.summary} />
          <Fragility analysis={d.analysis} layers={d.layers} />
          <Anatomy analysis={d.analysis} layers={d.layers} />
          <Method summary={d.summary} cond={d.cond} />
          <Results cond={d.cond} summary={d.summary} />
          <Explorer videos={d.videos} summary={d.summary} />
          <Replication analysis={d.analysis} earlier={EARLIER} />
          <Negative analysis={d.analysis} cond={d.cond} />
          <Playground />
          <footer className="border-t border-edge">
            <div className="mx-auto max-w-6xl px-5 py-10 text-sm muted">
              <p>Built on DFD-FCG by Han, Huang, Hua and Chen (CVPR 2025) and its released checkpoint (research use only).
                Videos: FaceForensics++ (c23), used under its terms of use and not redistributed. Explainability engine:
                {" "}<a className="text-accent" href="https://github.com/roshanraj9136/dfd-fcg-explainability">dfd-fcg-explainability</a>.</p>
              <p className="mt-2">Roshan Raj · B.Tech project · <a className="text-accent" href={REPO}>source code</a></p>
            </div>
          </footer>
        </>
      )}
    </div>
  );
}
