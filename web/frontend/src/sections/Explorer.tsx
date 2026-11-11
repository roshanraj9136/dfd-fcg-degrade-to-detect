import { useEffect, useMemo, useState } from "react";
import type { Summary, VideoDetail, VideoIndex } from "../data";
import { FAKE_METHODS, videoDetail } from "../data";
import { Section, Toggle, Verdict } from "../components/ui";
import { FaceMap, Legend } from "../components/FaceMap";

const HEADS = ["spatial", "temporal", "fusion"];

export function Explorer({ videos, summary }: { videos: VideoIndex[]; summary: Summary }) {
  const [group, setGroup] = useState<string>("REAL");
  const list = useMemo(() => videos.filter((v) => (group === "REAL" ? v.label === 0 : v.method === group) && v.split !== "fit"),
    [videos, group]);
  const conds = Object.keys(summary.conditions);
  const [vid, setVid] = useState<string>("REAL/000");
  const [cond, setCond] = useState<string>("GB_5");
  const [detail, setDetail] = useState<VideoDetail | null>(null);
  const v = videos.find((x) => x.vid === vid) ?? list[0];
  useEffect(() => {
    if (!v) return;
    let alive = true;
    videoDetail(v.vid).then((d) => alive && setDetail(d)).catch(() => alive && setDetail(null));
    return () => { alive = false; };
  }, [v]);
  useEffect(() => { if (list.length && !list.some((x) => x.vid === vid)) setVid(list[0].vid); }, [list, vid]);
  if (!v) return null;
  const c = v.scores[cond] ? cond : "clean";
  const sc = v.scores[c];
  const map = detail?.maps[c];
  const img = v.public ? `/samples/${v.vid.replace("/", "__")}${c === "clean" ? "" : `__${c}`}.png` : undefined;
  const flagged = list.filter((x) => (x.scores[c]?.p ?? 0) >= 0.5).length;
  return (
    <Section id="explorer" kicker="Explore" title="Every test video, every condition"
      lead={<>Pick a video and a degradation. You get the released verdict, the corrected verdict, how much of the
        logit Evidence Nulling attributes to video quality, and the exact 16×16 evidence maps of both branches.
        Faces are drawn from landmarks because FF++ frames cannot be redistributed; the two samples that ship with the
        upstream repository (REAL/000, DF/000_003) show the actual crop.</>}>
      <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
        <div className="card space-y-4">
          <Toggle value={group} onChange={setGroup} options={[{ value: "REAL", label: "real" },
            ...Object.keys(FAKE_METHODS).map((k) => ({ value: k, label: k }))]} />
          <select className="w-full rounded-lg border border-edge bg-ink p-2 text-sm" value={v.vid} onChange={(e) => setVid(e.target.value)}>
            {list.map((x) => <option key={x.vid} value={x.vid}>{x.vid}{x.public ? " ★" : ""}</option>)}
          </select>
          <select className="w-full rounded-lg border border-edge bg-ink p-2 text-sm" value={c} onChange={(e) => setCond(e.target.value)}>
            {conds.filter((k) => k in v.scores).map((k) => <option key={k} value={k}>{k} · {summary.conditions[k]}</option>)}
          </select>
          <div className="text-xs muted">{flagged}/{list.length} videos in this group are called FAKE by the released model under {c}.</div>
          <div className="space-y-2 border-t border-edge pt-4 text-sm">
            <div className="flex justify-between"><span className="muted">ground truth</span><span>{v.label ? `FAKE (${FAKE_METHODS[v.method] ?? v.method})` : "REAL"}</span></div>
            <div className="flex justify-between"><span className="muted">released DFD-FCG</span><Verdict p={sc.p} /></div>
            <div className="flex justify-between"><span className="muted">Evidence Nulling</span><Verdict p={sc.p_null} /></div>
            <div className="flex justify-between"><span className="muted">… type unseen in fit</span><Verdict p={sc.p_null_unseen} /></div>
          </div>
          <div className="border-t border-edge pt-4">
            <div className="mb-2 text-xs uppercase tracking-widest muted">logit per head: total vs quality share</div>
            {HEADS.map((h, i) => {
              const d = sc.d[i], n = sc.nuisance[i];
              const w = (x: number) => `${Math.min(100, Math.abs(x) * 25)}%`;
              return (
                <div key={h} className="mb-2">
                  <div className="flex justify-between text-xs"><span>{h}</span><span className="font-mono">d={d.toFixed(2)} · quality {n >= 0 ? "+" : ""}{n.toFixed(2)}</span></div>
                  <div className="mt-1 h-2 rounded bg-white/5"><div className={`h-2 rounded ${d >= 0 ? "bg-fake/70" : "bg-real/70"}`} style={{ width: w(d) }} /></div>
                  <div className="mt-0.5 h-1.5 rounded bg-white/5"><div className="h-1.5 rounded bg-warn/80" style={{ width: w(n) }} /></div>
                </div>
              );
            })}
          </div>
        </div>
        <div className="card">
          {map && detail ? (
            <>
              <div className="grid gap-6 sm:grid-cols-2">
                <FaceMap values={map.s} landmarks={detail.landmarks} image={img} title="spatial head, per patch" />
                <FaceMap values={map.t} landmarks={detail.landmarks} image={img} title="temporal head, per patch" />
              </div>
              <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                <Legend />
                <span className="text-xs muted">Each map sums exactly to its head's logit minus the constant.</span>
              </div>
              {c !== "clean" && detail.maps.clean && (
                <div className="mt-6 border-t border-edge pt-6">
                  <div className="mb-3 text-sm muted">Change caused by {c} (degraded − clean)</div>
                  <div className="grid gap-6 sm:grid-cols-2">
                    <FaceMap values={map.s.map((x, i) => x - detail.maps.clean.s[i])} landmarks={detail.landmarks} image={img} title="spatial Δ" />
                    <FaceMap values={map.t.map((x, i) => x - detail.maps.clean.t[i])} landmarks={detail.landmarks} image={img} title="temporal Δ" />
                  </div>
                </div>
              )}
            </>
          ) : <div className="muted">Loading maps…</div>}
        </div>
      </div>
    </Section>
  );
}
