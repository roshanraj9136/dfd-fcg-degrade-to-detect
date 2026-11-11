import { useEffect, useMemo, useState } from "react";
import type { Summary, VideoDetail, VideoIndex } from "../data";
import { FAKE_METHODS, sgn, videoDetail } from "../data";
import { Section, Toggle, Verdict } from "../components/ui";
import { FaceMap, Legend } from "../components/FaceMap";

const HEADS = ["spatial", "temporal", "fusion"];

export function Explorer({ videos, summary }: { videos: VideoIndex[]; summary: Summary }) {
  const [group, setGroup] = useState<string>("REAL");
  const list = useMemo(() => videos.filter((v) => v.split !== "fit" && (group === "REAL" ? v.label === 0 : v.method === group)),
    [videos, group]);
  const conds = Object.keys(summary.conditions);
  const [vid, setVid] = useState<string>("REAL/000");
  const [cond, setCond] = useState<string>("GB_5");
  const [view, setView] = useState<"evidence" | "probe">("evidence");
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
  const flagged2 = list.filter((x) => (x.scores[c]?.p_d2d ?? 0) >= 0.5).length;
  const r = sc.r?.pB2 ?? [0, 0, 0];
  return (
    <Section id="explorer" kicker="Explore" title="Every test video, every condition"
      lead={<>Pick a video and a degradation. You see the released verdict, the Degrade-to-Detect verdict, how each head
        responded to the blur probe, and the exact 16×16 evidence maps. Faces are drawn from landmarks because FF++
        frames cannot be redistributed; the two samples shipped with the upstream repository (REAL/000, DF/000_003, ★)
        show the actual crop.</>}>
      <div className="grid gap-6 lg:grid-cols-[330px_1fr]">
        <div className="card space-y-4">
          <Toggle value={group} onChange={setGroup} options={[{ value: "REAL", label: "real" },
            ...Object.keys(FAKE_METHODS).map((k) => ({ value: k, label: k }))]} />
          <select className="w-full rounded-lg border border-edge bg-ink p-2 text-sm" value={v.vid} onChange={(e) => setVid(e.target.value)}>
            {list.map((x) => <option key={x.vid} value={x.vid}>{x.vid}{x.public ? " ★" : ""}</option>)}
          </select>
          <select className="w-full rounded-lg border border-edge bg-ink p-2 text-sm" value={c} onChange={(e) => setCond(e.target.value)}>
            {conds.filter((k) => k in v.scores).map((k) => <option key={k} value={k}>{k} · {summary.conditions[k]}</option>)}
          </select>
          <div className="text-xs muted">Called FAKE in this group under {c}: released {flagged}/{list.length},
            Degrade-to-Detect {flagged2}/{list.length}.</div>
          <div className="space-y-2 border-t border-edge pt-4 text-sm">
            <div className="flex justify-between"><span className="muted">ground truth</span><span>{v.label ? `FAKE (${FAKE_METHODS[v.method] ?? v.method})` : "REAL"}</span></div>
            <div className="flex justify-between"><span className="muted">released DFD-FCG</span><Verdict p={sc.p} /></div>
            <div className="flex justify-between"><span className="muted">Degrade-to-Detect</span><Verdict p={sc.p_d2d} /></div>
            <div className="flex justify-between"><span className="muted">… type unseen in fit</span><Verdict p={sc.p_d2d_unseen} /></div>
          </div>
          <div className="border-t border-edge pt-4">
            <div className="mb-2 text-xs uppercase tracking-widest muted">head logit and its response to the blur probe</div>
            {HEADS.map((h, i) => (
              <div key={h} className="mb-2 flex items-center justify-between text-xs">
                <span className="w-16">{h}</span>
                <span className="font-mono">d = {sgn(sc.d[i])}</span>
                <span className={`font-mono ${r[i] < 0 ? "text-real" : "text-fake"}`}>r = {sgn(r[i])}</span>
              </div>
            ))}
            <p className="mt-2 text-xs muted">{r[2] < -0.25 ? "The FAKE evidence collapsed under the probe: fragile, typical of manipulation."
              : r[2] > 0.25 ? "The evidence grew under the probe: fine detail removed, typical of authentic video."
              : "The evidence barely moved under the probe."}</p>
          </div>
        </div>
        <div className="card">
          <div className="mb-4"><Toggle value={view} onChange={setView} options={[
            { value: "evidence", label: "evidence" }, { value: "probe", label: "change under the blur probe" }]} /></div>
          {map && detail ? (
            <>
              <div className="grid gap-6 sm:grid-cols-2">
                <FaceMap values={view === "evidence" ? map.s : map.sp} landmarks={detail.landmarks} image={img}
                  title={view === "evidence" ? "spatial head, per patch" : "spatial Δ under probe"} />
                <FaceMap values={view === "evidence" ? map.t : map.tp} landmarks={detail.landmarks} image={img}
                  title={view === "evidence" ? "temporal head, per patch" : "temporal Δ under probe"} />
              </div>
              <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                <Legend />
                <span className="text-xs muted">Each evidence map sums exactly to its head's logit minus a constant.</span>
              </div>
              {view === "evidence" && c !== "clean" && detail.maps.clean && (
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
