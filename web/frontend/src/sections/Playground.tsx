import { useEffect, useState } from "react";
import type { Decision } from "../data";
import { API, apiDecide, apiHealth, getStatic, pct, sgn } from "../data";
import { Section } from "../components/ui";

type Example = { vid: string; cond: string; clip: number; d: number[]; y_s: number[]; y_t: number[]; d_probe: Record<string, number[]> };

export function Playground() {
  const [ex, setEx] = useState<Example | null>(null);
  const [health, setHealth] = useState<string>(API ? "waking the API (free tier, can take up to a minute)…" : "no API configured");
  const [out, setOut] = useState<Decision | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    getStatic<Example>("example.json").then(setEx).catch(() => setEx(null));
    apiHealth().then((h) => setHealth(h ? `API ${h.status} · database ${h.database ?? "?"} · ${h.videos ?? "?"} videos in Neon Postgres`
      : "API unreachable right now; every number on this page is also bundled with the site"));
  }, []);
  const run = async () => {
    if (!ex) return;
    setBusy(true);
    setErr("");
    try {
      setOut(await apiDecide({ d: ex.d, y_s: ex.y_s, y_t: ex.y_t, d_probe: ex.d_probe }));
    } catch (e) {
      setErr(`request failed: ${(e as Error).message}`);
    }
    setBusy(false);
  };
  const v = (s?: string) => (s === "FAKE" ? "text-fake" : "text-real");
  return (
    <Section id="api" kicker="Use it" title="The correction is one API call"
      lead={<>Run DFD-FCG with our engine on a clip (qa/run_video.py does it for any video), send the three head
        logits and the two head inputs, optionally the logits of the blurred copy, and get the released, self-calibrated
        and strict verdicts. FastAPI; the fitted read-out and all results live in Neon Postgres.</>}>
      <div className="card space-y-4">
        <div className="font-mono text-xs muted">{health}</div>
        {ex && <div className="text-sm">Example: clip {ex.clip} of <b>{ex.vid}</b> under <b>{ex.cond}</b>, an authentic video.</div>}
        <pre className="eq whitespace-pre-wrap">{`POST ${API || "<api>"}/api/decide
{ "d": [d_s, d_t, d_a], "y_s": [1024 values], "y_t": [256 values], "d_probe": { "pB2": [d_s, d_t, d_a] } }`}</pre>
        <button className="btn btn-on" disabled={!ex || busy || !API} onClick={run}>{busy ? "calling…" : "Send the example"}</button>
        {err && <pre className="eq text-fake">{err}</pre>}
        {out && (
          <div className="eq space-y-2 whitespace-pre-wrap">
            <div>released: <b className={v(out.verdict_released)}>{out.verdict_released} {pct(out.p_released)}</b>
              {"  →  "}self-calibrated: <b className={v(out.verdict_selfcal)}>{out.verdict_selfcal} {pct(out.p_selfcal)}</b>
              {out.verdict_strict && <>{"  ·  "}strict: <b className={v(out.verdict_strict)}>{out.verdict_strict} {pct(out.p_strict)}</b></>}</div>
            <div>spurious evidence attributed to video quality (spatial, temporal, fusion): {out.spurious_evidence.map((x) => sgn(x, 3)).join(", ")}</div>
            {out.response && <div>response to the blur probe: {out.response.map((x) => sgn(x, 3)).join(", ")}. {out.reading}</div>}
          </div>
        )}
      </div>
    </Section>
  );
}
