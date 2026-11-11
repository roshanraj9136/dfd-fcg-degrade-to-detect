import { useEffect, useState } from "react";
import type { Decision } from "../data";
import { API, apiDecide, apiHealth, getStatic, pct, sgn } from "../data";
import { Section } from "../components/ui";

type Example = { vid: string; cond: string; clip: number; d: number[]; d_probe: Record<string, number[]> };

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
      setOut(await apiDecide({ d: ex.d, d_probe: ex.d_probe }));
    } catch (e) {
      setErr(`request failed: ${(e as Error).message}`);
    }
    setBusy(false);
  };
  return (
    <Section id="api" kicker="Use it" title="The decision rule is one API call"
      lead={<>Run DFD-FCG on a clip and on its blurred copy (qa/run_video.py in the repository does both), send the two
        sets of head logits, get both verdicts and a plain-language reading of the response. FastAPI on Render; the
        fitted rule and all results live in Neon Postgres.</>}>
      <div className="card space-y-4">
        <div className="font-mono text-xs muted">{health}</div>
        {ex && <div className="text-sm">Example: clip {ex.clip} of <b>{ex.vid}</b> under <b>{ex.cond}</b>, an authentic video.</div>}
        <pre className="eq whitespace-pre-wrap">{`POST ${API || "<api>"}/api/decide
{ "d": [d_s, d_t, d_a], "d_probe": { "pB2": [d_s, d_t, d_a], "pJ50": [d_s, d_t, d_a] } }`}</pre>
        <button className="btn btn-on" disabled={!ex || busy || !API} onClick={run}>{busy ? "calling…" : "Send the example"}</button>
        {err && <pre className="eq text-fake">{err}</pre>}
        {out && (
          <div className="eq space-y-2 whitespace-pre-wrap">
            <div>released: <b className={out.verdict_released === "FAKE" ? "text-fake" : "text-real"}>{out.verdict_released} {pct(out.p_released)}</b>
              {"  →  "}Degrade-to-Detect: <b className={out.verdict_degrade_to_detect === "FAKE" ? "text-fake" : "text-real"}>
                {out.verdict_degrade_to_detect} {pct(out.p_degrade_to_detect)}</b>
              {out.p_learned != null && <> · learned variant {pct(out.p_learned)}</>}</div>
            <div>response to the blur probe (spatial, temporal, fusion): {out.response.map((x) => sgn(x, 3)).join(", ")} · λ = {out.lam}</div>
            <div>{out.reading}</div>
          </div>
        )}
      </div>
    </Section>
  );
}
