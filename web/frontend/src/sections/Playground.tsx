import { useEffect, useState } from "react";
import { API, apiHealth, apiNullify, getStatic, pct } from "../data";
import { Section } from "../components/ui";

type Example = { vid: string; cond: string; clip: number; s: number[]; t: number[]; a: number[]; d: number[]; q_bg: number[] };
type Out = Awaited<ReturnType<typeof apiNullify>>;

export function Playground() {
  const [ex, setEx] = useState<Example | null>(null);
  const [health, setHealth] = useState<string>(API ? "waking the API (free tier, can take up to a minute)…" : "no API configured");
  const [out, setOut] = useState<Out | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    getStatic<Example>("example.json").then(setEx).catch(() => setEx(null));
    apiHealth().then((h) => setHealth(h ? `API ${h.status} · database ${h.database ?? "?"} · ${h.videos ?? "?"} videos in Neon` : "API unreachable right now; every number on this page is also bundled with the site"));
  }, []);
  const run = async () => {
    if (!ex) return;
    setBusy(true);
    setErr("");
    try {
      setOut(await apiNullify({ s: ex.s, t: ex.t, a: ex.a, d: ex.d, q_bg: ex.q_bg }));
    } catch (e) {
      setErr(`request failed: ${(e as Error).message}`);
    }
    setBusy(false);
  };
  return (
    <Section id="api" kicker="Use it" title="The correction is one API call"
      lead={<>Run DFD-FCG with our attribution engine wherever you have a GPU and send, per clip, the three head
        logits and eight background-quality numbers. The terms are optional and only used to name where the quality
        evidence sits. The API is FastAPI on Render, reading the fitted correction from Neon Postgres.</>}>
      <div className="card space-y-4">
        <div className="font-mono text-xs muted">{health}</div>
        {ex && <div className="text-sm">Example: clip {ex.clip} of <b>{ex.vid}</b> under <b>{ex.cond}</b>, an authentic video.</div>}
        <pre className="eq whitespace-pre-wrap">{`POST ${API || "<api>"}/api/nullify
{ "d": [d_s, d_t, d_a], "q_bg": [8 background descriptors], "s": [96], "t": [24], "a": [120] }`}</pre>
        <button className="btn btn-on" disabled={!ex || busy || !API} onClick={run}>{busy ? "calling…" : "Send the example"}</button>
        {err && <pre className="eq text-fake">{err}</pre>}
        {out && (
          <div className="eq space-y-2 whitespace-pre-wrap">
            <div>released: <b className={out.verdict_original === "FAKE" ? "text-fake" : "text-real"}>{out.verdict_original} {pct(out.p_original)}</b>
              {"  →  "}Evidence Nulling: <b className={out.verdict_corrected === "FAKE" ? "text-fake" : "text-real"}>{out.verdict_corrected} {pct(out.p_corrected)}</b></div>
            <div>quality evidence per head (logit): {out.nuisance_logit.map((x) => x.toFixed(3)).join(", ")}</div>
            <div>terms carrying most quality evidence: {out.top_quality_terms.map((t) => `${t.term} (${t.evidence >= 0 ? "+" : ""}${t.evidence.toFixed(3)})`).join("; ")}</div>
          </div>
        )}
      </div>
    </Section>
  );
}
