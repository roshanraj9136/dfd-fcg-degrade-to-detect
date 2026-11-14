import { useCallback, useEffect, useRef, useState } from "react";
import type { Health, Progress, Result } from "./api";
import { SAMPLES, check, fetchSample, health, onSpaceStatus } from "./api";
import { Logo, People, Spinner, Timeline } from "./art";
import { DetectorBar, FlowArrow, FlowArt, MiniIcon, ScanArt, StatRing, VerdictEmblem } from "./art2";
import ACC from "./accuracy.json";

const YEAR = new Date().getFullYear();
const MIN_SECONDS = 4;
const MAX_MB = 60;
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

// optional quality test: one short line each
const OPTIONS = [
  { id: "none", title: "As uploaded", hint: "Checks your video exactly as it is. Best for a normal check." },
  { id: "blur", title: "Add blur", hint: "Blurs the video first, like an out-of-focus phone clip, to show the checker still works." },
  { id: "compress", title: "Add compression", hint: "Squeezes the video into a tiny, blocky file first, like a clip forwarded many times." },
] as const;

// what the visitor sees while the check runs: four simple steps (the server reports finer steps)
const STAGES = ["Uploading", "Finding faces", "Running 3 detectors", "Your answer"] as const;
const STAGE_OF: Record<string, number> = { upload: 0, face: 1, gpu: 2, model: 2, quality: 2, blur: 2 };
// typical seconds per server step once the detectors start; the bar is an estimate
const GPU_PLAN: [string, number][] = [["gpu", 4], ["model", 5], ["quality", 0.5], ["blur", 3]];
const CPU_PLAN: [string, number][] = [["gpu", 0.5], ["model", 28], ["quality", 0.5], ["blur", 14]];

type Stage = "idle" | "uploading" | "working" | "done" | "error";

export default function App() {
  const [h, setH] = useState<Health | undefined>(undefined);
  const [wake, setWake] = useState("");
  const [stage, setStage] = useState<Stage>("idle");
  const [step, setStep] = useState<string>("upload");
  const [prog, setProg] = useState(0);
  const [ahead, setAhead] = useState(0);
  const [runStart, setRunStart] = useState(0);
  const [cpu, setCpu] = useState(false);
  const [err, setErr] = useState("");
  const [res, setRes] = useState<Result | null>(null);
  const [name, setName] = useState("");
  const [preview, setPreview] = useState<string>("");
  const [degrade, setDegrade] = useState("none");
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);
  const progressRef = useRef<HTMLElement>(null);
  const uploadRef = useRef<HTMLElement>(null);

  useEffect(() => {
    let alive = true;
    const off = onSpaceStatus((s) => {
      if (!alive) return;
      const d = s.detail ?? "";
      setWake(d === "SLEEPING" || d === "APP_STARTING" || d === "BUILDING" || d === "RUNNING_BUILDING"
        ? "The checker is waking up. This takes a minute or two the first time." : "");
    });
    const tick = async () => {
      const x = await health();
      if (!alive) return;
      setH(x);
      if (x.status !== "ready") setTimeout(tick, 10000);
      else setWake("");
    };
    tick();
    return () => { alive = false; off(); };
  }, []);

  // step-by-step estimate once the detectors have started (the face step reports real progress)
  useEffect(() => {
    if (stage !== "working" || !runStart) return;
    const plan = cpu ? CPU_PLAN : GPU_PLAN;
    const total = plan.reduce((s, [, d]) => s + d, 0);
    const id = setInterval(() => {
      const t = (Date.now() - runStart) / 1000;
      setStep("model");
      setProg(Math.min(t / total, 0.95));
    }, 200);
    return () => clearInterval(id);
  }, [stage, runStart, cpu]);

  const start = useCallback(async (blob: Blob, filename: string) => {
    setErr("");
    setRes(null);
    setName(filename);
    if (blob.size > MAX_MB * 1024 * 1024) {
      setErr(`This video is larger than ${MAX_MB} MB. Please trim it to a few seconds.`);
      setStage("error");
      return;
    }
    const secs = await videoSeconds(blob);
    if (secs != null && secs < MIN_SECONDS) {
      setErr(`This video is only ${secs.toFixed(1)} seconds long. Please use a video of at least ${MIN_SECONDS} seconds.`);
      setStage("error");
      setTimeout(() => progressRef.current?.scrollIntoView({ behavior: "smooth", block: "center" }), 100);
      return;
    }
    setPreview((old) => { if (old) URL.revokeObjectURL(old); return URL.createObjectURL(blob); });
    setStage("uploading");
    setStep("upload");
    setProg(0.5);
    setRunStart(0);
    setCpu(false);
    setAhead(0);
    setTimeout(() => progressRef.current?.scrollIntoView({ behavior: "smooth", block: "center" }), 100);
    try {
      const r = await check(blob, filename, degrade, (p: Progress) => {
        if (p.phase === "queue") { setStage("working"); setStep("face"); setProg(0.02); setAhead(Math.max(0, (p.position ?? 1) - 1)); }
        if (p.phase === "run") { setStage("working"); setStep("face"); setProg(0.03); setAhead(0); }
        if (p.phase === "step") {
          setStage("working");
          setAhead(0);
          if (p.step === "face") { setStep("face"); if (p.frac != null) setProg(Math.min(p.frac, 0.97)); }
          if (p.step === "gpu" || p.step === "cpu") { setCpu(p.step === "cpu"); setRunStart(Date.now()); }
          if (p.step === "quality") { setRunStart(0); setStep("model"); setProg(0.96); }
        }
      });
      setRes(r);
      setStage("done");
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    } catch (e) {
      setErr((e as Error).message);
      setStage("error");
    }
  }, [degrade]);

  const onFiles = (files: FileList | null) => {
    const f = files?.[0];
    if (f) start(f, f.name);
  };

  const sample = async (urls: readonly string[], filename: string) => {
    setName(filename);
    setStage("uploading");
    setStep("upload");
    setProg(0.1);
    setTimeout(() => progressRef.current?.scrollIntoView({ behavior: "smooth", block: "center" }), 100);
    try {
      const b = await fetchSample(urls);
      start(b, filename);
    } catch {
      setErr("Could not download the sample video. Please check your internet connection and try again.");
      setStage("error");
    }
  };

  const busy = stage === "uploading" || stage === "working";
  const again = () => {
    setStage("idle");
    setRes(null);
    setErr("");
    setTimeout(() => uploadRef.current?.scrollIntoView({ behavior: "smooth", block: "center" }), 50);
  };
  const opt = OPTIONS.find((o) => o.id === degrade) ?? OPTIONS[0];

  return (
    <div className="min-h-screen">
      <a href="#check" className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded-lg focus:bg-white focus:px-3 focus:py-2 focus:text-sm focus:shadow">
        Skip to the checker
      </a>
      <header className="sticky top-0 z-30 border-b border-slate-200/70 bg-white/85 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-3 px-4 py-3 sm:px-5">
          <a href="/" className="flex min-w-0 items-center gap-2.5">
            <Logo className="h-8 w-8 shrink-0" />
            <span className="truncate text-base font-bold text-slate-900 sm:text-lg">Deepfake Checker</span>
          </a>
          <nav className="hidden items-center gap-1 text-sm text-slate-600 md:flex" aria-label="Sections">
            {[["#how", "How it works"], ["#accuracy", "Accuracy"], ["#faq", "FAQ"]].map(([href, l]) => (
              <a key={href} href={href} className="rounded-lg px-3 py-1.5 hover:bg-slate-100 hover:text-slate-900">{l}</a>
            ))}
          </nav>
          <div className="flex shrink-0 items-center gap-2">
            <ServerBadge h={h} />
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 sm:px-5">
        {/* hero + upload */}
        <section className="hero-glow pt-10 pb-6 text-center sm:pt-14">
          <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600">
            <span className="h-1.5 w-1.5 rounded-full bg-real" /> Free · {pct(ACC.overall_acc)} accurate on unseen test videos
          </span>
          <h1 className="mx-auto mt-4 max-w-3xl text-3xl font-extrabold leading-tight tracking-tight text-slate-900 sm:text-5xl">
            Is this video <span className="text-real">real</span> or a <span className="text-fake">deepfake</span>?
          </h1>
          <p className="mx-auto mt-3 max-w-xl text-base text-slate-600 sm:text-lg">
            Upload a face video. Get a clear answer, <b className="text-real">REAL</b> or <b className="text-fake">FAKE</b>, in about a minute.
          </p>
        </section>

        <section id="check" ref={uploadRef} className="mx-auto max-w-3xl scroll-mt-24">
          <div
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); if (!busy) onFiles(e.dataTransfer.files); }}
            className={`card relative overflow-hidden p-6 text-center transition sm:p-8 ${drag ? "ring-2 ring-brand" : ""}`}>
            <ScanArt className="mx-auto h-36 w-auto sm:h-40" />
            <p className="mt-3 text-lg font-bold text-slate-900">
              <span className="[@media(pointer:coarse)]:hidden">Drop a video here</span>
              <span className="hidden [@media(pointer:coarse)]:inline">Choose a video from your phone</span>
            </p>
            <p className="mt-1 text-xs text-slate-500">MP4, MOV or WEBM · 4 seconds or longer · up to {MAX_MB} MB</p>
            <button className="btn-primary mt-5 px-6 py-3 text-base" disabled={busy} onClick={() => input.current?.click()}>
              <MiniIcon kind="bolt" className="h-5 w-5" /> Choose a video
            </button>
            <p className="mt-4 text-sm text-slate-500">
              or try a sample:{" "}
              {SAMPLES.map((s, i) => (
                <span key={s.id}>
                  {i > 0 && " · "}
                  <button className="font-semibold text-brand hover:underline disabled:opacity-50" disabled={busy}
                    onClick={() => sample(s.urls, s.name)}>{s.id === "real" ? "real video" : "deepfake"}</button>
                </span>
              ))}
            </p>
            <input ref={input} type="file" accept="video/*" className="hidden" onChange={(e) => { onFiles(e.target.files); e.target.value = ""; }} />

            <div className="mt-6 border-t border-slate-100 pt-5">
              <div className="mx-auto inline-flex rounded-2xl bg-slate-100 p-1" role="radiogroup" aria-label="Quality test">
                {OPTIONS.map((o) => (
                  <button key={o.id} role="radio" aria-checked={degrade === o.id} disabled={busy} onClick={() => setDegrade(o.id)}
                    className={`rounded-xl px-3 py-1.5 text-xs font-semibold transition sm:px-4 sm:text-sm
                      ${degrade === o.id ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-800"}`}>
                    {o.title}
                  </button>
                ))}
              </div>
              <p className="mt-2 text-xs text-slate-500">{opt.hint}</p>
            </div>
          </div>

          {h?.status !== "ready" && (wake || h !== undefined) && (
            <div className="mt-4 flex items-center justify-center gap-3 rounded-2xl bg-amber-50 px-4 py-3 text-sm text-amber-800">
              <Spinner className="h-4 w-4 shrink-0" />
              {wake || (h?.status === "error" && h.detail
                ? `The checker is not available right now (${h.detail.toLowerCase().replace(/_/g, " ")}). Please try again later.`
                : "Connecting to the checker. If it was asleep, waking it up takes a minute or two.")}
            </div>
          )}

          <ul className="mt-5 grid gap-2 text-sm text-slate-600 sm:grid-cols-3">
            {([["target", `${pct(ACC.overall_acc)} correct on unseen videos`], ["layers", "3 AI detectors combined"],
              ["lock", "Video deleted after the check"]] as const).map(([k, t]) => (
              <li key={k} className="flex items-center justify-center gap-2">
                <span className="text-brand"><MiniIcon kind={k} /></span>{t}
              </li>
            ))}
          </ul>
        </section>

        {(busy || stage === "error") && (
          <section ref={progressRef} className="mx-auto mt-8 max-w-3xl scroll-mt-24">
            <div className="card p-6 sm:p-8">
              {stage === "error" ? (
                <div className="text-center">
                  <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-fake-soft text-fake">
                    <svg viewBox="0 0 24 24" className="h-7 w-7" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" aria-hidden>
                      <circle cx="12" cy="12" r="9" /><path d="M12 7.5v5.5M12 16.5v.01" />
                    </svg>
                  </div>
                  <h2 className="mt-3 text-xl font-bold text-slate-900">The check could not finish</h2>
                  <p className="mx-auto mt-2 max-w-md text-slate-600" role="alert">{err}</p>
                  <button className="btn-primary mt-5" onClick={again}>Try another video</button>
                </div>
              ) : (
                <Working stage={STAGE_OF[step] ?? 2} prog={stage === "uploading" ? 0.5 : prog} name={name} ahead={ahead} cpu={cpu} />
              )}
            </div>
          </section>
        )}

        {stage === "done" && res && <ResultView r={res} preview={preview} refEl={resultRef} onAgain={again} />}

        <HowItWorks />
        <Accuracy />
        <Faq />
      </main>

      <footer id="privacy" className="mt-20 border-t border-slate-200 bg-white">
        <div className="mx-auto grid max-w-5xl gap-8 px-5 py-10 text-sm text-slate-600 md:grid-cols-3">
          <div>
            <div className="flex items-center gap-2 font-bold text-slate-900"><Logo className="h-6 w-6" /> Deepfake Checker</div>
            <p className="mt-2">Free, non-commercial and made for the public good. The answer is a strong signal, not legal proof.</p>
          </div>
          <div>
            <div className="font-semibold text-slate-900">Privacy</div>
            <p className="mt-2">Your video is used for one check and deleted right after. Only anonymous numbers are kept. No cookies,
              ads or trackers. The checker runs on Hugging Face (<a className="text-brand underline underline-offset-2" href="https://huggingface.co/privacy">privacy policy</a>).</p>
          </div>
          <div>
            <div className="font-semibold text-slate-900">Fair use</div>
            <p className="mt-2">Only upload videos you may share. Built on DFD-FCG (Han et al., CVPR 2025; weights for research use only)
              and GenD (Yermakov et al., WACV 2026; MIT licence).</p>
          </div>
        </div>
        <div className="border-t border-slate-200">
          <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-2 px-5 py-4 text-xs text-slate-500">
            <span>Made by <b className="font-semibold text-slate-900">Roshan Raj</b></span>
            <span>© {YEAR} Roshan Raj. All rights reserved.</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

/** While the check runs: an animated scan and four simple steps. */
function Working({ stage, prog, name, ahead, cpu }: { stage: number; prog: number; name: string; ahead: number; cpu: boolean }) {
  return (
    <div className="grid items-center gap-6 sm:grid-cols-[auto_1fr]">
      <ScanArt className="mx-auto h-32 w-auto" active />
      <div>
        <h2 className="text-xl font-bold text-slate-900">Checking your video…</h2>
        {name && <p className="truncate text-sm text-slate-500">{name}</p>}
        <ol className="mt-4 space-y-2.5">
          {STAGES.map((t, i) => {
            const st = i < stage ? "done" : i === stage ? "now" : "later";
            return (
              <li key={t} className="flex items-center gap-3">
                <span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-bold
                  ${st === "done" ? "bg-real text-white" : st === "now" ? "bg-brand text-white" : "bg-slate-100 text-slate-400"}`}>
                  {st === "done" ? (
                    <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                      <path d="M5 10.5l3.2 3.2L15 7" />
                    </svg>
                  ) : i + 1}
                </span>
                <div className="flex-1">
                  <div className={`text-sm font-semibold ${st === "later" ? "text-slate-400" : "text-slate-900"}`}>{t}</div>
                  {st === "now" && (
                    <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-slate-100">
                      <div className="h-1.5 rounded-full bg-brand transition-all" style={{ width: `${Math.max(5, prog * 100)}%` }} />
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
        {ahead > 0 && <p className="mt-3 text-xs text-slate-500">{ahead} video{ahead > 1 ? "s" : ""} ahead of yours.</p>}
        {cpu && <p className="mt-3 text-xs text-slate-500">The free GPU is busy, so this check runs on the server's CPU. Same answer, about a minute slower.</p>}
      </div>
    </div>
  );
}

const REGION_WORDS: Record<string, string> = { eyes: "the eyes", nose: "the nose", lips: "the mouth", skin: "the skin and cheeks" };

function ResultView({ r, preview, refEl, onAgain }: { r: Result; preview: string; refEl: React.RefObject<HTMLDivElement | null>; onAgain: () => void }) {
  const fake = r.verdict === "FAKE";
  const sorted = Object.entries(r.regions).sort((a, b) => b[1] - a[1]);
  const topFake = sorted[0];
  const topReal = sorted[sorted.length - 1];
  const conf = Math.min(0.99, r.confidence ?? Math.max(r.p, 1 - r.p));
  const nFaces = r.meta.faces_checked ?? 1;
  const shown = r.meta.face_shown ?? 1;
  const dets = detectorBars(r);
  const band = ACC.bands?.find((b) => conf >= b.lo && conf < b.hi);
  const where = fake ? (topFake[1] > 0 ? REGION_WORDS[topFake[0]] : null) : (topReal[1] < 0 ? REGION_WORDS[topReal[0]] : null);
  return (
    <section ref={refEl} className="mx-auto mt-8 max-w-3xl scroll-mt-24 space-y-5">
      <div className={`card overflow-hidden ring-2 ${fake ? "ring-red-200" : "ring-green-200"}`}>
        <div className={`flex flex-col items-center gap-5 p-6 text-center sm:flex-row sm:p-8 sm:text-left ${fake ? "bg-fake-soft" : "bg-real-soft"}`}>
          <VerdictEmblem fake={fake} className="h-20 w-20 shrink-0" />
          <div className="flex-1">
            <div className="label">Result</div>
            <h2 className={`text-5xl font-black tracking-tight ${fake ? "text-fake" : "text-real"}`}>{fake ? "FAKE" : "REAL"}</h2>
            <p className="mt-1 text-lg font-semibold text-slate-800">{fake ? "This video is a deepfake." : "This video is authentic."}</p>
            <div className="mt-3 flex flex-wrap items-center justify-center gap-2 sm:justify-start">
              <span className="rounded-full bg-white px-3 py-1 text-sm font-bold text-slate-800">{Math.round(conf * 100)}% confidence</span>
              {r.meta.degrade !== "none" && (
                <span className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-600">
                  tested with {r.meta.degrade === "blur" ? "blur" : "compression"} added
                </span>
              )}
            </div>
            {band && <p className="mt-2 text-xs text-slate-600">In our tests, answers this confident were right {pct(band.acc ?? 0)} of the time.</p>}
          </div>
          {preview && <video src={preview} className="hidden h-28 rounded-2xl shadow-sm md:block" muted autoPlay loop playsInline />}
        </div>
      </div>

      <div className="card p-6 sm:p-8">
        <h3 className="text-lg font-bold text-slate-900">Why</h3>
        <div className="mt-4 grid gap-6 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
          <div>
            <div className="grid grid-cols-2 gap-2">
              <img src={r.face_png} alt="the face that was checked" className="w-full rounded-2xl" />
              <img src={r.heat_png} alt="where the detector looked" className="w-full rounded-2xl" />
            </div>
            <p className="mt-2 text-xs text-slate-500">
              Right: <span className="font-semibold text-fake">red</span> points to fake, <span className="font-semibold text-real">green</span> to real.
              {where && <> Clearest sign around <b className="text-slate-700">{where}</b>.</>}
            </p>
          </div>
          <div className="space-y-3">
            {dets.map(([n, v]) => (
              <div key={n}>
                <div className="flex justify-between text-xs"><span className="text-slate-600">{n}</span>
                  <span className={`font-semibold ${v >= 0.5 ? "text-fake" : "text-real"}`}>{v >= 0.5 ? "fake" : "real"}</span></div>
                <DetectorBar p={v} />
              </div>
            ))}
            <div className="border-t border-slate-100 pt-3">
              <div className="flex justify-between text-sm"><span className="font-semibold text-slate-900">Final answer</span>
                <span className={`font-bold ${fake ? "text-fake" : "text-real"}`}>{fake ? "FAKE" : "REAL"}</span></div>
              <DetectorBar p={r.p} strong />
            </div>
            <p className="text-xs text-slate-500">Bars to the left point to real, to the right to fake. Each detector is corrected for blur and compression.</p>
          </div>
        </div>

        {r.timeline && r.timeline.length > 0 && (
          <details className="group mt-6 rounded-2xl bg-slate-50 px-4 py-3">
            <summary className="flex cursor-pointer list-none items-center justify-between text-sm font-semibold text-slate-800">
              Across the video{nFaces > 1 ? ` · ${nFaces} people` : ""}
              <Plus />
            </summary>
            <div className="mt-3 overflow-x-auto"><Timeline clips={r.timeline} seconds={r.meta.seconds_covered ?? 10} shown={shown} /></div>
            {nFaces > 1 && r.faces && (
              <div className="mt-3 text-center">
                <People ps={r.faces.map((f) => f.p)} shown={shown} />
                <p className="mt-1 text-xs text-slate-500">A face swap can be on either person, so the answer is for the more suspicious one.</p>
              </div>
            )}
            <p className="mt-2 text-xs text-slate-500">
              Each dot is one 3-second clip. {r.meta.clips} clips from the first {r.meta.seconds_covered ?? 10} seconds ·
              {" "}{r.meta.width}×{r.meta.height} · {r.meta.seconds}s.
            </p>
          </details>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-center gap-3">
        <button className="btn-primary px-6" onClick={onAgain}>Check another video</button>
      </div>
    </section>
  );
}

/** One bar per detector network (the quality-corrected score where the checker has one). */
function detectorBars(r: Result): [string, number][] {
  const m = r.members;
  if (!m || Object.keys(m).length < 2) return [["Video detector", r.p_dfd ?? r.p]];
  const out: [string, number][] = [];
  const pick = (a: string, b: string) => (a in m ? m[a] : b in m ? m[b] : undefined);
  const add = (n: string, v: number | undefined) => { if (v != null) out.push([n, v]); };
  add("Motion detector (3-second clips)", m.dfd);
  add("Detail detector, trained on 2025–26 fakes", pick("pe_ft_sc", "pe_ft"));
  add("Detail detector (skin and edges)", pick("pe_sc", "pe"));
  add("Detail detector 2", pick("clip_ft_sc", "clip_ft"));
  add("Motion detector, retrained", m.dfdft);
  return out;
}

function HowItWorks() {
  const steps = [
    { art: "face" as const, title: "Find the face", text: "Every clear face in the first 10 seconds is tracked and lined up." },
    { art: "detectors" as const, title: "Three detectors", text: "One watches movement, two study fine detail. All are corrected for blur and compression." },
    { art: "answer" as const, title: "One clear answer", text: "Their scores are combined into REAL or FAKE, with a confidence." },
  ];
  return (
    <section id="how" className="mt-20 scroll-mt-24">
      <h2 className="text-center text-2xl font-bold text-slate-900 sm:text-3xl">How it works</h2>
      <div className="mt-8 flex flex-col items-center gap-6 md:flex-row md:items-start md:justify-center md:gap-2">
        {steps.map((s, i) => (
          <div key={s.title} className="contents">
            {i > 0 && <FlowArrow className="hidden h-6 w-12 shrink-0 md:mt-5 md:block" />}
            <div className="max-w-[16rem] text-center">
              <FlowArt step={s.art} className="mx-auto h-16 w-16" />
              <div className="mt-3 font-bold text-slate-900">{s.title}</div>
              <p className="mt-1 text-sm text-slate-600">{s.text}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

/** Accuracy on held-out test videos (numbers read from accuracy.json, written by service/ens/report.py). */
function Accuracy() {
  const a = ACC;
  const hi = a.bands?.find((b) => b.lo >= 0.8);
  const rings: [number, string, string, string][] = [
    [a.overall_acc, "#16a34a", `correct on ${a.overall_videos} unseen test videos`, `a single detector alone: ${pct(a.overall_acc_original)}`],
    [a.new_bacc, "#4f46e5", "on the newest deepfakes", "2025–2026 datasets, real and fake counted equally"],
    [hi?.acc ?? 0, "#0ea5e9", "right when confidence is 80%+", hi ? `${pct(hi.share)} of all answers` : ""],
  ];
  return (
    <section id="accuracy" className="mt-20 scroll-mt-24">
      <h2 className="text-center text-2xl font-bold text-slate-900 sm:text-3xl">How accurate is it?</h2>
      <p className="mx-auto mt-2 max-w-xl text-center text-sm text-slate-600">
        Tested on {a.overall_videos} videos from {a.datasets} public datasets (2019–2026) that were never used to train or tune it.
      </p>
      <div className="card mx-auto mt-8 grid max-w-4xl gap-8 p-6 sm:grid-cols-3 sm:p-8">
        {rings.map(([v, c, t, n]) => (
          <div key={t} className="text-center">
            <StatRing v={v} color={c} className="mx-auto h-28 w-28" />
            <div className="mt-2 font-semibold text-slate-900">{t}</div>
            <p className="mt-0.5 text-xs text-slate-500">{n}</p>
          </div>
        ))}
      </div>
      <p className="mx-auto mt-4 max-w-2xl text-center text-xs text-slate-500">
        Like every detector it can be wrong, most often on heavily blurred or compressed copies and on brand-new kinds of fake.
      </p>
    </section>
  );
}

const FAQ: [string, string][] = [
  ["Is the answer proof?", "It is a strong, tested signal from three AI detectors, but not legal proof. Never use it alone to accuse someone."],
  ["What does the confidence mean?", `How strongly the detectors agree. In our tests, answers with 80% confidence or more were right ${pct(ACC.bands?.find((b) => b.lo >= 0.8)?.acc ?? 0)} of the time.`],
  ["Which videos work best?", "At least 4 seconds, one or two clear faces looking roughly at the camera, in good light. The first 10 seconds are checked."],
  ["What can it detect?", "Face swaps, changed expressions, animated talking heads and fully AI-generated faces. It can miss brand-new generators, lip-sync-only fakes and faces that are turned away or covered."],
  ["What happens to my video?", "It is used for this one check and deleted right after. We keep only anonymous numbers, never the video or a picture of the face."],
];

function Faq() {
  return (
    <section id="faq" className="mt-20 scroll-mt-24">
      <h2 className="text-center text-2xl font-bold text-slate-900 sm:text-3xl">Questions</h2>
      <div className="card mx-auto mt-8 max-w-3xl divide-y divide-slate-100 px-2">
        {FAQ.map(([q, a]) => (
          <details key={q} className="group px-4 py-4">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-4 font-semibold text-slate-900">
              {q}<Plus />
            </summary>
            <p className="mt-2 text-sm text-slate-600">{a}</p>
          </details>
        ))}
      </div>
    </section>
  );
}

function Plus() {
  return (
    <svg viewBox="0 0 20 20" className="h-4 w-4 shrink-0 text-brand transition group-open:rotate-45" aria-hidden>
      <path d="M10 4v12M4 10h12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function ThemeToggle() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  const flip = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    document.querySelectorAll('meta[name="theme-color"]').forEach((m) => m.setAttribute("content", next ? "#0b1120" : "#4f46e5"));
    try { localStorage.setItem("theme", next ? "dark" : "light"); } catch { /* storage blocked */ }
  };
  return (
    <button onClick={flip} aria-label={dark ? "Switch to light mode" : "Switch to dark mode"} title={dark ? "Light mode" : "Dark mode"}
      className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-700 transition hover:bg-slate-100">
      {dark ? (
        <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
          <circle cx="12" cy="12" r="4.2" /><path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6M5.3 18.7l1.6-1.6M17.1 6.9l1.6-1.6" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
          <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />
        </svg>
      )}
    </button>
  );
}

function ServerBadge({ h }: { h: Health | undefined }) {
  const ok = h?.status === "ready";
  return (
    <span className={`hidden items-center gap-2 rounded-full px-3 py-1 text-xs font-medium sm:inline-flex
      ${ok ? "bg-real-soft text-green-800" : "bg-amber-50 text-amber-800"}`}>
      <span className={`h-2 w-2 rounded-full ${ok ? "bg-real" : "bg-amber-500"}`} />
      {ok ? "Online" : "Connecting…"}
    </span>
  );
}

/** Length of a video in seconds from the browser, or null if the browser cannot read it (the server then decides). */
function videoSeconds(blob: Blob): Promise<number | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(blob);
    const v = document.createElement("video");
    const done = (s: number | null) => { URL.revokeObjectURL(url); resolve(s); };
    const timer = setTimeout(() => done(null), 4000);
    v.preload = "metadata";
    v.muted = true;
    v.onloadedmetadata = () => { clearTimeout(timer); done(Number.isFinite(v.duration) ? v.duration : null); };
    v.onerror = () => { clearTimeout(timer); done(null); };
    v.src = url;
  });
}
