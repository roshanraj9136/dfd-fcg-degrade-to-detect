import { useCallback, useEffect, useRef, useState } from "react";
import type { Health, Progress, Result } from "./api";
import { SAMPLES, check, fetchSample, health, onSpaceStatus } from "./api";
import { FaceRegions, Gauge, HeroArt, Logo, OptionIcon, People, ScoreBar, Spinner, StepIcon, SureIcon, Timeline, UploadIcon } from "./art";
import ACC from "./accuracy.json";

const YEAR = new Date().getFullYear();
const MIN_SECONDS = 4;

// the three ways to check: one short line each, in plain words
const CHECK_OPTIONS = [
  { id: "none", title: "As it is", text: "Checks your video exactly as you uploaded it. Use this for a normal check." },
  { id: "blur", title: "Strong blur", text: "Blurs your video first, like an out-of-focus phone clip. Shows how the checker copes with blur." },
  { id: "compress", title: "Heavy compression", text: "Squeezes your video into a tiny, blocky file first, like a clip forwarded many times." },
] as const;

type Stage = "idle" | "uploading" | "working" | "done" | "error";
const STEPS = [
  { id: "upload", title: "Uploading", text: "Sending your video securely" },
  { id: "face", title: "Finding the faces", text: "Locating and aligning every face in the first 10 seconds" },
  { id: "gpu", title: "Getting a GPU", text: "A free GPU on Hugging Face is attached for the detector" },
  { id: "model", title: "Checking for manipulation", text: "The DFD-FCG detector checks 8 short clips of every face" },
  { id: "quality", title: "Quality check", text: "Removing the effect of blur and compression" },
  { id: "blur", title: "Blur test", text: "Seeing how the face reacts to a tiny extra blur" },
] as const;
const CPU_STEP = {
  title: "Using the server's CPU",
  text: "No free GPU time is left for your internet connection right now, so the detector runs on the CPU (same result, about a minute slower)",
};
// typical time per step (seconds) once the detector starts; the bar is an estimate, the answer arrives when the check is done
const GPU_PLAN: [string, number][] = [["gpu", 4], ["model", 5], ["quality", 0.5], ["blur", 3]];
const CPU_PLAN: [string, number][] = [["gpu", 0.5], ["model", 28], ["quality", 0.5], ["blur", 14]];
const MAX_MB = 60;

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

  useEffect(() => {
    let alive = true;
    const off = onSpaceStatus((s) => {
      if (!alive) return;
      const d = s.detail ?? "";
      setWake(d === "SLEEPING" || d === "APP_STARTING" || d === "BUILDING" || d === "RUNNING_BUILDING"
        ? "The checker is waking up on Hugging Face. This takes a minute or two the first time." : "");
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

  // step-by-step estimate once the detector has started (the face step reports real progress)
  useEffect(() => {
    if (stage !== "working" || !runStart) return;
    const plan = cpu ? CPU_PLAN : GPU_PLAN;
    const id = setInterval(() => {
      let t = (Date.now() - runStart) / 1000;
      for (let i = 0; i < plan.length; i++) {
        const [s, d] = plan[i];
        if (t < d || i === plan.length - 1) {
          setStep(s);
          setProg(Math.min(t / d, 0.95));
          return;
        }
        t -= d;
      }
    }, 200);
    return () => clearInterval(id);
  }, [stage, runStart, cpu]);

  const start = useCallback(async (blob: Blob, filename: string) => {
    setErr("");
    setRes(null);
    if (blob.size > MAX_MB * 1024 * 1024) {
      setErr(`This video is larger than ${MAX_MB} MB. Please trim it to a few seconds.`);
      setStage("error");
      return;
    }
    const secs = await videoSeconds(blob);
    if (secs != null && secs < MIN_SECONDS) {
      setName(filename);
      setErr(`This video is only ${secs.toFixed(1)} seconds long. Please use a video of at least ${MIN_SECONDS} seconds.`);
      setStage("error");
      setTimeout(() => progressRef.current?.scrollIntoView({ behavior: "smooth", block: "center" }), 100);
      return;
    }
    setName(filename);
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
          if (p.step === "quality") { setRunStart(0); setStep("blur"); setProg(0.95); }
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
    setStage("uploading");
    setStep("upload");
    setProg(0);
    try {
      const b = await fetchSample(urls);
      start(b, filename);
    } catch {
      setErr("Could not download the sample video. Please check your internet connection and try again.");
      setStage("error");
    }
  };

  const busy = stage === "uploading" || stage === "working";
  const uploadRef = useRef<HTMLElement>(null);
  const again = () => {
    setStage("idle");
    setRes(null);
    setErr("");
    setTimeout(() => uploadRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  };

  return (
    <div className="min-h-screen">
      <a href="#check" className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded-lg focus:bg-white focus:px-3 focus:py-2 focus:text-sm focus:shadow">
        Skip to the checker
      </a>
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-3 px-4 py-3 sm:px-5">
          <a href="/" className="flex min-w-0 items-center gap-2.5">
            <Logo className="h-8 w-8 shrink-0 sm:h-9 sm:w-9" />
            <span className="truncate text-base font-bold text-slate-900 sm:text-lg">Deepfake Video Checker</span>
          </a>
          <nav className="hidden items-center gap-1 text-sm text-slate-600 lg:flex" aria-label="Sections">
            {[["#how", "How it works"], ["#accuracy", "Accuracy"], ["#faq", "Questions"], ["#privacy", "Privacy"]].map(([href, l]) => (
              <a key={href} href={href} className="rounded-lg px-2.5 py-1.5 hover:bg-slate-100 hover:text-slate-900">{l}</a>
            ))}
          </nav>
          <div className="flex shrink-0 items-center gap-2">
            <ServerBadge h={h} />
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 sm:px-5">
        <section className="grid items-center gap-8 py-8 sm:py-10 md:grid-cols-2">
          <div>
            <h1 className="text-[1.75rem] font-extrabold leading-tight tracking-tight text-slate-900 sm:text-3xl lg:text-4xl">
              Is this video <span className="text-real">real</span> or a <span className="text-fake">deepfake</span>?
            </h1>
            <p className="mt-3 text-base text-slate-600">
              Upload a short video of a face. In about a minute you get the answer, a simple reason, and an honest
              “not sure” when the checker can't tell.
            </p>
            <ul className="mt-5 space-y-1.5 text-sm text-slate-600">
              <li className="flex gap-2"><Check /> Spots face swaps and changed facial expressions</li>
              <li className="flex gap-2"><Check /> Checks every person in the video</li>
              <li className="flex gap-2"><Check /> Fair to blurry or low-quality videos</li>
              <li className="flex gap-2"><Check /> Your video is deleted right after the check</li>
            </ul>
          </div>
          <HeroArt className="hero-art hidden w-full max-w-sm justify-self-center md:block" />
        </section>

        <section id="check" ref={uploadRef} className="card scroll-mt-20 p-4 sm:p-5 md:p-7">
          <StepTitle n={1} title="Choose how to check" />
          <div className="mt-3 grid gap-3 md:grid-cols-3">
            {CHECK_OPTIONS.map((o) => {
              const on = degrade === o.id;
              return (
                <button key={o.id} disabled={busy} onClick={() => setDegrade(o.id)} aria-pressed={on}
                  className={`flex min-h-[44px] items-start gap-3 rounded-2xl border p-3 text-left transition disabled:cursor-not-allowed disabled:opacity-60
                    ${on ? "border-brand bg-brand-soft ring-1 ring-brand" : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"}`}>
                  <OptionIcon kind={o.id} className="h-11 w-11 shrink-0" />
                  <span className="min-w-0">
                    <span className="flex flex-wrap items-center gap-1.5 text-sm font-semibold text-slate-900">
                      {o.title}
                      {o.id === "none" && <span className="rounded-full bg-real-soft px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-green-800">best</span>}
                    </span>
                    <span className="mt-0.5 block text-xs leading-snug text-slate-600">{o.text}</span>
                  </span>
                </button>
              );
            })}
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Not sure which? Keep <b className="font-semibold text-slate-600">As it is</b>. The other two make a worse copy of your
            video on purpose, to show how the checker copes with poor quality.
          </p>

          <StepTitle n={2} title="Pick a video" className="mt-6" />
          <div
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); if (!busy) onFiles(e.dataTransfer.files); }}
            className={`mt-3 flex flex-col items-center rounded-2xl border-2 border-dashed px-5 py-7 text-center transition
              ${drag ? "border-brand bg-brand-soft" : "border-slate-300 bg-slate-50"}`}>
            <UploadIcon className="h-11 w-11 text-brand" />
            <p className="mt-3 text-base font-semibold text-slate-900"><span className="[@media(pointer:coarse)]:hidden">Drop a video here</span><span className="hidden [@media(pointer:coarse)]:inline">Choose a video from your phone or tablet</span></p>
            <p className="mt-1 text-xs text-slate-500">MP4, MOV, AVI, MKV or WEBM · at least 4 seconds · one or two clear faces · up to {MAX_MB} MB</p>
            <div className="mt-5 flex flex-wrap justify-center gap-3">
              <button className="btn-primary" disabled={busy} onClick={() => input.current?.click()}>
                Choose a video
              </button>
              {SAMPLES.map((s) => (
                <button key={s.id} className="btn-ghost" disabled={busy} onClick={() => sample(s.urls, s.name)}>
                  {s.label}
                </button>
              ))}
            </div>
            
            <input ref={input} type="file" accept="video/*" className="hidden" onChange={(e) => { onFiles(e.target.files); e.target.value = ""; }} />
          </div>

          {h?.status !== "ready" && (wake || h !== undefined) && (
            <div className="mt-6 flex items-center gap-3 rounded-xl bg-amber-50 px-4 py-3 text-sm text-amber-800">
              <Spinner className="h-4 w-4 shrink-0" />
              {wake || (h?.status === "error" && h.detail
                ? `The checker is not available right now (${h.detail.toLowerCase().replace(/_/g, " ")}). Please try again later.`
                : "Connecting to the checker. If it was asleep, waking it up takes a minute or two.")}
            </div>
          )}
        </section>

        {(busy || stage === "error") && (
          <section ref={progressRef} className="card mt-8 scroll-mt-20 p-4 sm:p-6 md:p-8">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-slate-900">{stage === "error" ? "Something went wrong" : "Checking your video"}</h2>
              {name && <span className="truncate pl-4 text-sm text-slate-500">{name}</span>}
            </div>
            {stage === "error" ? (
              <>
                <div className="mt-4 rounded-xl bg-fake-soft px-4 py-3 text-fake" role="alert">{err}</div>
                <button className="btn-ghost mt-4" onClick={again}>Try another video</button>
              </>
            ) : (
              <>
                {ahead > 0 && <p className="mt-2 text-sm text-slate-500">{ahead} video{ahead > 1 ? "s" : ""} ahead of yours.</p>}
                <ol className="mt-6 space-y-4">
                  {STEPS.map((s0, i) => {
                    const s = s0.id === "gpu" && cpu ? { ...s0, ...CPU_STEP } : s0;
                    const cur = STEPS.findIndex((x) => x.id === step);
                    const state = i < cur ? "done" : i === cur ? "now" : "later";
                    return (
                      <li key={s.id} className="flex items-start gap-4">
                        <div className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl
                          ${state === "done" ? "bg-real-soft text-real" : state === "now" ? "bg-brand-soft text-brand" : "bg-slate-100 text-slate-400"}`}>
                          {state === "done" ? <StepIcon step="done" /> : <StepIcon step={s.id} />}
                        </div>
                        <div className="flex-1">
                          <div className={`font-semibold ${state === "later" ? "text-slate-500" : "text-slate-900"}`}>{s.title}</div>
                          <div className="text-sm text-slate-500">{s.text}</div>
                          {state === "now" && (
                            <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100">
                              <div className="h-2 rounded-full bg-brand transition-all" style={{ width: `${Math.max(4, prog * 100)}%` }} />
                            </div>
                          )}
                        </div>
                      </li>
                    );
                  })}
                </ol>
              </>
            )}
          </section>
        )}

        {stage === "done" && res && <ResultView r={res} preview={preview} refEl={resultRef} onAgain={again} />}

        <HowItWorks />
        <Accuracy />
        <Faq />
        <Privacy />
      </main>

      <footer className="mt-16 border-t border-slate-200 bg-white">
        <div className="mx-auto grid max-w-5xl gap-6 px-5 pt-8 pb-6 text-sm text-slate-600 md:grid-cols-[1.4fr_1fr]">
          <div>
            <div className="flex items-center gap-2 font-semibold text-slate-900"><Logo className="h-6 w-6" /> Deepfake Video Checker</div>
            <p className="mt-2">Find out in about a minute whether a face video is real or a deepfake. Free to use. The answer is a
              strong hint, not proof.</p>
          </div>
          <div className="grid grid-cols-2 gap-2 content-start">
            <a className="text-brand hover:underline" href="#how">How it works</a>
            <a className="text-brand hover:underline" href="#accuracy">Accuracy</a>
            <a className="text-brand hover:underline" href="#faq">Questions</a>
            <a className="text-brand hover:underline" href="#privacy">Privacy and terms</a>
          </div>
        </div>
        <div className="border-t border-slate-200">
          <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-2 px-5 py-4 text-xs text-slate-600">
            <span>Made by <b className="font-semibold text-slate-900">Roshan Raj</b></span>
            <span>© {YEAR} Roshan Raj. All rights reserved.</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

function StepTitle({ n, title, className = "" }: { n: number; title: string; className?: string }) {
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-brand text-xs font-bold text-white">{n}</span>
      <span className="text-sm font-semibold text-slate-900">{title}</span>
    </div>
  );
}

function Check() {
  return (
    <svg viewBox="0 0 20 20" className="mt-0.5 h-5 w-5 shrink-0 text-real" aria-hidden>
      <circle cx="10" cy="10" r="9" fill="#dcfce7" />
      <path d="M6 10.5l2.5 2.5L14 7.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
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
      className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-700 transition hover:bg-slate-100">
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
      {ok ? `Checker online${h?.analysed_total ? ` · ${h.analysed_total} videos checked` : ""}` : "Connecting…"}
    </span>
  );
}

const REGION_WORDS: Record<string, string> = { eyes: "the eyes", nose: "the nose", lips: "the mouth", skin: "the skin and cheeks" };

function ResultView({ r, preview, refEl, onAgain }: { r: Result; preview: string; refEl: React.RefObject<HTMLDivElement | null>; onAgain: () => void }) {
  const fake = r.verdict === "FAKE";
  const sorted = Object.entries(r.regions).sort((a, b) => b[1] - a[1]);
  const topFake = sorted[0];
  const topReal = sorted[sorted.length - 1];
  const changed = r.verdict !== r.verdict_original;
  const qualityPts = Math.round(r.quality_shift * 100);
  const blur = r.blur_response[2];
  const blurLike = blur > 0.25 ? "real" : blur < -0.25 ? "fake" : "unclear";
  const sure = r.sure ?? true;
  const nFaces = r.meta.faces_checked ?? 1;
  const shown = r.meta.face_shown ?? 1;
  const tone = !sure ? "ring-amber-200" : fake ? "ring-red-200" : "ring-green-200";
  const bg = !sure ? "bg-amber-50" : fake ? "bg-fake-soft" : "bg-real-soft";
  return (
    <section ref={refEl} className="mt-8 scroll-mt-20 space-y-6">
      <div className={`card overflow-hidden ring-2 ${tone}`}>
        <div className={`grid items-center justify-items-center gap-5 p-5 text-center sm:justify-items-start sm:text-left md:grid-cols-[auto_1fr_auto] md:gap-6 md:p-8 ${bg}`}>
          <Gauge p={r.p} verdict={r.verdict} unsure={!sure} />
          <div>
            <div className="flex flex-wrap items-center justify-center gap-2 sm:justify-start sm:gap-3">
              <span className="label">Result</span>
              <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold
                ${sure ? "bg-white text-slate-700" : "bg-amber-100 text-amber-800"}`}>
                <SureIcon sure={sure} className="h-4 w-4" />{sure ? "Sure" : "Not sure"}
              </span>
              {r.meta.degrade !== "none" && (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-white px-2.5 py-1 text-xs font-medium text-slate-600">
                  <OptionIcon kind={r.meta.degrade} className="h-4 w-4" />
                  checked with {r.meta.degrade === "blur" ? "strong blur" : "heavy compression"} added
                </span>
              )}
            </div>
            <h2 className={`mt-1 text-2xl font-extrabold sm:text-3xl ${!sure ? "text-amber-700" : fake ? "text-fake" : "text-real"}`}>
              {sure ? (fake ? "This looks like a deepfake" : "This looks like a real video")
                : `The checker can't tell. It leans ${fake ? "fake" : "real"}`}
            </h2>
            <p className="mt-2 text-slate-600">
              {sure ? (fake ? "The face shows signs of manipulation that the detector has learned from known deepfakes."
                : "The face and its movement look like an authentic recording.")
                : <>The score is {Math.round(r.p * 100)}% fake, close to the 50% line. On our test videos, about
                  {" "}{Math.round((1 - ACC.unsure_acc) * 10)} in 10 answers like this were wrong. <b className="font-semibold text-slate-800">Don't rely on
                  this answer</b>; check the video another way. New kinds of fakes, phone videos and faces unlike the detector's
                  training videos often end up here.</>}
            </p>
          </div>
          {preview && <video src={preview} className="hidden h-32 rounded-xl md:block" muted autoPlay loop playsInline />}
        </div>
      </div>

      {r.timeline && r.timeline.length > 0 && (
        <div className="card p-6 md:p-8">
          <h3 className="text-xl font-bold text-slate-900">Across the video</h3>
          <p className="mt-1 text-sm text-slate-500">
            Each dot is one 3-second clip{r.meta.mirrored ? ", checked normal and mirrored" : ""}. Dots in the red area point to fake.
            {nFaces > 1 && " The dashed line is the other person."}
          </p>
          <div className={`mt-4 grid gap-6 ${nFaces > 1 ? "md:grid-cols-[1fr_auto] md:items-center" : ""}`}>
            <div className="overflow-x-auto"><Timeline clips={r.timeline} seconds={r.meta.seconds_covered ?? 10} shown={shown} /></div>
            {nFaces > 1 && r.faces && (
              <div className="text-center">
                <People ps={r.faces.map((f) => f.p)} shown={shown} />
                <p className="mt-2 max-w-xs text-sm text-slate-500">
                  We found {nFaces} people. A face swap can be on either of them, so the answer is for the more suspicious one.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="card p-6 md:p-8">
        <h3 className="text-xl font-bold text-slate-900">How we got there</h3>
        <div className="mt-6 grid gap-6 md:grid-cols-3">
          <Explain n={1} title="We found the face" icon="face">
            <img src={r.face_png} alt="aligned face" className="mt-3 w-full rounded-xl" />
            <p className="mt-2 text-sm text-slate-500">The face is cut out and lined up the same way for every frame, so the detector compares like with like.</p>
          </Explain>
          <Explain n={2} title="Where the detector looked" icon="model">
            <img src={r.heat_png} alt="evidence map" className="mt-3 w-full rounded-xl" />
            <p className="mt-2 text-sm text-slate-500">
              <span className="font-semibold text-fake">Red</span> areas pushed towards fake, <span className="font-semibold text-real">green</span> towards real.
              {topFake[1] > 0 && <> The strongest fake signal came from <b>{REGION_WORDS[topFake[0]]}</b>.</>}
              {topReal[1] < 0 && <> The strongest real signal came from <b>{REGION_WORDS[topReal[0]]}</b>.</>}
            </p>
          </Explain>
          <Explain n={3} title="Fair to low-quality videos" icon="quality">
            <div className="mt-3 space-y-1">
              <div className="text-xs text-slate-500">Original detector</div>
              <ScoreBar p={r.p_original} label="original" muted />
              <div className="text-xs text-slate-500">After our quality check</div>
              <ScoreBar p={r.p} label="corrected" />
            </div>
            <p className="mt-2 text-sm text-slate-500">
              Blur and compression can make a real face look fake to the original detector. We estimate that effect from
              the detector's own features and remove it
              {Math.abs(qualityPts) >= 1 ? ` (${qualityPts > 0 ? "−" : "+"}${Math.abs(qualityPts)} points here)` : " (it was negligible here)"}.
              {changed && <b className="text-slate-700"> This changed the answer from {r.verdict_original} to {r.verdict}.</b>}
            </p>
          </Explain>
        </div>

        <div className="mt-8 grid gap-6 rounded-2xl bg-slate-50 p-5 md:grid-cols-[auto_1fr] md:items-center">
          <FaceRegions regions={r.regions} />
          <div>
            <div className="flex items-center gap-2 font-semibold text-slate-900"><StepIcon step="blur" className="h-5 w-5 text-brand" /> Extra check: the blur test</div>
            <p className="mt-2 text-sm text-slate-600">
              We blurred the face very slightly and ran the detector again. Real faces usually look <i>more</i> suspicious after a
              blur (their natural fine detail disappears); deepfakes usually do not. This video's reaction
              was <b>{blur >= 0 ? "+" : ""}{blur.toFixed(2)}</b>, which{" "}
              {blurLike === "real" ? <span className="font-semibold text-real">behaves like a real video</span>
                : blurLike === "fake" ? <span className="font-semibold text-fake">behaves like a deepfake</span>
                  : <span className="font-semibold text-slate-700">is not clearly either</span>}.
            </p>
            <p className="mt-2 text-xs text-slate-500">
              Strict mode (quality check + blur test together): {Math.round(r.p_strict * 100)}% fake.
              Checked {r.meta.clips} clips of 10 frames from the first {r.meta.seconds_covered ?? 5} seconds
              {r.meta.mirrored ? " (each also mirrored)" : ""}{nFaces > 1 ? ` · ${nFaces} people` : ""} · {r.meta.width}×{r.meta.height} · {r.meta.seconds}s
              {r.meta.device === "cpu" && " on the CPU"}
              {r.meta.degrade !== "none" && ` · with added ${r.meta.degrade === "blur" ? "blur" : "compression"}`}.
            </p>
          </div>
        </div>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-3">
        <button className="btn-primary" onClick={onAgain}>Check another video</button>
        <a className="btn-ghost" href="#faq">What does this answer mean?</a>
      </div>
    </section>
  );
}

function Explain({ n, title, icon, children }: { n: number; title: string; icon: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="flex items-center gap-3">
        <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-brand-soft text-brand"><StepIcon step={icon} className="h-5 w-5" /></span>
        <div className="font-semibold text-slate-900"><span className="text-slate-400">{n}.</span> {title}</div>
      </div>
      {children}
    </div>
  );
}

function HowItWorks() {
  const items = [
    { icon: "face", title: "1. Find the faces", text: "We track every clear face and line it up frame by frame." },
    { icon: "model", title: "2. Look for manipulation", text: "A state-of-the-art detector (DFD-FCG) studies 8 short clips of each face from the first 10 seconds." },
    { icon: "quality", title: "3. Correct for quality", text: "Blur and compression fool the original detector. Our add-on measures that effect and removes it." },
    { icon: "blur", title: "4. Know when unsure", text: "If the score is close to the line, the checker says it is not sure instead of guessing." },
  ];
  return (
    <section id="how" className="mt-16 scroll-mt-20">
      <h2 className="text-2xl font-bold text-slate-900">How it works</h2>
      <div className="mt-6 grid gap-4 md:grid-cols-4">
        {items.map((it) => (
          <div key={it.title} className="card p-5">
            <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-brand-soft text-brand"><StepIcon step={it.icon} /></span>
            <div className="mt-4 font-semibold text-slate-900">{it.title}</div>
            <p className="mt-1 text-sm text-slate-600">{it.text}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

/** Honest accuracy, from the 225-video test (numbers read from accuracy.json, made by service/study/checker_numbers.py). */
function Accuracy() {
  const a = ACC;
  const tiles: [string, string, string, string][] = [
    [pct(a.ffpp_clean_acc), "right on normal test videos", "FaceForensics++, the dataset the detector was made for", "text-real"],
    [pct(a.ffpp_blur_acc), "right on strongly blurred videos", `the original detector gets ${pct(a.ffpp_blur_acc_original)}`, "text-brand"],
    [pct(a.sure_acc), "right when it says “sure”", `when it says “not sure”: ${pct(a.unsure_acc)}`, "text-slate-900"],
  ];
  return (
    <section id="accuracy" className="mt-16 scroll-mt-20">
      <h2 className="text-2xl font-bold text-slate-900">How accurate is it?</h2>
      <p className="mt-2 max-w-3xl text-sm text-slate-600">
        Tested on {a.videos} research test videos that the checker had never seen, each also blurred and compressed.
      </p>
      <div className="mt-5 grid gap-4 md:grid-cols-3">
        {tiles.map(([big, what, note, cls]) => (
          <div key={what} className="card p-5">
            <div className={`text-3xl font-extrabold ${cls}`}>{big}</div>
            <div className="mt-1 text-sm font-semibold text-slate-900">{what}</div>
            <p className="mt-1 text-xs text-slate-600">{note}</p>
          </div>
        ))}
      </div>
      <p className="mt-3 text-xs text-slate-600">
        It is weaker on videos unlike its training data: {pct(a.dfdc_clean_acc)} on DFDC, a different dataset. That is why it says
        “not sure” instead of guessing.
      </p>
    </section>
  );
}

const FAQ: [string, string][] = [
  ["Is the answer proof?", "No. It is a strong hint from a research detector, not proof. Never use it alone to accuse someone or as legal evidence."],
  ["What does “not sure” mean?", "The score is close to the line between real and fake. In our tests such answers were wrong far more often, so check the video another way too."],
  ["Which videos work best?", "At least 4 seconds, one or two clear faces looking roughly at the camera, good light. Only the first 10 seconds are checked."],
  ["What can fool it?", "It was trained on face swaps and changed expressions in talking-head videos. It can miss videos made entirely by AI (for example Sora or Veo), lip-sync fakes, newer face-swap apps, faces turned away or covered, and crowds (it checks at most two people)."],
  ["Why does it take about a minute?", "It finds every face and checks 8 short clips of each. If the free GPU is busy or used up, it runs on a slower CPU instead."],
  ["What happens to my video?", "It is sent to the checker, used for this one check and deleted right after. We keep only anonymous numbers, never the video."],
];

function Faq() {
  return (
    <section id="faq" className="mt-16 scroll-mt-20">
      <h2 className="text-2xl font-bold text-slate-900">Questions</h2>
      <div className="mt-5 grid items-start gap-3 md:grid-cols-2">
        {FAQ.map(([q, a]) => (
          <details key={q} className="group card px-5 py-4">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-4 text-sm font-semibold text-slate-900">
              {q}
              <svg viewBox="0 0 20 20" className="h-4 w-4 shrink-0 text-brand transition group-open:rotate-45" aria-hidden>
                <path d="M10 4v12M4 10h12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            </summary>
            <p className="mt-2 text-sm text-slate-600">{a}</p>
          </details>
        ))}
      </div>
    </section>
  );
}

function Privacy() {
  return (
    <section id="privacy" className="mt-16 scroll-mt-20">
      <h2 className="text-2xl font-bold text-slate-900">Privacy and terms</h2>
      <div className="mt-5 grid gap-4 md:grid-cols-2">
        <div className="card p-5 text-sm text-slate-600">
          <div className="flex items-center gap-2 font-semibold text-slate-900"><ShieldIcon /> Your privacy</div>
          <ul className="mt-2 space-y-1.5">
            <li>• Your video goes to our checker on Hugging Face, is used for this one check and is deleted right after.</li>
            <li>• We keep only anonymous numbers (the answer, the scores, the video size and how long it took) to count checks.
              No video, name or face picture is stored.</li>
            <li>• This site sets no cookies of its own and has no ads or trackers. Hugging Face handles the connection under its
              own <a className="text-brand underline underline-offset-2" href="https://huggingface.co/privacy">privacy policy</a>.</li>
          </ul>
        </div>
        <div className="card p-5 text-sm text-slate-600">
          <div className="flex items-center gap-2 font-semibold text-slate-900"><ScaleIcon /> Fair use</div>
          <ul className="mt-2 space-y-1.5">
            <li>• Free to use, non-commercial. The detector inside it (DFD-FCG, Han et al., CVPR 2025) and its weights are licensed for research use only.</li>
            <li>• The answer is a hint, not proof, and can be wrong. Do not use it alone to accuse anyone or to make legal, job or money decisions.</li>
            <li>• Only upload videos you are allowed to share. Do not upload someone's private video without their consent.</li>
          </ul>
        </div>
      </div>
    </section>
  );
}

function ShieldIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5 text-brand" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 3l7 3v5c0 4.5-3 8.2-7 10-4-1.8-7-5.5-7-10V6z" /><rect x="9" y="10.5" width="6" height="5" rx="1" /><path d="M10.5 10.5V9a1.5 1.5 0 0 1 3 0v1.5" />
    </svg>
  );
}

function ScaleIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5 text-brand" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 4v16M7 20h10M5 8h14" /><path d="M5 8l-3 6a3 3 0 0 0 6 0zM19 8l-3 6a3 3 0 0 0 6 0z" />
    </svg>
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
