import { useCallback, useEffect, useRef, useState } from "react";
import type { Health, Progress, Result } from "./api";
import { SAMPLES, SPACE_URL, check, health, onSpaceStatus } from "./api";
import { FaceRegions, Gauge, HeroArt, Logo, People, ScoreBar, Spinner, StepIcon, SureIcon, Timeline, UploadIcon } from "./art";

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

  const sample = async (url: string, filename: string) => {
    setStage("uploading");
    setStep("upload");
    setProg(0);
    try {
      const b = await (await fetch(url)).blob();
      start(b, filename);
    } catch {
      setErr("Could not download the sample video.");
      setStage("error");
    }
  };

  const busy = stage === "uploading" || stage === "working";

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-5 py-4">
          <a href="/" className="flex items-center gap-3">
            <Logo />
            <span className="text-lg font-bold text-slate-900">Deepfake Video Checker</span>
          </a>
          <ServerBadge h={h} />
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-5">
        <section className="grid items-center gap-10 py-12 md:grid-cols-2">
          <div>
            <h1 className="text-4xl font-extrabold leading-tight tracking-tight text-slate-900 md:text-5xl">
              Is this video <span className="text-real">real</span> or a <span className="text-fake">deepfake</span>?
            </h1>
            <p className="mt-4 text-lg text-slate-600">
              Upload a short video of a person's face. In about a minute you get an answer, a simple reason for it,
              and an honest note when the checker is not sure.
            </p>
            <ul className="mt-6 space-y-2 text-slate-600">
              <li className="flex gap-2"><Check /> Works on face-swap and face-reenactment fakes</li>
              <li className="flex gap-2"><Check /> Checks every person in the video, not just one</li>
              <li className="flex gap-2"><Check /> Corrects for blurry or heavily compressed videos</li>
              <li className="flex gap-2"><Check /> Your video is deleted right after the check</li>
            </ul>
          </div>
          <HeroArt className="w-full max-w-md justify-self-center" />
        </section>

        <section className="card p-6 md:p-8">
          <div
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); if (!busy) onFiles(e.dataTransfer.files); }}
            className={`flex flex-col items-center rounded-2xl border-2 border-dashed px-6 py-10 text-center transition
              ${drag ? "border-brand bg-brand-soft" : "border-slate-300 bg-slate-50"}`}>
            <UploadIcon className="h-14 w-14 text-brand" />
            <p className="mt-4 text-lg font-semibold text-slate-900">Drop a video here</p>
            <p className="mt-1 text-sm text-slate-500">MP4, MOV, AVI, MKV or WEBM · at least 4 seconds · one or two clear faces · up to {MAX_MB} MB</p>
            <div className="mt-6 flex flex-wrap justify-center gap-3">
              <button className="btn-primary" disabled={busy} onClick={() => input.current?.click()}>
                Choose a video
              </button>
              {SAMPLES.map((s) => (
                <button key={s.id} className="btn-ghost" disabled={busy} onClick={() => sample(s.url, s.name)}>
                  {s.label}
                </button>
              ))}
            </div>
            <input ref={input} type="file" accept="video/*" className="hidden" onChange={(e) => { onFiles(e.target.files); e.target.value = ""; }} />
          </div>

          <div className="mt-6">
            <div className="label">Optional: test it on a low-quality copy</div>
            <div className="mt-2 flex flex-wrap gap-2">
              {[["none", "Use my video as it is"], ["blur", "Add strong blur"], ["compress", "Add heavy compression"]].map(([v, l]) => (
                <button key={v} disabled={busy} onClick={() => setDegrade(v)} className={`pill ${degrade === v ? "pill-on" : "pill-off"}`}>{l}</button>
              ))}
            </div>
            <p className="mt-2 text-sm text-slate-500">Low quality is where the original detector makes most mistakes. Try it on the real sample.</p>
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
          <section ref={progressRef} className="card mt-8 scroll-mt-6 p-6 md:p-8">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-slate-900">{stage === "error" ? "Something went wrong" : "Checking your video"}</h2>
              {name && <span className="truncate pl-4 text-sm text-slate-500">{name}</span>}
            </div>
            {stage === "error" ? (
              <div className="mt-4 rounded-xl bg-fake-soft px-4 py-3 text-fake">{err}</div>
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
                          <div className={`font-semibold ${state === "later" ? "text-slate-400" : "text-slate-900"}`}>{s.title}</div>
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

        {stage === "done" && res && <ResultView r={res} preview={preview} refEl={resultRef} />}

        <HowItWorks />
      </main>

      <footer className="mt-16 border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-5xl px-5 py-8 text-sm text-slate-500">
          <p>A B.Tech research project by Roshan Raj. It uses the DFD-FCG detector (CVPR 2025) with a quality correction.
            It gives a strong hint, not proof; results are most reliable on talking-head videos similar to FaceForensics++.</p>
          <p className="mt-2 flex gap-4"><a className="text-brand hover:underline" href="https://github.com/roshanraj9136/dfd-fcg-degrade-to-detect">Source code</a>
            <a className="text-brand hover:underline" href={SPACE_URL}>Checker on Hugging Face</a></p>
        </div>
      </footer>
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

function ServerBadge({ h }: { h: Health | undefined }) {
  const ok = h?.status === "ready";
  return (
    <span className={`hidden items-center gap-2 rounded-full px-3 py-1 text-xs font-medium sm:inline-flex
      ${ok ? "bg-real-soft text-real" : "bg-amber-50 text-amber-700"}`}>
      <span className={`h-2 w-2 rounded-full ${ok ? "bg-real" : "bg-amber-500"}`} />
      {ok ? `Checker online${h?.analysed_total ? ` · ${h.analysed_total} videos checked` : ""}` : "Connecting…"}
    </span>
  );
}

const REGION_WORDS: Record<string, string> = { eyes: "the eyes", nose: "the nose", lips: "the mouth", skin: "the skin and cheeks" };

function ResultView({ r, preview, refEl }: { r: Result; preview: string; refEl: React.RefObject<HTMLDivElement | null> }) {
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
    <section ref={refEl} className="mt-8 scroll-mt-6 space-y-6">
      <div className={`card overflow-hidden ring-2 ${tone}`}>
        <div className={`grid items-center gap-6 p-6 md:grid-cols-[auto_1fr_auto] md:p-8 ${bg}`}>
          <Gauge p={r.p} verdict={r.verdict} />
          <div>
            <div className="flex items-center gap-3">
              <span className="label">Result</span>
              <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold
                ${sure ? "bg-white text-slate-700" : "bg-amber-100 text-amber-800"}`}>
                <SureIcon sure={sure} className="h-4 w-4" />{sure ? "Sure" : "Not sure"}
              </span>
            </div>
            <h2 className={`mt-1 text-3xl font-extrabold ${!sure ? "text-amber-700" : fake ? "text-fake" : "text-real"}`}>
              {sure ? (fake ? "This looks like a deepfake" : "This looks like a real video")
                : (fake ? "Probably a deepfake, but not sure" : "Probably real, but not sure")}
            </h2>
            <p className="mt-2 text-slate-600">
              {fake ? "The face shows signs of manipulation that the detector has learned from known deepfakes."
                : "The face and its movement look like an authentic recording."}
              {!sure && " The score is close to the line. On our test videos such answers were wrong much more often, so treat this as a hint and check the video another way too."}
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
    <section className="mt-16">
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
