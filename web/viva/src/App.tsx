import { useState } from "react";
import type { ReactNode } from "react";
import ensemble from "./ensemble.json";
import retrain from "./retrain.json";
import scores from "./scores.json";
import { BlurTestFig, FixFig, ModelArchFig, ScoreStrips } from "./figs";
import { DeploymentFlow, ExtraWork, LiveArchitecture, Source } from "./Evidence";
import { ArchitectureHero, PaperMap, TensorArchitecture } from "./TensorArchitecture";

const LINKS = {
  checker: "https://dfd-fcg-degrade-to-detect.vercel.app",
  github: "https://github.com/roshanraj9136/dfd-fcg-degrade-to-detect",
  tables: "https://github.com/roshanraj9136/dfd-fcg-degrade-to-detect/blob/main/results/tables.md",
  paper: "https://github.com/aiiu-lab/DFD-FCG",
};

const NAV: [string, string][] = [
  ["final", "Architecture"], ["dimensions", "Dimensions"], ["novelty", "Novelty"], ["extra", "Extra work"],
  ["papers", "Papers"], ["results", "Results"], ["questions", "Viva Q&A"], ["research", "Research"],
];

/* ------------------------------------------------------------------ data (all numbers come from these files) */
type S = Record<string, { real: number[]; fake: number[]; real_fix: number[]; fake_fix: number[] }>;
const SC = scores as S;
type EM = { n?: number; acc?: number; false_alarm?: number; caught?: number; bacc?: number; auroc?: number };
const EN = ensemble as unknown as {
  chosen: string; members: string[]; n_val: number;
  options: Record<string, Record<string, EM> & { mean_new_bacc: number; cv_val_bacc?: number }>;
  selfcal: Record<string, Record<string, { raw: EM; cal: EM }>>;
  decision: { gain_new_sets: number; change_ffpp: number; change_dfdc: number; deploy: boolean; live?: string; live_reason?: string };
  ci95: { new_sets: number[]; ffpp: number[]; dfdc: number[] };
  checker: {
    bands: { lo: number; hi: number; acc: number | null; share: number; n: number }[];
    overall_acc: number; overall_acc_original: number; overall_videos: number; videos: number;
    new_bacc: number; new_bacc_original: number; new_videos: number;
  };
};
const RT = retrain as unknown as { decision: { gain_new_sets: number; change_ffpp: number; change_dfdc: number } };
const CK = EN.checker;
const BASE = EN.options.dfd, FIN = EN.options[EN.chosen];
const GSC = EN.selfcal.pe ?? EN.selfcal[Object.keys(EN.selfcal)[0]];
const pc = (v?: number | null) => (v == null || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(1)}%`);
const p0 = (v?: number | null) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const pts = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(1)}`;
const range = (r: number[]) => `${pts(r[0])} to ${pts(r[1])}`;

export default function App() {
  return (
    <div className="min-h-screen">
      <a className="skip-link" href="#content">Skip to content</a>
      <Header />
      <main id="content" className="mx-auto max-w-6xl px-4 sm:px-5">
        <ArchitectureHero />
        <Story />
        <Final />
        <TensorArchitecture />
        <Novelty />
        <ExtraWork />
        <PaperMap />
        <Results />
        <Proof />
        <Questions />
        <Script />
        <div id="research" className="scroll-mt-24 border-t-2 border-slate-200 pt-12">
          <p className="font-semibold text-brand">The research behind the checker</p>
          <h2 className="mt-2 text-3xl font-extrabold tracking-tight">Go deeper into the evidence.</h2>
          <p className="mt-3 max-w-3xl text-slate-600">The original experiments explain why we added self-calibration. These use the research protocol; the live architecture and its dimensions are described above.</p>
          <div className="mt-4 flex flex-wrap gap-3 text-sm font-semibold text-brand"><a href="#model">Spatial & temporal</a><a href="#problem">Failure under blur</a><a href="#heatmap">Heatmaps</a><a href="#fix">Fitting the correction</a></div>
        </div>
        <Model />
        <Problem />
        <Cause />
        <Heatmap />
        <Fix />
      </main>
      <footer className="mt-10 border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-6xl px-5 py-8 text-sm text-slate-600">
          B.Tech project by Roshan Raj. Builds on DFD-FCG by Han, Huang, Hua and Chen (CVPR 2025) and GenD by Yermakov et al. (WACV 2026).
          Measurements link to result files; architecture dimensions link to the implementation. <a className="text-brand underline" href={LINKS.github}>Code and evidence</a> · <a className="text-brand underline" href="/architecture/architecture-explained.pdf">Architecture PDF</a> · <a className="text-brand underline" href="#top">Back to top</a>.
        </div>
      </footer>
    </div>
  );
}

/* ------------------------------------------------------------------ layout helpers */
function Header() {
  const [open, setOpen] = useState(false);
  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/95 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-5 py-3">
        <a href="#top" className="flex shrink-0 items-center gap-2 whitespace-nowrap font-bold text-slate-900">
          <img src="/favicon.svg" alt="" className="h-7 w-7" /><span className="hidden sm:inline">Degrade to Detect</span><span className="sm:hidden">DFD viva</span>
        </a>
        <nav className="hidden gap-0.5 xl:flex">
          {NAV.map(([id, label]) => <a key={id} href={`#${id}`} className="whitespace-nowrap rounded-lg px-2 py-1.5 text-sm text-slate-600 hover:bg-slate-100 hover:text-slate-900">{label}</a>)}
        </nav>
        <button aria-expanded={open} aria-controls="section-menu" className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm xl:hidden" onClick={() => setOpen(!open)}>Sections</button>
      </div>
      {open && (
        <nav id="section-menu" className="grid grid-cols-2 gap-1 border-t border-slate-200 px-5 py-3 xl:hidden">
          {NAV.map(([id, label]) => <a key={id} href={`#${id}`} onClick={() => setOpen(false)} className="rounded-lg px-2 py-1.5 text-sm text-slate-700 hover:bg-slate-100">{label}</a>)}
        </nav>
      )}
    </header>
  );
}

function Section({ id, n, title, lead, children, tone }: { id: string; n: string; title: ReactNode; lead?: ReactNode; children: ReactNode; tone?: "red" | "green" | "brand" | "amber" }) {
  const k = tone === "red" ? "text-fake" : tone === "green" ? "text-real" : tone === "amber" ? "text-amber-700" : "text-brand";
  return (
    <section id={id} className="scroll-mt-20 py-12">
      <div className={`text-sm font-semibold ${k}`}>{n}</div>
      <h2 className="mt-1 text-2xl font-extrabold tracking-tight text-slate-900 md:text-3xl">{title}</h2>
      {lead && <p className="mt-3 max-w-3xl text-slate-600">{lead}</p>}
      <div className="mt-6">{children}</div>
    </section>
  );
}

function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  const bg = /(^|\s)bg-/.test(className) ? "" : "bg-white";
  const border = /(^|\s)border-(brand|fake|real|[a-z]+-\d)/.test(className) ? "" : "border-slate-200";
  return <div className={`rounded-3xl border ${border} ${bg} p-5 shadow-sm md:p-6 ${className}`}>{children}</div>;
}

function Fig({ children, caption }: { children: ReactNode; caption?: ReactNode }) {
  return (
    <Card className="p-4 md:p-5">
      <div className="overflow-x-auto">{children}</div>
      {caption && <p className="mt-3 text-sm text-slate-500">{caption}</p>}
    </Card>
  );
}

function Say({ children }: { children: ReactNode }) {
  return (
    <div className="mt-6 rounded-2xl border-l-4 border-brand bg-brand-soft px-5 py-4 text-slate-800">
        <div className="text-sm font-bold text-brand">Say it like this</div>
      <p className="mt-1">{children}</p>
    </div>
  );
}

function Big({ k, v, text, tone = "ink" }: { k: string; v: ReactNode; text: ReactNode; tone?: "ink" | "fake" | "real" | "brand" | "amber" }) {
  const c = { ink: "text-slate-900", fake: "text-fake", real: "text-real", brand: "text-brand", amber: "text-amber-700" }[tone];
  return (
    <Card>
      <div className="text-xs font-bold uppercase tracking-wider text-slate-500">{k}</div>
      <div className={`mt-1 text-3xl font-extrabold ${c}`}>{v}</div>
      <p className="mt-1 text-sm text-slate-600">{text}</p>
    </Card>
  );
}

function Table({ head, rows, note }: { head: string[]; rows: ReactNode[][]; note?: ReactNode }) {
  return (
    <Card className="overflow-x-auto p-0">
      <table className="w-full min-w-[560px] text-sm">
        <thead>
          <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs uppercase tracking-wider text-slate-500">
            {head.map((h, i) => <th key={h} className={`px-4 py-3 ${i ? "text-right" : ""}`}>{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-slate-100">
              {r.map((c, j) => <td key={j} className={`px-4 py-2 ${j ? "text-right tabular-nums" : "font-medium text-slate-900"}`}>{c}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      {note && <p className="px-4 pb-4 pt-2 text-xs text-slate-500">{note}</p>}
    </Card>
  );
}

/** "a → b", b coloured green when better, red when worse */
function Ch({ a, b, lowerBetter = false, f = pc }: { a?: number | null; b?: number | null; lowerBetter?: boolean; f?: (v?: number | null) => string }) {
  const good = a != null && b != null && (lowerBetter ? b < a - 1e-9 : b > a + 1e-9);
  const bad = a != null && b != null && (lowerBetter ? b > a + 1e-9 : b < a - 1e-9);
  return <span>{f(a)} → <b className={good ? "text-real" : bad ? "text-fake" : "text-slate-900"}>{f(b)}</b></span>;
}

function Steps({ items }: { items: [string, ReactNode][] }) {
  return (
    <ol className="grid gap-3 md:grid-cols-3">
      {items.map(([t, x], i) => (
        <li key={t} className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-brand text-sm font-bold text-white">{i + 1}</div>
          <div className="mt-2 font-semibold text-slate-900">{t}</div>
          <p className="mt-1 text-sm text-slate-600">{x}</p>
        </li>
      ))}
    </ol>
  );
}

/* ------------------------------------------------------------------ sections */
function Story() {
  return (
    <Section id="story" n="Start here" title="The problem we are trying to solve">
      <div className="grid gap-6 md:grid-cols-3">
        <div><h3 className="text-lg font-bold">Low quality can look like a fake.</h3><p className="mt-2 text-slate-600">Blur and compression change a detector’s internal evidence. A genuine video can cross the FAKE threshold even when it was never manipulated.</p></div>
        <div><h3 className="text-lg font-bold">Estimate the quality-related shift.</h3><p className="mt-2 text-slate-600">Our small fitted read-out uses the detector’s own features to predict a score correction. The deployed DFD-FCG weights stay frozen.</p></div>
        <div><h3 className="text-lg font-bold">Combine complementary scores.</h3><p className="mt-2 text-slate-600">DFD reads clips; GenD reads frames. We adapt one GenD model and learn how to combine five scores into a practical video checker.</p></div>
      </div>
      <Card className="mt-4">
        <div className="text-xs font-bold uppercase tracking-wider text-slate-500">Words I use</div>
        <dl className="mt-3 grid gap-x-8 gap-y-2 text-sm md:grid-cols-3">
          {[
            ["Fake score", "a model estimate from 0 to 1, not a guarantee. 0.5 or more = FAKE."],
            ["False alarm", "a real video called FAKE."],
            ["Fakes caught", "the share of deepfakes called FAKE."],
            ["Balanced accuracy", "average of “reals called REAL” and “fakes called FAKE”, so both count equally."],
            ["AUROC", "chance a random fake scores above a random real. Ignores the 0.5 line. 0.5 = coin flip."],
            ["Damage", "7 degradation types. The study tests 13 damaged conditions, plus clean and a re-encode control."],
            ["Spatial", "looking at one frame: texture, edges, eyes, lips."],
            ["Temporal", "looking at how the face changes over several frames: flicker, unnatural motion."],
            ["Self-calibration", "a fitted add-on that estimates and subtracts the score shift associated with low quality."],
          ].map(([a, b]) => <div key={a}><dt className="font-semibold text-slate-900">{a}</dt><dd className="text-slate-600">{b}</dd></div>)}
        </dl>
      </Card>
    </Section>
  );
}

function Model() {
  const rows: [string, string, string, string][] = [
    ["Real videos (70)", "93%", "87%", "97%"],
    ["Fake videos (60)", "95%", "98%", "98%"],
    ["FaceShifter fakes (15), never trained on", "40%", "80%", "73%"],
  ];
  return (
    <Section id="model" n="How the detector sees a video" title="Spatial and temporal: two views of the same face"
      lead={<>DFD-FCG cuts out the face, takes 10 frames from a 3-second clip, and reads them with CLIP (frozen). Its side network then asks
        two questions and gives three scores, which are averaged.</>}>
      <div className="grid gap-4 md:grid-cols-3">
        <Card><div className="text-xs font-bold uppercase tracking-wider text-brand">Spatial head</div>
          <div className="mt-1 font-bold text-slate-900">“Does any single frame look wrong?”</div>
          <p className="mt-1 text-sm text-slate-600">4 learned queries look at eyes, nose, lips and skin on each frame's 16×16 patches: blending edges, odd texture.</p></Card>
        <Card><div className="text-xs font-bold uppercase tracking-wider text-brand">Temporal head</div>
          <div className="mt-1 font-bold text-slate-900">“Does the face move naturally?”</div>
          <p className="mt-1 text-sm text-slate-600">Reads how patch features change over 10 frames. It can use temporal patterns; it does not explicitly measure lip-sync or check audio.</p></Card>
        <Card><div className="text-xs font-bold uppercase tracking-wider text-brand">Fusion head</div>
          <div className="mt-1 font-bold text-slate-900">“Both together”</div>
          <p className="mt-1 text-sm text-slate-600">Combines both feature vectors. Convert each corrected head logit to a probability, then average. In the live checker this is one of five combiner inputs.</p></Card>
      </div>
      <div className="mt-4"><Fig caption="Grey = authors' model, unchanged. Blue = our additions. d_s, d_t, d_a are head logits; g is an estimated quality-related shift. Sigmoid converts each corrected logit to a probability before averaging. This diagram is the DFD-FCG branch only.">
        <ModelArchFig />
      </Fig></div>
      <h3 className="mt-8 text-lg font-bold text-slate-900">Why both views matter (my measurement, clean FF++ test videos)</h3>
      <div className="mt-2">
        <Table head={["Right answers", "Spatial head alone", "Temporal head alone", "All heads (released)"]} rows={rows}
          note="Spatial alone misses most FaceShifter fakes (40%); motion catches 80%. Each head's score is read at the 0.5 line. Source: results/tables.md." />
      </div>
      <Card className="mt-4">
        <div className="font-bold text-slate-900">The blur bias is in both views</div>
        <p className="mt-1 text-sm text-slate-600">A 2-pixel blur moves real faces towards FAKE by <b>+1.13</b> in the spatial head, <b>+1.11</b> in the temporal head and
          <b> +1.44</b> in the fused head. So the fix must correct all three heads, and it does: the read-out predicts one shift per head.
          In the final checker, DFD-FCG supplies both views from clips, and GenD adds a second, stronger spatial view from 24 single frames.</p>
      </Card>
      <Say>DFD-FCG looks at a face two ways: spatially, frame by frame, and temporally, how it moves over 3 seconds. They catch different
        fakes: on FaceShifter, motion catches 80% and single frames only 40%. Blur fools both views, so my fix corrects all three heads.</Say>
    </Section>
  );
}

function Problem() {
  const rows = [
    { label: "Clean video", sub: "as recorded (FF++ c23)", real: SC.clean.real, fake: SC.clean.fake },
    { label: "Strong blur", sub: "authors' protocol, level 5", real: SC.GB_5.real, fake: SC.GB_5.fake },
    { label: "With my fix", sub: "same blurred videos", real: SC.GB_5.real_fix, fake: SC.GB_5.fake_fix, mine: true },
  ];
  return (
    <Section id="problem" n="What the paper missed" tone="red" title={<>The ranking holds, but the <span className="text-fake">decision breaks</span></>}
      lead={<>The paper reports robustness as AUROC, which stays high. I re-ran the authors' own damage code on 130 FF++ test videos and
        looked at the actual REAL / FAKE answers a user would see.</>}>
      <Fig caption={<>Each dot is one test video (70 real, 60 fake). Under strong blur all scores slide right: the order is still decent (AUROC 0.905),
        but most real videos cross the line. With my fix they move back.</>}>
        <ScoreStrips rows={rows} />
      </Fig>
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <Big k="What the paper sees" v="AUROC 0.999 → 0.946" text="clean → average of 13 damaged versions. Looks robust." />
        <Big k="What a user gets" tone="fake" v="2.9% → 22.5%" text="false alarms, clean → the same 13 damaged versions. Strong blur 63%, strong down/up-sampling 59%, strong H.264 33%." />
      </div>
      <Say>The paper's numbers are correct, but AUROC only checks the order of scores. At the model's own 0.5 line, strong blur turns 63% of
        real videos into “fakes”.</Say>
    </Section>
  );
}

function Cause() {
  return (
    <Section id="cause" n="Why it happens" title="The model uses fine detail as proof of “real”"
      lead={<>I blurred the aligned face by just 2 pixels and ran the model again. Real faces look much more fake; fakes hardly change.</>}>
      <Fig><BlurTestFig /></Fig>
      <div className="mt-4 grid gap-3 text-sm text-slate-600 md:grid-cols-3">
        <Card><b className="text-slate-900">Real faces have camera detail.</b> Skin texture and sharp edges are what DFD-FCG reads as “real”.
          The blur reaction alone separates real from fake on clean video (AUROC 0.992).</Card>
        <Card><b className="text-slate-900">Damage removes that detail.</b> Blur, resizing and compression wipe it out, so real videos lose their
          “proof” and drift to FAKE. Fakes have less detail to lose.</Card>
        <Card><b className="text-slate-900">Exact score accounting.</b> There are 240 terms across the three heads: 96 spatial, 24 temporal and 120 fused.
          Each head's terms plus its constant reconstruct its logit within 0.0000003 on 5,976 clips. This measures where scores shift; the fine-detail explanation is an interpretation of those observations.</Card>
      </div>
      <Say>In this study, real faces reacted much more to a small blur on average. That supports the explanation that DFD-FCG uses fine detail as evidence of “real”. It is not a rule that every real or fake video must follow.</Say>
    </Section>
  );
}

function Heatmap() {
  return (
    <Section id="heatmap" n="The heatmap" title="Yes, we use a heatmap, and it is exact"
      lead={<>Every answer on the live checker shows where on the face the evidence came from. It is built from the same exact split, not from
        Grad-CAM.</>}>
      <div className="grid items-start gap-4 md:grid-cols-[1fr_1.2fr]">
        <Card className="p-4">
          <div className="grid grid-cols-2 gap-3">
            <figure><img src="/heat_face.jpg" alt="aligned face crop" className="w-full rounded-2xl" />
              <figcaption className="mt-1 text-center text-xs text-slate-500">aligned face</figcaption></figure>
            <figure><img src="/heat_map.jpg" alt="evidence heatmap" className="w-full rounded-2xl" />
              <figcaption className="mt-1 text-center text-xs text-slate-500"><b className="text-fake">red</b> = towards FAKE, <b className="text-real">green</b> = towards REAL</figcaption></figure>
          </div>
          <p className="mt-3 text-xs text-slate-500">A fake test video on the live checker: the strongest fake evidence came from the eyes, the strongest real evidence from the nose.</p>
        </Card>
        <div className="grid gap-3">
          <Card><div className="font-bold text-slate-900">How it is made</div>
            <p className="mt-1 text-sm text-slate-600">For a fixed input, the LayerNorm and linear head can be decomposed into patch contributions.
              The patch terms <b>plus a constant</b> reconstruct the spatial head's raw logit, not the final probability. The displayed overlay averages contributions over the sampled clips.</p></Card>
          <Card><div className="font-bold text-slate-900">Proof that it is meaningful</div>
            <p className="mt-1 text-sm text-slate-600">From the explanation pieces alone I could tell which of 4 fake methods made a video <b>77%</b> of the time,
              against <b>57%</b> from the 3 head scores and 25% by chance (79 fake videos). On clean FF++, real videos get most “real” evidence from the
              nose (49%), fakes most “fake” evidence from the eyes.</p></Card>
          <Card className="border-amber-200 bg-amber-50"><div className="font-bold text-amber-800">What it is not</div>
            <p className="mt-1 text-sm text-slate-700">It explains DFD-FCG's spatial head on the deciding face. It does not decide the answer, and the GenD
              detector has no heatmap. Use it to show <i>where</i>, not as proof that a video is fake.</p></Card>
        </div>
      </div>
      <Say>The patch contributions plus a constant reconstruct DFD-FCG's spatial logit. The heatmap helps inspect that branch; it is not an explanation of the complete five-score verdict or proof of manipulation.</Say>
    </Section>
  );
}

function Fix() {
  return (
    <Section id="fix" n="My fix" tone="brand" title="Self-calibration: the model corrects its own quality bias"
      lead={<>The model's own features already show how damaged a face is. A small read-out turns that into an estimate of the fake evidence the
        damage added, and subtracts it.</>}>
      <Fig><FixFig /></Fig>
      <ul className="mt-4 grid gap-2 text-sm text-slate-600 md:grid-cols-2">
        <li>• <b className="text-slate-900">Ridge regression</b> from the heads' 1,280 feature numbers to 3 shifts (spatial, temporal, fused). The read-out is fitted; the detector weights stay frozen.</li>
        <li>• <b className="text-slate-900">Target from real videos only</b>: how much real scores moved under damage. Real and fake clips get the same target,
          for each condition. It is not trained to predict real/fake labels, but its correction can still lower fake scores.</li>
        <li>• <b className="text-slate-900">Fitted on 49 training videos</b> (30 real, 19 fake) at damage levels 2 and 4; tested on 130 test videos at levels 1, 3, 5.</li>
        <li>• <b className="text-slate-900">Tested on damage it never saw</b>: each damage type is left out of fitting in turn. No quality label needed.</li>
      </ul>
      <div className="mt-6">
        <Table head={["FF++ test, damage type unseen", "DFD-FCG as released", "With my fix"]} rows={[
          ["Real called FAKE, damaged video (mean)", "22.5%", <b className="text-real">8.7%</b>],
          ["Real called FAKE, strong blur", "63%", <b className="text-real">11%</b>],
          ["Balanced accuracy, damaged (mean)", "82.7%", <b className="text-real">84.7%</b>],
          ["Fakes caught at 5% false alarms", "67.2%", <b className="text-real">71.0%</b>],
          ["AUROC, all qualities mixed", "0.920", <b className="text-real">0.951</b>],
          ["Fakes caught at the 0.5 line", "87.9%", <b className="text-fake">78.1%</b>],
        ]} note={<>Last row is the cost: fewer fakes are caught at the default line. Noise (which hides fakes) is not fixed. All rows:
          {" "}<a className="text-brand underline" href={LINKS.tables}>results/tables.md</a>.</>} />
      </div>
      <Say>I keep the detector weights frozen. A fitted linear read-out estimates the score shift associated with poor quality from the detector's own features,
        and subtracts it. False alarms on damaged video drop from 22.5% to 8.7%, even for damage types it never saw.</Say>
    </Section>
  );
}

/* ------------------------------------------------------------------ the final, live system */
const SETS: [string, string, "bacc" | "caught"][] = [
  ["celebdf", "Celeb-DF v2 (2020)", "bacc"], ["social", "Social-media deepfakes (2026)", "bacc"],
  ["mavos", "MAVOS-DD, methods seen in training (2025)", "bacc"], ["mavos_open", "MAVOS-DD, methods never seen (2025)", "bacc"],
  ["sf", "SynthForensics, AI-made people (2026) · fakes caught", "caught"], ["synth", "AI talking heads (2026) · fakes caught", "caught"],
  ["ffpp", "FF++ test (clean, blur, compression)", "bacc"], ["fsh", "FaceShifter, never trained on", "bacc"], ["dfdc", "DFDC, never trained on", "bacc"],
];

function Final() {
  return (
    <Section id="final" n="The architecture" tone="brand" title="One video → three networks → five scores → one answer"
      lead={<>DFD-FCG reads short video clips. Two versions of GenD-PE read individual frames: the released model and our fine-tuned model.
        A small learned combiner brings their evidence together. Open a path below for the technical detail.</>}>
      <LiveArchitecture />
      <DeploymentFlow />
      <Say>We cut out and align the faces. DFD-FCG checks appearance and changes across frames. Two GenD models check individual frames.
        DFD-FCG gives one corrected score; each GenD gives a raw and a corrected score. That makes five scores from three networks.
        A learned combiner gives each face a fake probability, and the most suspicious tracked face decides the video verdict.</Say>
      <details className="mt-6 rounded-2xl border border-slate-200 bg-white p-5">
      <summary className="cursor-pointer font-semibold text-brand">Why we built this combination</summary>
      <div className="mt-4">
      <Steps items={[
        ["Retrain DFD-FCG alone", <>On 2025–26 data: {pts(RT.decision.gain_new_sets)} points on newer fakes, but {pts(RT.decision.change_dfdc)} on DFDC.
          It failed my pre-written rule, so I did not use it.</>],
        ["Add a frame detector", <>GenD (2026) fails the same way: strong blur made it call <b>{p0(GSC["ffpp:GB_5"].raw.false_alarm)}</b> of real FF++ videos FAKE.
          The same recipe, fitted on GenD features, cut that to <b>{p0(GSC["ffpp:GB_5"].cal.false_alarm)}</b>, with lower fake recall.</>],
        ["Fine-tune GenD, combine", <>Fine-tuning the LayerNorm parameters and classifier head raised validation AUROC from about 0.72 to 0.92.
          The combiner uses {EN.n_val} validation videos. The evidence section explains the protocol's limits.</>],
      ]} />
      </div>
      </details>
    </Section>
  );
}

function Results() {
  const d = EN.decision;
  const b = CK.bands;
  return (
    <Section id="results" n="What the measurements show" title="The gains—and the cost" lead="These are stored benchmark results. Read the baseline, evaluation settings and limitations alongside the headline numbers.">
      <div className="mt-6 grid gap-4 md:grid-cols-3">
        <Big k={`${CK.overall_videos} videos · published comparison`} tone="real" v={<>{pc(CK.overall_acc_original)} → {pc(CK.overall_acc)}</>} text="correct answers: DFD-FCG + self-calibration → chosen five-score fusion" />
        <Big k="Four two-class benchmarks" tone="real" v={<>{pc(BASE.mean_new_bacc)} → {pc(FIN.mean_new_bacc)}</>}
          text={<>mean balanced accuracy: Celeb-DF, social media and two MAVOS splits; 95% gain range {range(EN.ci95.new_sets)} points</>} />
        <Big k="Old FF++ (2019)" tone="fake" v={<>{pc(BASE.ffpp?.bacc)} → {pc(FIN.ffpp?.bacc)}</>}
          text={<>the price: false alarms on clean FF++ reals {p0(BASE["ffpp:clean"]?.false_alarm)} → {p0(FIN["ffpp:clean"]?.false_alarm)}</>} />
      </div>
      <details className="mt-6">
        <summary className="mb-3 cursor-pointer font-semibold text-brand">See the results for each dataset</summary>
        <Table head={["Test set", "DFD-FCG + self-cal. → chosen fusion"]}
          rows={SETS.map(([g, name, k]) => [name, <Ch a={BASE[g]?.[k]} b={FIN[g]?.[k]} />])}
          note="Balanced accuracy at 0.5; fake-only sets show recall, not two-class accuracy. FF++, FaceShifter and DFDC rows average clean / strong blur / heavy compression. The 705-video headline counts unique videos without added damage; FaceShifter reuses FF++ reals. These are different summaries." />
      </details>
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <Card><div className="font-bold text-slate-900">What the confidence number means</div>
          <p className="mt-1 text-sm text-slate-600">On the {CK.overall_videos} test videos: answers at 80%+ confidence were right {pc(b[2]?.acc)} of the time ({p0(b[2]?.share)} of answers),
            65–80% right {pc(b[1]?.acc)}, under 65% right {pc(b[0]?.acc)}. These are observed bands, not proof that the confidence is calibrated for new uploads.</p></Card>
        <Card className="border-amber-200 bg-amber-50"><div className="font-bold text-amber-800">Honest: my own rule was not met</div>
          <p className="mt-1 text-sm text-slate-700">Rule written before testing: switch only if newer sets gain ≥ 5 points and FF++ and DFDC lose ≤ 2.
            Measured: newer {pts(d.gain_new_sets)}, FF++ {pts(d.change_ffpp)}, DFDC {pts(d.change_dfdc)}. I deployed it anyway because people upload modern
            videos, and I report the FF++ loss.</p></Card>
      </div>
      <p className="mt-4 text-sm text-slate-600"><b>Read the comparison correctly:</b> its DFD baseline already includes our self-calibration.
        The FF++ / FaceShifter / DFDC results use cached two-clip DFD features; production defaults to eight clips. These published offline results are not a fresh end-to-end measurement of every current upload setting.
        Sources: <Source path="service/ens/data.py">score loading and clip protocol</Source>, <Source path="results/ensemble/results.json">results</Source> and <Source path="results/ensemble/PLAN.md">deployment decision</Source>.</p>
      <Say>The live checker combines five scores from three networks. In the published {CK.overall_videos}-video comparison, the chosen fusion reaches
        {" "}{pc(CK.overall_acc)} accuracy against {pc(CK.overall_acc_original)} for DFD-FCG with self-calibration. It is a trade-off: on 2019 FF++ videos it
        loses {pts(-d.change_ffpp).replace("+", "")} points, which broke my own pre-written rule, and I say so.</Say>
    </Section>
  );
}

/* ------------------------------------------------------------------ why the results can be trusted */
function Proof() {
  const items: [string, ReactNode][] = [
    ["Separate fit and evaluation videos", <>The read-out uses 49 train-split videos. The deployed combiner uses {EN.n_val} validation videos.
      The early study included test-pilot choices, and later comparisons reused known benchmarks; see the qualifications below.</>],
    ["Rules written before testing", "Switch rules were saved in results/ensemble/PLAN.md before any test run. Both failures (round 1 and the FF++ loss) are reported."],
    ["Unseen damage", "The fix is tested on damage types left out of fitting. A 5-fold check over test identities gives the same picture (7.8% false alarms)."],
    ["Beats simple baselines", "Same test, unseen damage: Platt re-calibration 12.2% false alarms, quality from the face crop 15.1%, blur test 18.1%, mine 8.7%."],
    ["Exact explanation", "The 240 terms across all heads, plus each head's constant, reconstruct their raw logits within 0.0000003 on 5,976 clips. This is score accounting, not causal proof."],
    ["Transfer to GenD-PE", <>The same recipe, fitted separately on GenD features: strong-blur false alarms {p0(GSC["ffpp:GB_5"].raw.false_alarm)} → {p0(GSC["ffpp:GB_5"].cal.false_alarm)}. Fake recall also falls, so this is a trade-off.</>],
    ["Implementation parity", "The recorded parity checks matched live and offline scoring on selected examples with matched inputs. They do not replace a full-video accuracy evaluation under all production settings (tests/check_fusion_live.py)."],
    ["Uncertainty shown", <>Bootstrap 95% ranges: gain on newer sets {range(EN.ci95.new_sets)} points. Single seed and 50–120 videos per set, so small differences are noise.</>],
  ];
  return (
    <Section id="proof" n="Why trust it" tone="green" title="How I know the results are correct"
      lead="Separate what the measurements establish from what still needs a fresh, independent evaluation.">
      <div className="grid gap-3 md:grid-cols-2">
        {items.map(([t, x]) => (
          <div key={t} className="flex gap-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-real-soft text-sm font-bold text-real">✓</span>
            <div><div className="font-semibold text-slate-900">{t}</div><p className="mt-0.5 text-sm text-slate-600">{x}</p></div>
          </div>
        ))}
      </div>
      <Card className="mt-4 border-amber-200 bg-amber-50">
        <div className="font-bold text-amber-800">Protocol qualifications to state clearly</div>
        <ul className="mt-2 space-y-2 text-sm text-slate-700">
          <li>The original ridge strength (α = 1000) and strict-mode weight (0.5) were chosen after a pilot on part of the test data. The identity cross-validation check also uses test-split data. Do not describe the whole project as untouched test evaluation.</li>
          <li>The original protocol used Counterfactual groups 25–30 for validation despite a broader reservation of groups 25–36. Later comparisons exclude all synth validation rows; that does not undo the earlier checkpoint selection.</li>
          <li>CLIP and paired-training follow-ups reuse the published benchmarks. Their frozen validation selections are documented, but the subsequent tests are not new blind holdouts.</li>
        </ul>
        <p className="mt-3 text-sm"><Source path="README.md#limitations">Original study limitations</Source> · <Source path="results/reliability_followup.md">Split clarification and follow-up record</Source></p>
      </Card>
      <Card className="mt-4 border-amber-200 bg-amber-50">
        <div className="font-bold text-amber-800">Where it still fails (say it before they ask)</div>
        <ul className="mt-2 grid gap-1.5 text-sm text-slate-700 md:grid-cols-2">
          <li>• DFDC, a dataset none of the detectors trained on: {pc(FIN.dfdc?.bacc)} balanced accuracy.</li>
          <li>• Clean 2019 FF++ reals: {p0(FIN["ffpp:clean"]?.false_alarm)} false alarms (DFD-FCG + self-calibration: {p0(BASE["ffpp:clean"]?.false_alarm)}).</li>
          <li>• Fully AI-made people: {pc(FIN.sf?.caught)} caught. Lip-sync-only fakes: no audio detector.</li>
          <li>• Noise hides fakes, and heavy blur removes the clues every detector needs.</li>
        </ul>
      </Card>
    </Section>
  );
}

const NEW: [string, ReactNode, string, ReactNode][] = [
  ["A failure the paper does not report", "63% of real videos called FAKE under strong blur, while AUROC still reads 0.905.",
    "Everyone knows compression hurts detectors.",
    "In general, yes. This paper reports the model as robust using AUROC. I show with the authors' own damage code that its decisions are not, and by how much."],
  ["Evidence of the mechanism", "A 2-px blur changes the fused logit by +1.44 on real faces and −0.17 on fakes on average. Exact attribution extends the earlier two-video work to 5,976 clips.",
    "Explanation maps are unreliable.",
    "The terms plus a constant reconstruct each head logit. That establishes mathematical fidelity to the score, not that the highlighted pixels caused a forgery."],
  ["A fitted correction with a frozen detector", "False alarms 22.5% → 8.7% on unseen damage types; mixed AUROC 0.920 → 0.951. Fake recall at 0.5 falls from 87.9% to 78.1%.",
    "Isn't this just moving the threshold?",
    "A global threshold or Platt mapping applies the same score-only rule to every video. Our read-out uses each video's internal features to estimate its own correction. Ridge regression itself is standard."],
  ["Transfer evidence on a second detector family",
    <>Released GenD-PE: {pc(GSC["ffpp:GB_5"].raw.false_alarm)} → {pc(GSC["ffpp:GB_5"].cal.false_alarm)} false alarms under blur; recall {pc(GSC["ffpp:GB_5"].raw.caught)} → {pc(GSC["ffpp:GB_5"].cal.caught)}.</>,
    "Your fix is tailored to DFD-FCG.",
    "The same fitting recipe works on GenD's class token with a separately fitted read-out. That is evidence of transfer to GenD, not proof it works on every detector."],
  ["Engineering: an evaluated public checker",
    <>{pc(CK.overall_acc_original)} → {pc(CK.overall_acc)} correct in the {CK.overall_videos}-video comparison, from self-calibrated DFD-FCG to the chosen fusion.</>,
    "Ensembles always help; you tuned on the test set.",
    <>The combiner was fitted and chosen on {EN.n_val} validation videos only, and the rule was written before testing. It is not free: FF++ lost
      {" "}{pts(-EN.decision.change_ffpp).replace("+", "")} points, and I report that.</>],
];

function Novelty() {
  return (
    <Section id="novelty" n="The contribution" tone="brand" title="What is new in our work?"
      lead={<>We studied a practical failure: a real face can be called fake just because the video quality drops. Our main contribution is measuring that failure,
        tracing the score changes inside the model, and fitting a feature-based correction while keeping the detector frozen.</>}>
      <div className="grid gap-4 md:grid-cols-3">
        <Card><h3 className="font-bold text-slate-900">What we started with</h3>
          <p className="mt-2 text-sm text-slate-600">The authors supplied DFD-FCG, GenD, checkpoints and degradation tools. Our earlier project already had exact attribution on two videos. We do not claim these as newly invented here.</p></Card>
        <Card className="border-indigo-200 bg-brand-soft"><h3 className="font-bold text-indigo-900">What this study adds</h3>
          <p className="mt-2 text-sm text-slate-700">A larger threshold-level robustness study, exact analysis of how evidence changes under damage, a fitted quality correction, and transfer experiments on GenD. The correction reduces false alarms, with a measured recall cost.</p></Card>
        <Card><h3 className="font-bold text-slate-900">What we built around it</h3>
          <p className="mt-2 text-sm text-slate-600">A GenD fine-tune, a five-score combiner, multi-face video processing, an explanatory checker, reliability checks and documented follow-up experiments. These are project and engineering contributions.</p></Card>
      </div>
      <Say>Our contribution is to separate “looks fake” from “looks low quality” in the detector's decision. We measured the problem, traced how the evidence changes,
        and fitted a small correction from the model's own features. Then we tested transfer and built a usable checker around it. We did not invent CLIP, GenD, ridge regression or ensembles.</Say>
      <p lang="hi-Latn" className="mt-4 border-l-4 border-teal-600 bg-teal-50 px-5 py-4 text-sm leading-relaxed text-teal-950"><b>Easy Hinglish:</b> Base model authors ka hai. Hamne dikhaya ki low-quality real video bhi fake lag sakta hai, andar ke score changes analyse kiye, aur unhi features se correction fit kiya. Phir GenD par test karke final checker banaya. Novelty is study aur correction ke evidence mein hai; sirf models jodna ek naya foundation model banana nahi hai.</p>
      <p className="mt-4 text-sm text-slate-600">This describes the contribution relative to the base models and our earlier project. A claim of being the first in the literature would need a broader literature review.
        Sources: <Source path="qa/engine.py">attribution extension</Source>, <Source path="qa/methods.py">correction</Source>, <Source path="results/tables.md">study results</Source>.</p>
      <details className="mt-5">
      <summary className="mb-3 cursor-pointer font-semibold text-brand">Evidence and answers for a deeper viva discussion</summary>
      <Card className="overflow-x-auto p-0">
        <table className="w-full min-w-[760px] text-left text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-xs uppercase tracking-wider text-slate-500">
              <th className="px-4 py-3">What is new</th><th className="px-3 py-3">Proof</th><th className="px-4 py-3">If they say… answer</th>
            </tr>
          </thead>
          <tbody>
            {NEW.map(([w, p, q, a]) => (
              <tr key={w} className="border-t border-slate-100 align-top">
                <td className="px-4 py-3 font-semibold text-slate-900">{w}</td>
                <td className="px-3 py-3 text-slate-700">{p}</td>
                <td className="px-4 py-3 text-slate-600"><b className="text-fake">“{q}”</b><br />{a}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <Card>
          <div className="font-bold text-slate-900">Not mine</div>
          <ul className="mt-2 space-y-1 text-sm text-slate-600">
            <li>• DFD-FCG and GenD: models, released weights, training code, and the damage code.</li>
            <li>• The datasets (FF++, Celeb-DF, DFDC, MAVOS-DD, SynthForensics and others).</li>
            <li>• Ridge regression, logistic regression and Platt scaling are standard tools.</li>
          </ul>
        </Card>
        <Card>
          <div className="font-bold text-slate-900">Ideas that failed (also reported)</div>
          <ul className="mt-2 space-y-1 text-sm text-slate-600">
            <li>• Quality from the background or the face crop; the blur test alone as the fix.</li>
            <li>• Retraining DFD-FCG alone (round 1): rule not met.</li>
            <li>• Mirrored clips and moving the 0.5 line: no test gain.</li>
          </ul>
        </Card>
      </div>
      </details>
    </Section>
  );
}

const QA: [string, ReactNode][] = [
  ["What is the exact input dimension?", "DFD-FCG receives [B, 10, 3, 224, 224]: batch of clips, ten frames, RGB channels, height and width. Each GenD-PE receives [M, 3, 224, 224], with up to 24 sampled frames. The aligned face crops start at 150 × 150 before model transforms."],
  ["Why 1024 spatial features and 256 temporal features?", "1024 is the CLIP ViT-L/14 hidden width. Spatial pooling keeps that width. The temporal branch ends with a 16 × 16 patch-response grid, flattened to 256 values. The numbers describe learned features, not the number of input frames."],
  ["Exactly how do you concatenate the vectors?", "Append the spatial vector to the temporal vector along the last axis: [B, 1024] + [B, 256] becomes [B, 1280]. The authors’ fused head concatenates LayerNorm outputs. Our quality read-out concatenates raw features and then applies fitted per-feature standardization."],
  ["What gets concatenated in the final ensemble?", "Five score logits, one per member, form a [5] vector for each face. We do not join the 1280-dimensional DFD features with the 1024-dimensional GenD features. The final logistic regression combines their scores."],
  ["Which papers should I name?", "DFD-FCG (Han et al., CVPR 2025) is the video detector; GenD (Yermakov et al., WACV 2026) is the frame detector. CLIP (Radford et al., ICML 2021) is DFD’s backbone; Perception Encoder (Bolya et al., 2025) is the backbone of GenD-PE. Paper links and the contribution of each are above."],
  ["What exactly is wrong in the paper?",
    "Its numbers are right; its robustness test is incomplete. AUROC only checks the order of scores. A user sees the decision at 0.5, and there strong blur flips 63% of real videos to FAKE while AUROC still reads 0.905."],
  ["What is the difference between spatial and temporal?",
    "Spatial looks at one frame at a time (texture, edges, eyes, lips). Temporal looks at how the face changes across 10 frames (flicker, unnatural motion). DFD-FCG has a head for each plus a fused head; GenD is spatial only. On FaceShifter, temporal alone catches 80%, spatial alone 40%."],
  ["Did you use heatmaps?",
    "Yes. Patch contributions plus a constant reconstruct DFD-FCG's raw spatial logit. The overlay averages those contributions over clips. It does not reconstruct the final ensemble probability or prove the location of a manipulation."],
  ["Why not simply move the threshold?",
    "The right line depends on how damaged each video is, which you don't know. One global re-calibration still gives 12.2% false alarms; my per-video read-out gives 8.7%."],
  ["Did you tune anything on the test set?",
    <>Yes, the original ridge strength and strict-mode weight followed a pilot on part of the test data. The read-out itself fits 49 train-split videos;
      the live combiner fits {EN.n_val} validation videos. Later follow-ups reuse known benchmarks, and an earlier synth validation split has a documented qualification. We report those limits.</>],
  ["Your fix catches fewer fakes. Isn't that worse?",
    "At the 0.5 line, yes (87.9% → 78.1%). At the same false-alarm rate it catches more (71.0% vs 67.2% at 5%), and balanced accuracy and mixed AUROC go up. The released model catches many fakes partly because it calls almost every damaged video FAKE."],
  ["Why three detectors?",
    "There are three network instances from two detector families: DFD-FCG, released GenD-PE and fine-tuned GenD-PE. DFD supplies one corrected score; each GenD supplies raw and corrected scores. The five scores are combined using learned weights, not a vote."],
  ["Does “without retraining” mean nothing was learned?",
    "No. The self-calibration read-out is fitted using training data while the detector stays frozen. Separately, the final checker includes a GenD model whose LayerNorm parameters and classifier head were fine-tuned, plus a fitted logistic combiner."],
  ["What extra work did you complete after the ensemble?",
    "We deployed reliability fixes for invalid scores, tracking and shared-model sampling. We also completed reweighting, CLIP comparisons and paired PE training. None met the replacement rule, so the live model and published accuracy stayed unchanged. The detailed records and a paper draft are linked above."],
  ["Why is the final system worse on FF++?",
    <>The measured issue is more false alarms on clean FF++ reals. A domain/style mismatch is a plausible explanation, not a demonstrated cause. The live system gives
      {" "}{p0(FIN["ffpp:clean"]?.false_alarm)} false alarms ({p0(BASE["ffpp:clean"]?.false_alarm)} before). A measured trade-off: {pts(EN.decision.gain_new_sets)} points on newer fakes,
      {" "}{pts(EN.decision.change_ffpp)} on FF++.</>],
  ["Why does it never say “not sure”?",
    <>Users need a decision. The confidence number carries the doubt: answers above 80% confidence were right {pc(CK.bands[2]?.acc)} of the time, under 65% only {pc(CK.bands[0]?.acc)}.</>],
  ["What did you build yourself?",
    "The test pipeline around the authors' code, the exact attribution and heatmap, the self-calibration, the GenD fine-tune and transfer, the combiner, the live checker (backend, website) and this guide."],
  ["What would you do next?",
    "Add more real videos in the 2019 news style to cut the FF++ false alarms (a short paired-training run did not help), use several seeds, add an audio-visual detector for lip-sync fakes, and fit the self-calibration on phone re-encoding and screen recordings."],
];

function Questions() {
  return (
    <Section id="questions" n="Professor questions" title="Likely questions, short answers">
      <div className="grid gap-3 md:grid-cols-2">
        {QA.map(([q, a]) => (
          <details key={q} className="group rounded-2xl border border-slate-200 bg-white px-5 py-4 shadow-sm open:shadow-md">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-4 font-semibold text-slate-900">
              {q}<span className="text-brand transition group-open:rotate-45">＋</span>
            </summary>
            <p className="mt-2 text-sm text-slate-600">{a}</p>
          </details>
        ))}
      </div>
    </Section>
  );
}

function Script() {
  const [hindi, setHindi] = useState(false);
  const lines = hindi ? [
    "Hamare project ka problem hai: blurry real video bhi detector ko fake lag sakta hai. Hamne is quality bias ko measure kiya aur internal features se correction fit kiya.",
    "DFD-FCG 10-frame clips leta hai, har frame 224 by 224 RGB. Spatial branch 1024 numbers aur temporal branch 256 numbers deti hai. Dono concatenate hoke 1280 numbers bante hain.",
    "Original classifier in features se three heads ke logits banata hai. Hamara fitted read-out raw features se three quality shifts predict karta hai, jinhe subtract karke ek corrected DFD score milta hai.",
    "Released aur fine-tuned GenD-PE har frame ko separately dekhte hain. Dono ek raw aur ek corrected score dete hain. DFD ka ek aur GenD ke chaar: total five scores.",
    "Learned combiner five score logits se final probability deta hai. Hamara contribution robustness study, correction, transfer, adaptation aur deployed checker hai. Gains ke saath lower fake recall, FF++ loss aur protocol limits bhi report kiye hain.",
  ] : [
    "Our problem is that a blurry real video can look fake to a detector. We measured this quality bias and fitted a correction from its internal features.",
    "DFD-FCG takes ten-frame clips at 224 by 224 RGB. Its spatial branch produces 1024 values and its temporal branch produces 256. Concatenating them gives 1280 features.",
    "The authors’ classifier produces three head scores. Our fitted read-out uses raw features to predict three quality shifts. We subtract them, apply sigmoid and average to obtain one corrected DFD score.",
    "Released and fine-tuned GenD-PE process frames individually. Each gives a raw and a corrected score. Together with DFD, that makes five scores from three networks. A learned combiner gives the final probability.",
    "Our contribution is the robustness study, feature-based correction, transfer evidence and evaluated checker. We report the gains together with the fake-recall cost, FF++ loss and evaluation limits.",
  ];
  return (
    <Section id="script" n="Practise this" title="Your one-minute architecture answer">
      <div className="mb-4 flex gap-2" role="group" aria-label="Script language"><button aria-pressed={!hindi} onClick={() => setHindi(false)} className={!hindi ? "btn-primary" : "btn-ghost"}>English</button><button aria-pressed={hindi} onClick={() => setHindi(true)} className={hindi ? "btn-primary" : "btn-ghost"}>Easy Hinglish</button></div>
      <Card className="p-6 md:p-8">
        <ol className="space-y-3" lang={hindi ? "hi-Latn" : "en"}>
          {lines.map((l, i) => (
            <li key={i} className="flex gap-4 text-slate-800 md:text-lg">
              <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-brand-soft text-sm font-bold text-brand">{i + 1}</span>
              <span>{l}</span>
            </li>
          ))}
        </ol>
      </Card>
      <div className="mt-4 flex flex-wrap gap-3">
        <a className="btn-ghost" href={LINKS.checker}>Live checker →</a>
        <a className="btn-ghost" href={LINKS.github}>Code and results →</a>
        <a className="btn-ghost" href={LINKS.paper}>DFD-FCG (authors) →</a>
      </div>
    </Section>
  );
}
