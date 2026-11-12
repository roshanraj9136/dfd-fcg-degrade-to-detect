import { Fragment, useEffect, useState } from "react";
import type { ReactNode } from "react";
import scores from "./scores.json";
import websiteCheck from "./website_check.json";
import { AttributionFig, Bars, BlurTestFig, FixFig, HeroFig, ModelArchFig, ScoreStrips, SystemFig } from "./figs";

const LINKS = {
  checker: "https://dfd-fcg-degrade-to-detect.vercel.app",
  space: "https://huggingface.co/spaces/roshan9136/dfd-fcg-checker",
  github: "https://github.com/roshanraj9136/dfd-fcg-degrade-to-detect",
  tables: "https://github.com/roshanraj9136/dfd-fcg-degrade-to-detect/blob/main/results/tables.md",
  paper: "https://github.com/aiiu-lab/DFD-FCG",
  earlier: "https://github.com/roshanraj9136/dfd-fcg-explainability",
  api: "https://dfd-fcg-degrade-to-detect-api.onrender.com",
};

const NAV: [string, string][] = [
  ["summary", "Summary"], ["architecture", "Architecture"], ["problem", "Problem"], ["why", "Why"],
  ["found", "How I found it"], ["fix", "My fix"], ["results", "Results"], ["website-test", "Website test"], ["novelty", "Novelty"],
  ["questions", "Questions"], ["script", "Script"],
];

type S = Record<string, { real: number[]; fake: number[]; real_fix: number[]; fake_fix: number[] }>;
const SC = scores as S;

export default function App() {
  return (
    <div className="min-h-screen">
      <Header />
      <main className="mx-auto max-w-6xl px-5">
        <Hero />
        <Summary />
        <Architecture />
        <Problem />
        <Why />
        <Found />
        <Fix />
        <Results />
        <Novelty />
        <Questions />
        <Script />
      </main>
      <Footer />
    </div>
  );
}

/* ------------------------------------------------------------------ layout helpers */
function Header() {
  const [open, setOpen] = useState(false);
  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-5 py-3">
        <a href="#top" className="flex shrink-0 items-center gap-2 whitespace-nowrap font-bold text-slate-900">
          <img src="/favicon.svg" alt="" className="h-7 w-7" /> Viva guide
        </a>
        <nav className="hidden gap-0.5 xl:flex">
          {NAV.map(([id, label]) => <a key={id} href={`#${id}`} className="whitespace-nowrap rounded-lg px-2 py-1.5 text-sm text-slate-600 hover:bg-slate-100 hover:text-slate-900">{label}</a>)}
        </nav>
        <button className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm xl:hidden" onClick={() => setOpen(!open)}>Sections</button>
      </div>
      {open && (
        <nav className="grid grid-cols-2 gap-1 border-t border-slate-200 px-5 py-3 xl:hidden">
          {NAV.map(([id, label]) => <a key={id} href={`#${id}`} onClick={() => setOpen(false)} className="rounded-lg px-2 py-1.5 text-sm text-slate-700 hover:bg-slate-100">{label}</a>)}
        </nav>
      )}
    </header>
  );
}

function Section({ id, n, title, lead, children, tone }: { id: string; n?: string; title: ReactNode; lead?: ReactNode; children: ReactNode; tone?: "red" | "green" | "brand" }) {
  const k = tone === "red" ? "text-fake" : tone === "green" ? "text-real" : "text-brand";
  return (
    <section id={id} className="scroll-mt-20 py-14">
      {n && <div className={`text-sm font-bold uppercase tracking-wider ${k}`}>{n}</div>}
      <h2 className="mt-1 text-3xl font-extrabold tracking-tight text-slate-900 md:text-4xl">{title}</h2>
      {lead && <p className="mt-3 max-w-3xl text-lg text-slate-600">{lead}</p>}
      <div className="mt-8">{children}</div>
    </section>
  );
}

function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-3xl border border-slate-200 bg-white p-6 shadow-sm ${className}`}>{children}</div>;
}

function Fig({ children, caption }: { children: ReactNode; caption?: ReactNode }) {
  return (
    <Card className="p-4 md:p-6">
      <div className="overflow-x-auto">{children}</div>
      {caption && <p className="mt-3 text-sm text-slate-500">{caption}</p>}
    </Card>
  );
}

function Say({ children }: { children: ReactNode }) {
  return (
    <div className="mt-6 rounded-2xl border-l-4 border-brand bg-brand-soft px-5 py-4 text-slate-800">
      <div className="text-xs font-bold uppercase tracking-wider text-brand">Say it like this</div>
      <p className="mt-1">{children}</p>
    </div>
  );
}

/* ------------------------------------------------------------------ sections */
function Hero() {
  return (
    <section id="top" className="grid items-center gap-10 py-14 md:grid-cols-[1.1fr_1fr]">
      <div>
        <div className="text-sm font-semibold text-brand">B.Tech project · Roshan Raj · viva guide</div>
        <h1 className="mt-3 text-4xl font-extrabold leading-tight tracking-tight text-slate-900 md:text-5xl">
          Fake, or just <span className="text-brand">low quality</span>?
        </h1>
        <p className="mt-5 text-lg text-slate-600">
          DFD-FCG (CVPR 2025) is one of the best video deepfake detectors. I found that it calls <b className="text-slate-800">real
          videos fake</b> when they are blurry or compressed, found <b className="text-slate-800">the reason inside the model</b>, and
          fixed it <b className="text-slate-800">without retraining</b>.
        </p>
        <div className="mt-7 flex flex-wrap gap-3">
          <a className="btn-primary" href={LINKS.checker}>Try the live checker</a>
          <a className="btn-ghost" href="#architecture">See the architecture</a>
          <a className="btn-ghost" href={LINKS.github}>Code</a>
        </div>
      </div>
      <Card className="p-4"><HeroFig /></Card>
    </section>
  );
}

function Summary() {
  const items = [
    { k: "The problem", tone: "fake", big: "63%", text: "of real test videos are called FAKE under strong blur (3% on clean video). The paper's robustness test (AUROC) does not show this." },
    { k: "The reason", tone: "ink", big: "+1.44 vs −0.17", text: "A 2-pixel blur adds a lot of fake evidence to real faces and almost none to fakes. The model treats fine natural detail as proof of “real”." },
    { k: "My fix", tone: "brand", big: "22.5% → 8.7%", text: "real videos called fake on low-quality video, after a small add-on that removes the quality effect. No retraining, works on unseen kinds of damage." },
  ];
  return (
    <Section id="summary" n="In one minute" title="Three things to remember">
      <div className="grid gap-5 md:grid-cols-3">
        {items.map((it) => (
          <Card key={it.k}>
            <div className="text-sm font-bold uppercase tracking-wider text-slate-500">{it.k}</div>
            <div className={`mt-2 text-3xl font-extrabold ${it.tone === "fake" ? "text-fake" : it.tone === "brand" ? "text-brand" : "text-slate-900"}`}>{it.big}</div>
            <p className="mt-2 text-slate-600">{it.text}</p>
          </Card>
        ))}
      </div>
      <Card className="mt-5">
        <div className="text-sm font-bold uppercase tracking-wider text-slate-500">Words I use</div>
        <dl className="mt-3 grid gap-x-8 gap-y-3 text-sm md:grid-cols-2">
          {[
            ["Fake score", "the model's probability that the video is fake (0 to 1)."],
            ["Decision line (threshold)", "0.5. Above it the model says FAKE. This is what a user sees."],
            ["False alarm", "a real video called FAKE."],
            ["Fakes caught", "the share of deepfakes called FAKE."],
            ["AUROC", "the chance that a random fake gets a higher score than a random real video. 1.0 = perfect order. It ignores the decision line."],
            ["Logit", "the raw score before it becomes a probability; positive means fake."],
          ].map(([a, b]) => (
            <div key={a}><dt className="font-semibold text-slate-900">{a}</dt><dd className="text-slate-600">{b}</dd></div>
          ))}
        </dl>
      </Card>
    </Section>
  );
}

function Architecture() {
  return (
    <Section id="architecture" n="Architecture" title="How DFD-FCG works, and where my add-on plugs in"
      lead={<>The grey parts are the authors' model, used exactly as released. The blue parts are mine. They sit after the
        model's last features and before its decision, so <b>nothing inside DFD-FCG is retrained</b>.</>}>
      <Fig caption="Scroll sideways on a phone. d_s, d_t, d_a are the three head scores (logits); g is the fake evidence predicted to come from low quality.">
        <ModelArchFig />
      </Fig>
      <div className="mt-6 grid gap-5 md:grid-cols-2 lg:grid-cols-4">
        {[
          ["1. CLIP ViT-L/14", "A large image model from OpenAI. It turns each face frame into 16×16 patches and processes them through 24 layers. The authors keep it frozen."],
          ["2. Side network", "The part the authors trained. The spatial module has 4 queries that each look at one face part (eyes, nose, lips, skin). The temporal module follows each patch over time."],
          ["3. Three heads", "Small linear layers turn the features y_s (1024 numbers) and y_t (256) into a spatial, a temporal and a fused score. Their probabilities are averaged; 0.5 or more means FAKE."],
          ["4. My add-on", "A linear read-out on y_s and y_t predicts how much fake evidence the video's quality added (g) and subtracts it from each head score. Strict mode also uses a blur test."],
        ].map(([t, d]) => (
          <Card key={t}><div className={`font-bold ${t.startsWith("4") ? "text-brand" : "text-slate-900"}`}>{t}</div><p className="mt-2 text-sm text-slate-600">{d}</p></Card>
        ))}
      </div>

      <h3 className="mt-14 text-2xl font-bold text-slate-900">How the live system is built</h3>
      <p className="mt-2 max-w-3xl text-slate-600">
        The checker website sends the video to a Hugging Face Space. The Space finds and aligns the face on its own CPU,
        then borrows a free GPU for about 2 seconds to run DFD-FCG and the blur test, and returns the answer in about
        15 to 25 seconds. A Neon Postgres database keeps the research results and an anonymous log of checks; a small
        API on Render serves the research numbers.
      </p>
      <div className="mt-6"><Fig caption="CPU / GPU tags show where each step runs. If a visitor's free GPU time is used up, the two GPU steps run on the CPU instead: same answer, a few seconds slower."><SystemFig /></Fig></div>
      <LatestUpdate />
      <LiveNumbers />
    </Section>
  );
}

function Problem() {
  const rows = [
    { label: "Clean video", sub: "as recorded (FF++ c23)", real: SC.clean.real, fake: SC.clean.fake },
    { label: "Strong blur", sub: "authors' protocol, GB level 5", real: SC.GB_5.real, fake: SC.GB_5.fake },
    { label: "With my fix", sub: "same blurred videos", real: SC.GB_5.real_fix, fake: SC.GB_5.fake_fix, mine: true },
  ];
  return (
    <Section id="problem" n="What the paper missed" tone="red" title={<>A good ranking, but the <span className="text-fake">decision line breaks</span></>}
      lead={<>The paper reports robustness to blur, noise and compression as AUROC under each perturbation, and it stays high. I
        re-ran the authors' own perturbation code and looked at the actual decisions instead.</>}>
      <div className="grid gap-5 md:grid-cols-2">
        <Card>
          <div className="text-sm font-bold uppercase tracking-wider text-slate-500">What the paper looks at</div>
          <div className="mt-2 text-3xl font-extrabold text-slate-900">AUROC 0.999 → 0.946</div>
          <p className="mt-2 text-slate-600">clean video → average over 13 damaged versions. It looks robust. But AUROC only checks the <b>order</b> of the scores, never the 0.5 line.</p>
        </Card>
        <Card className="border-red-200 bg-fake-soft">
          <div className="text-sm font-bold uppercase tracking-wider text-fake">What a user gets</div>
          <div className="mt-2 text-3xl font-extrabold text-fake">22.5% false alarms</div>
          <p className="mt-2 text-slate-700">on average over the same 13 damaged versions (2.9% on clean video), and <b>63%</b> under strong blur. Blur, resampling and H.264 push the scores towards FAKE.</p>
        </Card>
      </div>
      <div className="mt-6">
        <Fig caption={<>Each dot is one test video (70 real, 60 fake), measured. Under strong blur all scores slide right: the order is still decent
          (AUROC 0.905) but most real videos cross the line. After my fix the real videos move back.</>}>
          <ScoreStrips rows={rows} />
        </Fig>
      </div>
      <div className="mt-6 grid gap-5">
        <Fig caption="Share of real test videos called FAKE by DFD-FCG as released.">
          <Bars aColor="#dc2626" aLabel="real videos called FAKE (released model)" bars={[
            { label: "Clean", a: 3 }, { label: "Blur, medium", a: 40, note: "GB 3" }, { label: "Blur, strong", a: 63, note: "GB 5" },
            { label: "Resampling, strong", a: 59, note: "“JPEG” 5" }, { label: "H.264, strong", a: 33, note: "VC 5" },
          ]} />
        </Fig>
        <Card className="border-amber-200 bg-amber-50">
          <div className="text-sm font-bold uppercase tracking-wider text-amber-700">The opposite failure</div>
          <div className="mt-2 text-3xl font-extrabold text-amber-800">98% → 17%</div>
          <p className="mt-2 text-slate-700">fakes caught, clean vs. Gaussian noise. Noise pushes scores towards REAL and hides deepfakes.
            My fix does not solve this one (13%); I report it as a limitation.</p>
        </Card>
      </div>
      <Say>The paper's numbers are correct, but its robustness test cannot see the problem. AUROC stays high, while at the
        model's own threshold blur and compression turn up to 63% of real videos into “fakes”.</Say>
    </Section>
  );
}

function Why() {
  return (
    <Section id="why" n="Why it happens" title="The model uses fine detail as proof of “real”"
      lead={<>I blurred the aligned face by just 2 pixels and ran the model again. Real faces suddenly look much more fake.
        Deepfakes hardly change.</>}>
      <Fig><BlurTestFig /></Fig>
      <div className="mt-6 grid gap-5 md:grid-cols-3">
        <Card><div className="font-bold text-slate-900">Real faces have natural detail</div><p className="mt-2 text-sm text-slate-600">Skin texture, pores and sharp edges come from a real camera. The blur test shows that DFD-FCG reads them as evidence of a real video.</p></Card>
        <Card><div className="font-bold text-slate-900">Damage removes that detail</div><p className="mt-2 text-sm text-slate-600">Blur, resampling and compression wipe it out. A real video loses its “proof” and drifts towards FAKE. Fakes have less natural detail to lose.</p></Card>
        <Card><div className="font-bold text-slate-900">It is a strong signal</div><p className="mt-2 text-sm text-slate-600">On clean videos the blur reaction alone separates real from fake with AUROC 0.992. It is weaker on damaged video and on DFDC (0.74), and noise reverses it.</p></Card>
      </div>
      <Say>A real face reacts to a tiny blur, a fake one doesn't. So the model's “real” evidence is fine detail, and that is
        exactly what low quality destroys.</Say>
    </Section>
  );
}

function Found() {
  return (
    <Section id="found" n="How I found it" title="An exact split of the model's score"
      lead={<>In my earlier project I showed that DFD-FCG's score can be split exactly into pieces: one per layer and face
        part. Here I compared the pieces of the same video before and after damage, on 5,976 clips.</>}>
      <Fig><AttributionFig /></Fig>
      <div className="mt-6 grid gap-5 md:grid-cols-3">
        <Card><div className="font-bold text-slate-900">Not a heatmap</div><p className="mt-2 text-sm text-slate-600">Attention maps only suggest where the model looked. These pieces add up to the actual score, so nothing is guessed.</p></Card>
        <Card><div className="font-bold text-slate-900">It showed where the shift enters</div><p className="mt-2 text-sm text-slate-600">The extra fake evidence from damage runs through the same head features (y_s, y_t) that also show how damaged the face is. That made it predictable, which is the idea behind the fix.</p></Card>
        <Card><div className="font-bold text-slate-900">It also re-tested my earlier claims</div><p className="mt-2 text-sm text-slate-600">Some hold at scale (eyes are the top query on fakes, 35%). One does not: real verdicts are not “70.9% temporal”, only 45%.</p></Card>
      </div>
    </Section>
  );
}

function Fix() {
  return (
    <Section id="fix" n="My change" tone="brand" title="Self-calibration: let the model correct its own quality bias"
      lead={<>The model's own features already know how damaged a face is. A small read-out turns that into an estimate of the
        fake evidence the damage added, and removes it.</>}>
      <Fig><FixFig /></Fig>
      <div className="mt-6 grid gap-5 md:grid-cols-2">
        <Card>
          <div className="font-bold text-slate-900">How the read-out was fitted</div>
          <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm text-slate-600">
            <li>Training-split videos only: 30 real and 19 fake, each clean and at 7 kinds of damage (levels 2 and 4).</li>
            <li>Target: for each kind of damage, how much the scores of <b>real</b> videos moved (same video, clean vs damaged). Clean = 0.</li>
            <li>The same target is used for real and fake clips, so the read-out cannot simply learn “push fakes down”.</li>
            <li>Ridge regression (α = 1000) from the 1280 feature numbers to the 3 head shifts.</li>
          </ol>
        </Card>
        <Card>
          <div className="font-bold text-slate-900">Why this is a good design</div>
          <ul className="mt-3 space-y-2 text-sm text-slate-600">
            <li>• <b>No retraining</b> of DFD-FCG, and the same 0.5 line.</li>
            <li>• <b>No quality label needed</b> for a new video: it is estimated from the features.</li>
            <li>• <b>Tested on damage types it never saw</b> (each type left out of fitting in turn).</li>
            <li>• <b>Tiny and fast</b>: 3 × 1280 weights, under a millisecond.</li>
            <li>• <b>Strict mode</b> (optional): also subtracts half of the blur-test reaction, for when a false accusation is costly.</li>
          </ul>
        </Card>
      </div>
      <Say>I don't change the detector. I add one linear layer that reads the detector's own features, estimates how much fake
        evidence came only from poor quality, and subtracts it.</Say>
    </Section>
  );
}

function Results() {
  return (
    <Section id="results" n="Results" tone="green" title="Fewer false alarms, better ranking across qualities"
      lead={<>Test split: 70 real and 60 fake videos (Deepfakes, Face2Face, FaceSwap, NeuralTextures), 13 damaged versions each.
        Numbers are averages, with the tested type of damage hidden from fitting.</>}>
      <div className="grid gap-5">
        <Fig caption="Lower is better.">
          <Bars bars={[
            { label: "Real called FAKE", note: "average, 13 damaged versions", a: 22.5, b: 8.7 },
            { label: "Strong blur", note: "GB 5", a: 63, b: 11 },
            { label: "Medium blur", note: "GB 3", a: 40, b: 4 },
            { label: "Strong H.264", note: "VC 5", a: 33, b: 13 },
          ]} />
        </Fig>
        <Fig caption="Higher is better. “At 5% false alarms”: fakes caught when each rule may wrongly flag at most 5% of real videos.">
          <Bars lowerIsBetter={false} bars={[
            { label: "Balanced accuracy", a: 82.7, b: 84.7 },
            { label: "Fakes caught", note: "at 5% false alarms", a: 67.2, b: 71.0 },
            { label: "Fakes caught", note: "at 10% false alarms", a: 77.6, b: 80.5 },
            { label: "AUROC", note: "all qualities mixed", a: 92.0, b: 95.1, fmt: (v) => (v / 100).toFixed(3) },
          ]} />
        </Fig>
      </div>
      <div className="mt-6 grid gap-5 md:grid-cols-3">
        <Card>
          <div className="text-sm font-bold uppercase tracking-wider text-slate-500">Clean video stays perfect</div>
          <div className="mt-2 text-2xl font-extrabold text-slate-900">AUROC 0.999 → 0.999</div>
          <p className="mt-2 text-sm text-slate-600">False alarms 2.9% → 1.4%, fakes caught 98.3% → 98.3%.</p>
        </Card>
        <Card>
          <div className="text-sm font-bold uppercase tracking-wider text-slate-500">Beats the simple fix</div>
          <div className="mt-2 text-2xl font-extrabold text-slate-900">12.2% vs 8.7%</div>
          <p className="mt-2 text-sm text-slate-600">One global re-calibration of the score (Platt scaling) leaves 12.2% false alarms and lowers mixed AUROC to 0.917.</p>
        </Card>
        <Card>
          <div className="text-sm font-bold uppercase tracking-wider text-slate-500">Strict mode</div>
          <div className="mt-2 text-2xl font-extrabold text-slate-900">2.3% false alarms</div>
          <p className="mt-2 text-sm text-slate-600">With the blur test as well; 61.4% of fakes caught. For cases where wrongly accusing someone is costly.</p>
        </Card>
      </div>
      <Card className="mt-6 border-amber-200 bg-amber-50">
        <div className="font-bold text-amber-800">The honest cost</div>
        <ul className="mt-2 grid gap-2 text-sm text-slate-700 md:grid-cols-2">
          <li>• At the 0.5 line it catches fewer fakes on damaged video: <b>87.9% → 78.1%</b>.</li>
          <li>• FaceShifter (a fake type the model never saw): AUROC unchanged at 0.925, but caught at 0.5 drops <b>73% → 53%</b>.</li>
          <li>• Gaussian noise still hides fakes (17% → 13% caught). Strong resampling still gives 44% false alarms.</li>
          <li>• On DFDC (another dataset) AUROC barely moves: 0.786 → 0.789.</li>
        </ul>
      </Card>
      <p className="mt-4 text-sm text-slate-500">Every number: <a className="text-brand hover:underline" href={LINKS.tables}>results/tables.md</a> in the repository.</p>
      <WebsiteCheck />
    </Section>
  );
}

/* ------------------------------------------------------------------ the live website, tested on the paper's videos */
type M = { acc: number; false_alarms?: number; caught?: number; balanced?: number; auroc?: number };
type B = { n: number; n_fake: number; site_original: M; site_fix: M; site_strict: M; research_original: M; research_fix: M; same_verdict: number; mean_abs_diff: number };
const WC = websiteCheck as unknown as {
  videos: number; failed: number; ffpp: Record<"clean" | "GB_5" | "VC_5", B>; faceshifter: B; dfdc: B;
  per_type: Record<string, { original: number; fix: number; blur_fix: number; n: number }>; two_face_mismatch: string[];
};
const pc = (v?: number) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);
const OPTIONS: ["clean" | "GB_5" | "VC_5", string][] = [["clean", "Video as it is"], ["GB_5", "Strong blur added"], ["VC_5", "Heavy compression added"]];

function WebsiteCheck() {
  const c = WC.ffpp.clean, b = WC.ffpp.GB_5, v = WC.ffpp.VC_5, fs = WC.faceshifter, dd = WC.dfdc;
  const rows: [string, keyof B, string][] = [
    ["Original DFD-FCG", "site_original", "text-slate-700"],
    ["Website answer (with my fix)", "site_fix", "text-brand font-semibold"],
    ["Strict mode (fix + blur test)", "site_strict", "text-slate-700"],
    ["Research pipeline, with my fix", "research_fix", "text-slate-500"],
  ];
  return (
    <div id="website-test" className="scroll-mt-20">
      <h3 className="mt-14 text-2xl font-bold text-slate-900">I tested the live website on the paper's videos</h3>
      <p className="mt-2 max-w-3xl text-slate-600">
        I ran the exact code behind the website, with the same settings as the live site, on all {WC.videos} test
        videos: the FF++ test split ({c.n - c.n_fake} real, {c.n_fake} fake), {fs.n} FaceShifter fakes the model never
        saw, and {dd.n} DFDC videos from another dataset. Each FF++ video was checked with all three website options.
        {WC.failed === 0 ? " None of the checks failed." : ` ${WC.failed} checks failed.`}
      </p>

      <div className="mt-6 grid gap-5 md:grid-cols-3">
        <Card>
          <div className="text-sm font-bold uppercase tracking-wider text-slate-500">Normal videos</div>
          <div className="mt-2 text-3xl font-extrabold text-real">{pc(c.site_fix.acc)}</div>
          <p className="mt-2 text-sm text-slate-600">accuracy on FF++. {pc(c.site_fix.caught)} of fakes caught, {pc(c.site_fix.false_alarms)} of real videos wrongly called fake. AUROC {c.site_fix.auroc!.toFixed(3)}.</p>
        </Card>
        <Card>
          <div className="text-sm font-bold uppercase tracking-wider text-slate-500">Blurry videos</div>
          <div className="mt-2 text-3xl font-extrabold text-brand">{pc(b.site_original.false_alarms)} → {pc(b.site_fix.false_alarms)}</div>
          <p className="mt-2 text-sm text-slate-600">real videos wrongly called fake, original detector → website. Accuracy {pc(b.site_original.acc)} → {pc(b.site_fix.acc)}.</p>
        </Card>
        <Card>
          <div className="text-sm font-bold uppercase tracking-wider text-slate-500">Same as my research</div>
          <div className="mt-2 text-3xl font-extrabold text-slate-900">{pc(c.same_verdict)}</div>
          <p className="mt-2 text-sm text-slate-600">same verdict as the research pipeline on normal videos ({pc(b.same_verdict)} blurred, {pc(v.same_verdict)} compressed). Scores differ by {c.mean_abs_diff.toFixed(3)} on average.</p>
        </Card>
      </div>

      <Card className="mt-6 overflow-x-auto p-0">
        <table className="w-full min-w-[760px] text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs uppercase tracking-wider text-slate-500">
              <th className="px-5 py-3">FF++ test, {c.n} videos</th>
              <th className="px-3 py-3 text-right">Accuracy</th>
              <th className="px-3 py-3 text-right">Real called fake</th>
              <th className="px-3 py-3 text-right">Fakes caught</th>
              <th className="px-3 py-3 text-right">Balanced acc.</th>
              <th className="px-5 py-3 text-right">AUROC</th>
            </tr>
          </thead>
          <tbody>
            {OPTIONS.map(([k, label]) => (
              <Fragment key={k}>
                <tr className="border-t border-slate-200 bg-white"><td colSpan={6} className="px-5 pb-1 pt-4 font-bold text-slate-900">{label}</td></tr>
                {rows.map(([name, key, cls]) => {
                  const x = WC.ffpp[k][key] as M;
                  return (
                    <tr key={name} className={key === "site_fix" ? "bg-brand-soft/60" : ""}>
                      <td className={`px-5 py-1.5 ${cls}`}>{name}</td>
                      <td className={`px-3 py-1.5 text-right tabular-nums ${cls}`}>{pc(x.acc)}</td>
                      <td className={`px-3 py-1.5 text-right tabular-nums ${cls}`}>{pc(x.false_alarms)}</td>
                      <td className={`px-3 py-1.5 text-right tabular-nums ${cls}`}>{pc(x.caught)}</td>
                      <td className={`px-3 py-1.5 text-right tabular-nums ${cls}`}>{pc(x.balanced)}</td>
                      <td className={`px-5 py-1.5 text-right tabular-nums ${cls}`}>{x.auroc?.toFixed(3) ?? "–"}</td>
                    </tr>
                  );
                })}
              </Fragment>
            ))}
          </tbody>
        </table>
        <p className="px-5 pb-4 pt-3 text-xs text-slate-500">The website shows the blue row. Strict mode is printed under every website result. Grey: the research pipeline on the same videos, for comparison.</p>
      </Card>

      <Card className="mt-6 border-amber-200 bg-amber-50">
        <div className="font-bold text-amber-800">The trade-offs, honestly</div>
        <ul className="mt-3 grid gap-3 text-sm text-slate-700 md:grid-cols-2">
          <li>• <b>Fewer false alarms costs some caught fakes.</b> On strongly blurred video the website wrongly flags {pc(b.site_fix.false_alarms)} of real
            videos instead of {pc(b.site_original.false_alarms)}, but catches {pc(b.site_fix.caught)} of fakes instead of {pc(b.site_original.caught)}.
            The original only catches so many because it calls almost every blurry video fake.</li>
          <li>• <b>Heavy compression is the weakest case.</b> False alarms drop only from {pc(v.site_original.false_alarms)} to {pc(v.site_fix.false_alarms)},
            and fakes caught drop from {pc(v.site_original.caught)} to {pc(v.site_fix.caught)}.</li>
          <li>• <b>Strict mode</b> gives the fewest false alarms ({pc(c.site_strict.false_alarms)} normal, {pc(b.site_strict.false_alarms)} blurred,
            {" "}{pc(v.site_strict.false_alarms)} compressed) but catches fewer fakes ({pc(c.site_strict.caught)}, {pc(b.site_strict.caught)}, {pc(v.site_strict.caught)}).</li>
          <li>• <b>A fake type it never saw</b> (FaceShifter, {fs.n} videos): caught {pc(fs.site_original.caught)} by the original and {pc(fs.site_fix.caught)} with
            my fix (research pipeline: {pc(fs.research_original.caught)} → {pc(fs.research_fix.caught)}).</li>
          <li>• <b>Another dataset</b> (DFDC, {dd.n} videos): accuracy {pc(dd.site_fix.acc)}, {pc(dd.site_fix.false_alarms)} false alarms, {pc(dd.site_fix.caught)} of
            fakes caught, AUROC {dd.site_fix.auroc!.toFixed(3)}. DFD-FCG itself is weak here (research: AUROC {dd.research_fix.auroc!.toFixed(3)}).</li>
          <li>• <b>Two people in the video.</b> In {WC.two_face_mismatch.length} of the {dd.n} DFDC videos two people sit side by side, and the website followed a
            different person than the research pipeline. In 2 of those fakes the website said REAL where the research pipeline said FAKE, most likely
            because it checked the person who was not swapped. This only affects videos with more than one face; the website asks for one clear face.</li>
          <li className="md:col-span-2">• <b>By fake type</b> (fakes caught by the website, normal → strongly blurred):{" "}
            {Object.entries(WC.per_type).map(([k, t], i) => (
              <span key={k}>{i > 0 && " · "}{({ DF: "Deepfakes", F2F: "Face2Face", FS: "FaceSwap", NT: "NeuralTextures" } as Record<string, string>)[k]} {pc(t.fix)} → {pc(t.blur_fix)}</span>
            ))}.</li>
        </ul>
      </Card>
      <Say>I ran the live website's own code on all {WC.videos} test videos. On normal FF++ videos it is {pc(c.site_fix.acc)} accurate, and on blurry
        videos it cuts wrongly accused real videos from {pc(b.site_original.false_alarms)} to {pc(b.site_fix.false_alarms)}. It agrees with my research
        pipeline on {pc(c.same_verdict)} of videos. The price is that it catches fewer fakes on damaged video, and it is weaker on other datasets, as
        DFD-FCG itself is.</Say>
    </div>
  );
}

function Novelty() {
  return (
    <Section id="novelty" n="Novelty" tone="brand" title="What is new, and what is not"
      lead="Be precise about this. It is the part a professor tests hardest.">
      <div className="grid gap-5 md:grid-cols-2">
        {[
          ["1", "A failure the paper does not report", "DFD-FCG's decisions collapse under blur and compression while AUROC stays high. I measured it at the model's own threshold, per condition and with all qualities mixed, using the authors' own perturbation code."],
          ["2", "The reason, measured inside the model", "Real faces react to a 2-pixel blur (+1.44), fakes barely (−0.17). Found with an exact split of the score, not with heatmaps."],
          ["3", "A fix without retraining", "Self-calibration from the model's own features: false alarms 22.5% → 8.7% on damage types it never saw, mixed-quality AUROC 0.920 → 0.951."],
          ["4", "A working public tool", "Anyone can upload a video and get REAL / FAKE with the evidence map, the quality correction and the blur test."],
        ].map(([n, t, d]) => (
          <Card key={n} className="flex gap-4">
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-brand text-lg font-extrabold text-white">{n}</span>
            <div><div className="font-bold text-slate-900">{t}</div><p className="mt-1 text-sm text-slate-600">{d}</p></div>
          </Card>
        ))}
      </div>
      <div className="mt-6 grid gap-5 md:grid-cols-2">
        <Card>
          <div className="font-bold text-slate-900">Not mine (say it before they ask)</div>
          <ul className="mt-2 space-y-1.5 text-sm text-slate-600">
            <li>• The DFD-FCG model, its training and its weights (Han et al., CVPR 2025).</li>
            <li>• The perturbation protocol and code (from the authors' repository).</li>
            <li>• Ridge regression and Platt scaling are standard tools.</li>
            <li>• That compression hurts deepfake detectors is known in general. New here is the measurement for this model at its threshold, the mechanism and the fix.</li>
          </ul>
        </Card>
        <Card>
          <div className="font-bold text-slate-900">Ideas that failed (also reported)</div>
          <ul className="mt-2 space-y-1.5 text-sm text-slate-600">
            <li>• Estimating quality from the background: did not transfer to new damage types.</li>
            <li>• Using the blur test alone: no gain at the same false-alarm rate.</li>
            <li>• Removing a “quality direction” from the explanation pieces.</li>
            <li>• A quality score from the face crop: 15.1% false alarms on unseen types, worst 69%.</li>
          </ul>
        </Card>
      </div>
      <Card className="mt-6 overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-slate-500"><th className="py-2 pr-4"></th><th className="py-2 pr-4">DFD-FCG paper</th><th className="py-2 pr-4">my earlier project</th><th className="py-2 text-brand">this project</th></tr>
          </thead>
          <tbody className="text-slate-700">
            {[
              ["Robustness check", "AUROC per perturbation", "–", "AUROC and the actual decisions at 0.5, per condition and mixed"],
              ["Videos", "large public benchmarks", "1 real + 1 fake", "275 videos: FF++ test split, FaceShifter, DFDC; 15 quality conditions"],
              ["Explanation", "attention maps", "exact pieces of one decision", "exact pieces of how a decision changes under damage"],
              ["Decision rule", "fixed", "–", "self-calibration, plus an optional strict mode"],
              ["Usable tool", "code", "–", "public upload checker"],
            ].map((r) => (
              <tr key={r[0]} className="border-b border-slate-100 last:border-0">
                <td className="py-2.5 pr-4 font-semibold text-slate-900">{r[0]}</td><td className="py-2.5 pr-4">{r[1]}</td><td className="py-2.5 pr-4">{r[2]}</td><td className="py-2.5 font-medium text-slate-900">{r[3]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </Section>
  );
}

const QA: [string, ReactNode][] = [
  ["So what exactly is wrong in the paper?",
    <>Its numbers are not wrong; its robustness test is incomplete. AUROC only checks whether fakes are ranked above real
      videos within one quality level. A user only sees the decision at 0.5, and there blur and compression flip many real
      videos to FAKE: 63% under strong blur, while AUROC still reads 0.905.</>],
  ["Why not simply move the threshold?",
    <>The right threshold depends on how damaged each video is, which you don't know in advance. One global re-calibration
      (Platt scaling) still gives 12.2% false alarms and lowers the mixed-quality AUROC to 0.917. My read-out adapts per video:
      8.7% and 0.951.</>],
  ["Did you tune anything on the test set?",
    <>No. The read-out is fitted on training-split videos (30 real, 19 fake) at damage levels 2 and 4. Testing uses the test
      split (70 real, 60 fake) at levels 1, 3 and 5. In the main protocol the tested type of damage is removed from fitting
      completely. A 5-fold cross-validation over test identities gives the same picture (7.8% false alarms).</>],
  ["Your method catches fewer fakes. Isn't that worse?",
    <>At the 0.5 line, yes: 87.9% → 78.1%. But when both rules are allowed the same false-alarm rate, mine catches more fakes
      (71.0% vs 67.2% at 5%), balanced accuracy goes up (82.7% → 84.7%) and the mixed-quality AUROC goes up. The released
      model catches many fakes partly because it calls almost everything damaged FAKE.</>],
  ["How do you know how damaged a new video is?",
    <>I don't need to. The read-out estimates the effect from the model's own features. It was tested on kinds of damage it
      never saw during fitting.</>],
  ["Why should I trust your explanation?",
    <>Because it is exact, not approximate. The 240 pieces add up to the model's actual score within 0.0000003 on all 5,976
      clips. A heatmap cannot be checked like that.</>],
  ["What does not work?",
    <>Gaussian noise still hides fakes (17% caught, 13% after the fix). Strong resampling still has 44% false alarms. On the
      unseen FaceShifter fakes fewer are caught at 0.5 (73% → 53%). On DFDC the gain is small (AUROC 0.786 → 0.789).</>],
  ["Is the dataset big enough?",
    <>275 videos and 5,976 clips: the FF++ test split (70 real, 60 fake), 15 FaceShifter fakes, 80 DFDC videos, 49 training
      videos for fitting and 1 demo clip. Enough to show the effects clearly (63% → 11% is not noise), but the exact
      percentages have wide error bars. I say so.</>],
  ["What did you build yourself?",
    <>The test pipeline around the authors' code (download, alignment, 15 quality conditions), the exact attribution engine,
      the evaluation protocols, the self-calibration and strict mode, the checker backend on Hugging Face (face finding on
      the CPU, the detector on a free GPU with a CPU fallback), both websites, the API and the database.</>],
  ["What if many people use the live checker, or the free GPU runs out?",
    <>Hugging Face gives each visitor a few free GPU minutes a day. I split the work so only the detector uses the GPU, about
      2 seconds per video; finding the face runs on the CPU. If no GPU time is left, the detector runs on the CPU instead
      and gives the same answer a few seconds later. Checks are handled one at a time, and the website shows how many
      videos are ahead.</>],
  ["Does the live checker give the same answer as your research pipeline?",
    <>It uses the same model, weights, alignment and add-on. To be fast on a CPU it looks for the face on every 12th frame
      and computes landmarks on every 3rd frame (plus every frame the model reads), then smooths them the same way. I ran
      the website's code on all 225 test videos: it gives the same verdict as the research pipeline on 99.2% of normal
      FF++ videos (96.2% blurred, 95.4% compressed), and the scores differ by about 0.01 on average.</>],
  ["How accurate is the website?",
    <>On the FF++ test videos: 97.7% on normal video, 80.8% with strong blur (the original detector: 66.2%) and 77.7% with
      heavy compression. On blurry video it wrongly accuses 7.1% of real videos instead of 61.4%, but catches 66.7% of
      fakes instead of 98.3%. On DFDC, another dataset, it is 67.5%, because DFD-FCG itself is weak there.</>],
  ["How is this different from your earlier project?",
    <>The earlier project explained 2 videos. This one tests 275 videos under 15 conditions, finds a failure and its cause,
      fixes it, and re-tests the earlier claims (some hold, the “70.9% temporal” one does not).</>],
];

function Questions() {
  return (
    <Section id="questions" n="Professor questions" title="Likely questions, short answers">
      <div className="space-y-3">
        {QA.map(([q, a]) => (
          <details key={q} className="group rounded-2xl border border-slate-200 bg-white px-5 py-4 shadow-sm open:shadow-md">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-4 font-semibold text-slate-900">
              {q}
              <span className="text-brand transition group-open:rotate-45">＋</span>
            </summary>
            <p className="mt-3 text-slate-600">{a}</p>
          </details>
        ))}
      </div>
    </Section>
  );
}

function Script() {
  const lines = [
    "DFD-FCG is a CVPR 2025 video deepfake detector built on CLIP. The paper says it is robust to blur and compression, because its AUROC stays high.",
    "I tested its real decisions, at the 0.5 line, on 275 videos and 15 quality conditions using the authors' own code. Under strong blur it calls 63% of real videos fake; under noise it misses most fakes.",
    "With an exact split of the score into 240 pieces I found the reason: the model uses fine natural detail as proof of “real”. A 2-pixel blur adds +1.44 fake evidence to real faces and almost nothing to fakes.",
    "So I added one small linear read-out. From the model's own features it estimates how much fake evidence came only from poor quality, and subtracts it. No retraining.",
    "False alarms on damaged video drop from 22.5% to 8.7%, also for damage types it never saw, and the mixed-quality AUROC rises from 0.920 to 0.951. The cost is that fewer fakes are caught at the default line, which I report.",
    "Finally, I deployed it as a public website where anyone can upload a video and see the answer and the reason.",
  ];
  return (
    <Section id="script" n="Practise this" title="The one-minute script">
      <Card className="p-7 md:p-9">
        <ol className="space-y-4">
          {lines.map((l, i) => (
            <li key={i} className="flex gap-4 text-lg text-slate-800">
              <span className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-brand-soft text-sm font-bold text-brand">{i + 1}</span>
              <span>{l}</span>
            </li>
          ))}
        </ol>
      </Card>
      <div className="mt-6 grid gap-4 md:grid-cols-4">
        {[["Live checker", LINKS.checker], ["Checker on Hugging Face", LINKS.space], ["Code and results", LINKS.github], ["DFD-FCG (authors)", LINKS.paper]].map(([t, u]) => (
          <a key={t} href={u} className="rounded-2xl border border-slate-200 bg-white px-5 py-4 font-semibold text-slate-900 shadow-sm transition hover:border-brand hover:text-brand">{t} →</a>
        ))}
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------ latest update: making the live checker reliable */
function LatestUpdate() {
  const rows: [string, string, string][] = [
    ["Problem I hit", "Hugging Face gives every visitor only a few free GPU minutes per day (2 minutes without an account). The old checker ran everything on the GPU and booked about a minute of GPU time per video, so a visitor could do only a few checks a day before getting a “quota exceeded” error.",
      "fake"],
    ["What I changed", "Finding the face (the slow part) now runs on the server's CPU, which has no daily limit. The GPU is used only for the detector itself, about 2 seconds per video instead of about 25. Each check now books 23 seconds of GPU instead of 53 to 68.",
      "brand"],
    ["Safety net", "If the free GPU time is still used up, or no GPU is free, the detector runs on the CPU instead. Same answer (tested: 93% fake on the sample deepfake either way), only a few seconds slower. The website then says “Using the server's CPU” instead of failing.",
      "real"],
    ["Honest progress bar", "The website now shows the real step the server is on (finding the face, getting a GPU or using the CPU, checking, blur test) instead of a timer guess.",
      "ink"],
  ];
  return (
    <Card className="mt-6">
      <div className="flex flex-wrap items-center gap-3">
        <span className="rounded-full bg-brand-soft px-3 py-1 text-xs font-bold uppercase tracking-wider text-brand">Latest update</span>
        <div className="font-bold text-slate-900">The live checker no longer runs out of GPU</div>
      </div>
      <div className="mt-5 grid gap-4 md:grid-cols-2">
        {rows.map(([t, d, tone]) => (
          <div key={t} className="rounded-2xl bg-slate-50 p-4">
            <div className={`font-semibold ${tone === "fake" ? "text-fake" : tone === "brand" ? "text-brand" : tone === "real" ? "text-real" : "text-slate-900"}`}>{t}</div>
            <p className="mt-1 text-sm text-slate-600">{d}</p>
          </div>
        ))}
      </div>
      <div className="mt-5 grid gap-3 text-center sm:grid-cols-3">
        {[["~2 s", "GPU time per video (was ~25 s)"], ["15–25 s", "answer time on the website"], ["Same answer", "on the GPU or on the CPU backup"]].map(([b, s]) => (
          <div key={s} className="rounded-2xl border border-slate-200 px-3 py-3">
            <div className="text-2xl font-extrabold text-slate-900">{b}</div>
            <div className="text-xs text-slate-500">{s}</div>
          </div>
        ))}
      </div>
      <Say>The heavy face-finding runs on the free CPU and only the detector borrows a GPU for two seconds. If the free GPU
        time runs out, it falls back to the CPU and still gives the same answer, so the demo always works.</Say>
    </Card>
  );
}

/* ------------------------------------------------------------------ live numbers from Neon via Render */
type Row = Record<string, number | string>;

function LiveNumbers() {
  const [state, setState] = useState<"idle" | "loading" | "ok" | "error">("idle");
  const [rows, setRows] = useState<{ orig: Row; mine: Row; n: Record<string, number> } | null>(null);
  const [since, setSince] = useState(0);
  useEffect(() => {
    if (state !== "loading") return;
    const t = setInterval(() => setSince((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [state]);
  const load = async () => {
    setState("loading");
    setSince(0);
    try {
      const r = await fetch(`${LINKS.api}/api/summary`);
      const s = await r.json();
      const pick = (m: string, g: string) => (s.table as Row[]).find((x) => x.method === m && x.regime === g)!;
      setRows({ orig: pick("original", "all"), mine: pick("selfcal", "lopo"), n: s.n });
      setState("ok");
    } catch {
      setState("error");
    }
  };
  const pct = (v: number | string) => `${(Number(v) * 100).toFixed(1)}%`;
  return (
    <Card className="mt-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="font-bold text-slate-900">Check the numbers live</div>
          <p className="text-sm text-slate-600">Reads the stored results from Neon Postgres through the Render API, right now.</p>
        </div>
        <button className="btn-ghost" onClick={load} disabled={state === "loading"}>
          {state === "loading" ? `Loading… ${since}s${since > 8 ? " (the free server is waking up)" : ""}` : "Load from the database"}
        </button>
      </div>
      {state === "error" && <p className="mt-3 text-sm text-fake">Could not reach the API. It may be waking up; try again in a minute.</p>}
      {state === "ok" && rows && (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[560px] text-left text-sm">
            <thead><tr className="border-b border-slate-200 text-slate-500"><th className="py-2 pr-4">from the database</th><th className="py-2 pr-4">released</th><th className="py-2 text-brand">self-calibrated</th></tr></thead>
            <tbody className="text-slate-800">
              {[
                ["real called FAKE (damaged, mean)", "deg_mean_fpr@0.5"], ["fakes caught (damaged, mean)", "deg_mean_tpr@0.5"],
                ["balanced accuracy", "deg_mean_bacc@0.5"], ["fakes caught at 5% false alarms", "tpr@mfpr05"],
              ].map(([l, k]) => (
                <tr key={k} className="border-b border-slate-100 last:border-0"><td className="py-2 pr-4">{l}</td><td className="py-2 pr-4">{pct(rows.orig[k])}</td><td className="py-2 font-bold text-brand">{pct(rows.mine[k])}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-slate-500">Test split: {rows.n.test_real} real + {rows.n.test_fake} fake videos; fitted on {rows.n.fit_real} + {rows.n.fit_fake} training videos.</p>
        </div>
      )}
    </Card>
  );
}

function Footer() {
  return (
    <footer className="mt-10 border-t border-slate-200 bg-white">
      <div className="mx-auto max-w-6xl px-5 py-8 text-sm text-slate-500">
        B.Tech project by Roshan Raj. Builds on DFD-FCG by Han, Huang, Hua and Chen (CVPR 2025), whose code and weights are for
        research use. Earlier work: <a className="text-brand hover:underline" href={LINKS.earlier}>dfd-fcg-explainability</a>.
      </div>
    </footer>
  );
}
