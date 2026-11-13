import { Fragment, useState } from "react";
import type { ReactNode } from "react";
import fails from "./fails.json";
import scores from "./scores.json";
import upgrade from "./upgrade.json";
import websiteCheck from "./website_check.json";
import { BlurTestFig, FixFig, HeroFig, ModelArchFig, ProtocolFig, ScoreStrips, SystemFig, V2Fig } from "./figs";

const LINKS = {
  checker: "https://dfd-fcg-degrade-to-detect.vercel.app",
  github: "https://github.com/roshanraj9136/dfd-fcg-degrade-to-detect",
  tables: "https://github.com/roshanraj9136/dfd-fcg-degrade-to-detect/blob/main/results/tables.md",
  study: "https://github.com/roshanraj9136/dfd-fcg-degrade-to-detect/blob/main/results/accuracy_study.md",
  paper: "https://github.com/aiiu-lab/DFD-FCG",
};

const NAV: [string, string][] = [
  ["summary", "In short"], ["model", "Model"], ["problem", "Problem"], ["cause", "Cause"], ["fix", "My fix"],
  ["results", "Results"], ["limits", "Where it fails"], ["live", "Live checker"], ["novelty", "Novelty"],
  ["questions", "Questions"], ["script", "Script"],
];

/* ------------------------------------------------------------------ data (all numbers come from these files) */
type S = Record<string, { real: number[]; fake: number[]; real_fix: number[]; fake_fix: number[] }>;
const SC = scores as S;
type Cond = "clean" | "GB_5" | "VC_5";
type UM = { acc: number | null; false_alarm: number | null; caught: number | null; auroc: number | null; bacc: number | null };
type UG = Record<Cond, { orig: UM; v1: UM; v2: UM }> & { n: number; n_fake: number };
type US = { share: number; acc_sure: number | null; acc_unsure: number | null; errors_sure: number; n: number };
const UP = upgrade as unknown as {
  videos: number; settings: { clips: number; window_s: number; cert_margin: number };
  select: { used: number; dfdc: number };
  groups: { ffpp: UG; fsh: UG; dfdc: UG };
  sure: Record<"all" | "ffpp" | "fsh" | "dfdc", US>;
  significance: { ffpp?: { delta: number; ci95: number[] } };
  two_face_summary: { n_fake: number; n_real: number; clean: { caught_v1: number; caught_v2: number } };
};
const WC = websiteCheck as unknown as { videos: number; ffpp: Record<Cond, { same_verdict: number }> };
type FRow = { kind: string; name: string; n: number; label: number } & Record<Cond, { right: number; margin: number }>;
const FL = fails as unknown as { rows: FRow[]; auroc: Record<"ffpp" | "dfdc", Record<Cond, number>> };
const CONDS: [Cond, string][] = [["clean", "As it is"], ["GB_5", "Strong blur"], ["VC_5", "Heavy compression"]];
const pc = (v?: number | null) => (v == null || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(1)}%`);
const p0 = (v?: number | null) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const avg = (g: UG, k: "orig" | "v1" | "v2") => CONDS.reduce((s, [c]) => s + (g[c][k].acc ?? 0), 0) / CONDS.length;

export default function App() {
  return (
    <div className="min-h-screen">
      <Header />
      <main className="mx-auto max-w-6xl px-4 sm:px-5">
        <Hero />
        <Summary />
        <Model />
        <Problem />
        <Cause />
        <Fix />
        <Results />
        <Limits />
        <Live />
        <Novelty />
        <Questions />
        <Script />
      </main>
      <footer className="mt-10 border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-6xl px-5 py-8 text-sm text-slate-600">
          B.Tech project by Roshan Raj. Builds on DFD-FCG by Han, Huang, Hua and Chen (CVPR 2025); its code and weights are for research use.
          Every number on this page comes from the result files in the <a className="text-brand underline" href={LINKS.github}>repository</a>.
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

function Section({ id, n, title, lead, children, tone }: { id: string; n: string; title: ReactNode; lead?: ReactNode; children: ReactNode; tone?: "red" | "green" | "brand" | "amber" }) {
  const k = tone === "red" ? "text-fake" : tone === "green" ? "text-real" : tone === "amber" ? "text-amber-700" : "text-brand";
  return (
    <section id={id} className="scroll-mt-20 py-12">
      <div className={`text-sm font-bold uppercase tracking-wider ${k}`}>{n}</div>
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
      <div className="text-xs font-bold uppercase tracking-wider text-brand">Say it like this</div>
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
      <table className="w-full min-w-[600px] text-sm">
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

const Up = ({ a, b, lowerBetter = false }: { a: string; b: string; lowerBetter?: boolean }) => {
  const x = parseFloat(a), y = parseFloat(b);
  const good = lowerBetter ? y < x : y > x, bad = lowerBetter ? y > x : y < x;
  return <b className={good ? "text-real" : bad ? "text-fake" : "text-slate-900"}>{b}</b>;
};

/* ------------------------------------------------------------------ sections */
function Hero() {
  return (
    <section id="top" className="grid items-center gap-8 py-10 md:grid-cols-[1.1fr_1fr]">
      <div>
        <div className="text-sm font-semibold text-brand">B.Tech project · Roshan Raj · viva guide</div>
        <h1 className="mt-2 text-3xl font-extrabold leading-tight tracking-tight text-slate-900 md:text-5xl">
          Fake, or just <span className="text-brand">low quality</span>?
        </h1>
        <p className="mt-4 text-lg text-slate-600">
          DFD-FCG (CVPR 2025) is a top video deepfake detector. I showed that it calls <b className="text-slate-800">real videos fake</b> when
          they are blurry or compressed, found <b className="text-slate-800">why</b> inside the model, fixed it <b className="text-slate-800">without
          retraining</b>, and built a public checker that says when it is <b className="text-slate-800">not sure</b>.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <a className="btn-primary" href={LINKS.checker}>Try the live checker</a>
          <a className="btn-ghost" href="#novelty">What is new</a>
        </div>
      </div>
      <Card className="p-4"><HeroFig /></Card>
    </section>
  );
}

function Summary() {
  return (
    <Section id="summary" n="In one minute" title="Three things to remember">
      <div className="grid gap-4 md:grid-cols-3">
        <Big k="The problem" tone="fake" v="63%" text="of real test videos are called FAKE under strong blur (3% on clean video). The paper's robustness number (AUROC) does not show it." />
        <Big k="The cause" v="+1.44 vs −0.17" text="A 2-pixel blur adds a lot of fake evidence to real faces and almost none to fakes. The model treats fine detail as proof of “real”." />
        <Big k="My fix" tone="brand" v="22.5% → 8.7%" text="real videos called fake on damaged video, after a small add-on that removes the quality effect. No retraining." />
      </div>
      <Card className="mt-4">
        <div className="text-xs font-bold uppercase tracking-wider text-slate-500">Words I use</div>
        <dl className="mt-3 grid gap-x-8 gap-y-2 text-sm md:grid-cols-3">
          {[
            ["Fake score", "the model's probability that a video is fake (0–1)."],
            ["Decision line", "0.5: above it the answer is FAKE. This is what a user sees."],
            ["False alarm", "a real video called FAKE."],
            ["Fakes caught", "the share of deepfakes called FAKE."],
            ["AUROC", "chance a random fake scores above a random real video. Ignores the line. 0.5 = coin flip."],
            ["Damage", "blur, resampling, compression, noise and colour changes (the authors' 7 types)."],
          ].map(([a, b]) => <div key={a}><dt className="font-semibold text-slate-900">{a}</dt><dd className="text-slate-600">{b}</dd></div>)}
        </dl>
      </Card>
    </Section>
  );
}

function Model() {
  return (
    <Section id="model" n="The model" title="DFD-FCG, and where my add-on plugs in"
      lead={<>Grey is the authors' model, used exactly as released. Blue is mine. It sits after the model's last features and before the
        decision, so nothing inside DFD-FCG is retrained.</>}>
      <Fig caption="Scroll sideways on a phone. d_s, d_t, d_a: the three head scores. g: the fake evidence predicted to come only from low quality.">
        <ModelArchFig />
      </Fig>
      <ul className="mt-4 grid gap-2 text-sm text-slate-600 md:grid-cols-2">
        <li>• <b className="text-slate-900">CLIP ViT-L/14</b> (frozen): turns each face frame into 16×16 patches through 24 layers.</li>
        <li>• <b className="text-slate-900">Side network</b> (trained by the authors): 4 queries for eyes, nose, lips and skin, plus motion over 10 frames.</li>
        <li>• <b className="text-slate-900">Three heads</b>: spatial, temporal and fused scores, averaged. 0.5 or more = FAKE.</li>
        <li>• <b className="text-brand">My add-on</b>: one linear read-out on the head features predicts the quality effect g and subtracts it.</li>
      </ul>
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
    <Section id="problem" n="What the paper missed" tone="red" title={<>The ranking holds, but the <span className="text-fake">decision line breaks</span></>}
      lead={<>The paper reports robustness as AUROC under each kind of damage, and it stays high. I re-ran the authors' own damage code
        on 130 test videos and looked at the actual REAL / FAKE decisions.</>}>
      <Fig caption={<>Each dot is one test video (70 real, 60 fake). Under strong blur all scores slide right: the order is still decent (AUROC 0.905),
        but most real videos cross the line. With my fix they move back.</>}>
        <ScoreStrips rows={rows} />
      </Fig>
      <div className="mt-4 grid gap-4 md:grid-cols-3">
        <Big k="What the paper sees" v="AUROC 0.999 → 0.946" text="clean → average of 13 damaged versions. Looks robust." />
        <Big k="What a user gets" tone="fake" v="22.5%" text="false alarms on average over the same 13 versions (2.9% clean): 40% medium blur, 63% strong blur, 59% strong resampling, 33% strong H.264." />
        <Big k="The opposite failure" tone="amber" v="98% → 17%" text="fakes caught, clean vs Gaussian noise. Noise hides fakes. My fix does not solve this (13%)." />
      </div>
      <Say>The paper's numbers are correct, but its robustness test cannot see this. AUROC stays high, while at the model's own line
        strong blur turns 63% of real videos into “fakes”.</Say>
    </Section>
  );
}

function Cause() {
  return (
    <Section id="cause" n="Why it happens" title="The model uses fine detail as proof of “real”"
      lead={<>I blurred the aligned face by just 2 pixels and ran the model again. Real faces suddenly look much more fake; fakes hardly change.</>}>
      <Fig><BlurTestFig /></Fig>
      <ul className="mt-4 grid gap-3 text-sm text-slate-600 md:grid-cols-3">
        <li className="rounded-2xl bg-white p-4 shadow-sm"><b className="text-slate-900">Real faces have camera detail.</b> Skin texture and sharp edges are what
          DFD-FCG reads as “real”. On clean video the blur reaction alone separates real from fake (AUROC 0.992).</li>
        <li className="rounded-2xl bg-white p-4 shadow-sm"><b className="text-slate-900">Damage removes that detail.</b> Blur, resampling and compression wipe it out,
          so real videos lose their “proof” and drift to FAKE. Fakes have less detail to lose.</li>
        <li className="rounded-2xl bg-white p-4 shadow-sm"><b className="text-slate-900">Found with an exact split.</b> I split every head score into 240 pieces (layer ×
          face part) that add up to the real score within 0.0000003 on all 5,976 clips, then compared clean and damaged copies. Not a heatmap.</li>
      </ul>
      <Say>A real face reacts to a tiny blur, a fake one doesn't. So the model's “real” evidence is fine detail, and that is exactly
        what low quality destroys.</Say>
    </Section>
  );
}

function Fix() {
  return (
    <Section id="fix" n="My change" tone="brand" title="Self-calibration: the model corrects its own quality bias"
      lead={<>The model's own features already show how damaged a face is. A small read-out turns that into an estimate of the fake
        evidence the damage added, and subtracts it.</>}>
      <Fig><FixFig /></Fig>
      <ul className="mt-4 grid gap-2 text-sm text-slate-600 md:grid-cols-2">
        <li>• <b className="text-slate-900">Fitted on training videos only</b>: 30 real, 19 fake, clean and at damage levels 2 and 4. Tested at levels 1, 3, 5.</li>
        <li>• <b className="text-slate-900">Target</b>: how much the scores of <i>real</i> videos moved under each damage. Real and fake clips get the same target,
          so it cannot learn “push fakes down”.</li>
        <li>• <b className="text-slate-900">Ridge regression</b> (α = 1000) from 1,280 feature numbers to the 3 head shifts. Under a millisecond.</li>
        <li>• <b className="text-slate-900">Tested on damage it never saw</b>: each damage type is left out of fitting in turn. No quality label needed.</li>
        <li>• <b className="text-slate-900">Strict mode</b> (optional): also subtracts half of the blur-test reaction, when a false accusation is costly.</li>
      </ul>
      <Say>I don't change the detector. I add one linear read-out that estimates, from the detector's own features, how much fake
        evidence came only from poor quality, and subtract it.</Say>
    </Section>
  );
}

function Results() {
  return (
    <Section id="results" n="Results" tone="green" title="Fewer false alarms, better ranking across qualities"
      lead={<>FF++ test split: 70 real and 60 fake videos, 13 damaged versions each. The tested damage type is hidden from fitting.</>}>
      <Table head={["", "DFD-FCG as released", "With my fix"]} rows={[
        ["Real called FAKE, damaged video (mean)", "22.5%", <b className="text-real">8.7%</b>],
        ["Real called FAKE, strong blur", "63%", <b className="text-real">11%</b>],
        ["Real called FAKE, medium blur", "40%", <b className="text-real">4%</b>],
        ["Real called FAKE, strong H.264", "33%", <b className="text-real">13%</b>],
        ["Balanced accuracy (damaged, mean)", "82.7%", <b className="text-real">84.7%</b>],
        ["Fakes caught at 5% false alarms", "67.2%", <b className="text-real">71.0%</b>],
        ["AUROC, all qualities mixed", "0.920", <b className="text-real">0.951</b>],
        ["AUROC, clean video", "0.999", "0.999"],
      ]} note={<>A single global re-calibration (Platt scaling) only reaches 12.2% false alarms. Strict mode: 2.3% false alarms, 61.4% of fakes caught.
        All numbers: <a className="text-brand underline" href={LINKS.tables}>results/tables.md</a>.</>} />
      <Card className="mt-4 border-amber-200 bg-amber-50">
        <div className="font-bold text-amber-800">The honest cost</div>
        <ul className="mt-2 grid gap-1.5 text-sm text-slate-700 md:grid-cols-2">
          <li>• At the 0.5 line it catches fewer fakes on damaged video: 87.9% → 78.1%.</li>
          <li>• FaceShifter (a fake type it never saw): AUROC unchanged (0.925), caught at 0.5: 73% → 53%.</li>
          <li>• Gaussian noise still hides fakes (17% → 13% caught). Strong resampling still has 44% false alarms.</li>
          <li>• On DFDC (another dataset) AUROC barely moves: 0.786 → 0.789.</li>
        </ul>
      </Card>
      <Say>On damaged video, wrongly accused real videos drop from 22.5% to 8.7%, and the ranking across mixed qualities improves from
        0.920 to 0.951. The price is that fewer fakes are caught at the default line, which I report.</Say>
    </Section>
  );
}

function Limits() {
  const cell = (v: number) => (
    <span className={`inline-block min-w-[3.5rem] rounded-md px-2 py-0.5 font-semibold ${v >= 0.9 ? "bg-real-soft text-real" : v >= 0.7 ? "bg-amber-50 text-amber-700" : "bg-fake-soft text-fake"}`}>
      {p0(v)}
    </span>
  );
  const groups = [["FF++ videos (what it was trained on)", ["REAL", "DF", "F2F", "FS", "NT"]], ["New fake type", ["FSh"]], ["Another dataset", ["DFDC_R", "DFDC_F"]]] as const;
  const s = UP.sure;
  return (
    <Section id="limits" n="Where it still fails" tone="amber" title="It only knows fakes like the ones it was trained on"
      lead={<>Right answers of the live checker on the {UP.videos} test videos. When it is wrong, the score usually sits just a few points from
        50%, which is why the checker then says “not sure”.</>}>
      <Card className="overflow-x-auto p-0">
        <table className="w-full min-w-[640px] text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs uppercase tracking-wider text-slate-500">
              <th className="px-4 py-3">Kind of video</th><th className="px-3 py-3 text-right">Videos</th>
              {CONDS.map(([, l]) => <th key={l} className="px-3 py-3 text-right">{l}</th>)}
            </tr>
          </thead>
          <tbody>
            {groups.map(([g, kinds]) => (
              <Fragment key={g}>
                <tr className="border-t border-slate-200"><td colSpan={5} className="px-4 pb-1 pt-3 text-xs font-bold uppercase tracking-wider text-slate-500">{g}</td></tr>
                {FL.rows.filter((r) => (kinds as readonly string[]).includes(r.kind)).map((r) => (
                  <tr key={r.kind} className="border-t border-slate-100">
                    <td className="px-4 py-1.5 font-medium text-slate-900">{r.name}</td>
                    <td className="px-3 py-1.5 text-right tabular-nums text-slate-500">{r.n}</td>
                    {CONDS.map(([c]) => <td key={c} className="px-3 py-1.5 text-right tabular-nums">{cell(r[c].right)}</td>)}
                  </tr>
                ))}
              </Fragment>
            ))}
          </tbody>
        </table>
        <p className="px-4 pb-4 pt-2 text-xs text-slate-500">
          Can the score separate real from fake at all (AUROC; 0.5 = coin flip)? FF++: {FL.auroc.ffpp.clean.toFixed(2)} as it is,
          {" "}{FL.auroc.ffpp.GB_5.toFixed(2)} blurred, {FL.auroc.ffpp.VC_5.toFixed(2)} compressed. DFDC: {FL.auroc.dfdc.clean.toFixed(2)},
          {" "}<b className="text-fake">{FL.auroc.dfdc.GB_5.toFixed(2)}</b> and <b className="text-fake">{FL.auroc.dfdc.VC_5.toFixed(2)}</b>.
        </p>
      </Card>
      <div className="mt-4 grid gap-4 md:grid-cols-3">
        <Card><div className="font-bold text-slate-900">1. Old, narrow training data</div><p className="mt-1 text-sm text-slate-600">
          DFD-FCG learned from about 1,000 YouTube news-style videos and 4 fake methods from 2018–19. It has never seen vertical phone video,
          social-media re-compression, newer face-swap apps or fully AI-generated video.</p></Card>
        <Card><div className="font-bold text-slate-900">2. Damage erases the clues</div><p className="mt-1 text-sm text-slate-600">
          The model reads fine detail. Blur and compression remove it. My fix stops real videos being accused, but it cannot bring back
          evidence that is gone: on damaged DFDC video the score is a coin flip.</p></Card>
        <Card><div className="font-bold text-slate-900">3. The model was not retrained</div><p className="mt-1 text-sm text-slate-600">
          Everything I added changes how the score is <i>read</i>, not what the model knows. Moving the line, more clips, more faces and
          mirrored clips did not raise test accuracy.</p></Card>
      </div>
      <Card className="mt-4 border-amber-200 bg-amber-50">
        <div className="font-bold text-amber-800">A real example from the live site</div>
        <p className="mt-1 text-sm text-slate-700">A fake vertical phone video (480×854) uploaded to the checker scored <b>48% fake</b>: right on the line. That is the
          pattern above. The checker now shows such answers as <b>NOT SURE</b> instead of a confident REAL. Over all test answers, “sure”
          answers are right {pc(s.all.acc_sure)} of the time and “not sure” ones {pc(s.all.acc_unsure)}; on DFDC only {pc(s.dfdc.share)} of answers are “sure”.</p>
      </Card>
      <Card className="mt-4">
        <div className="font-bold text-slate-900">What would make it surer</div>
        <p className="mt-1 text-sm text-slate-600">Retraining the authors' side network (0.25 M parameters; CLIP stays frozen) on newer data. I checked that it fits
          my 4 GB GPU (1.6 GB, 0.33 s per training step). Next step, in progress: fine-tuning on Celeb-DF v2, more DFDC parts, 2026 social-media
          deepfakes and 2026 AI-generated talking heads, with the same {UP.videos} test videos kept out of training for a fair before/after.</p>
      </Card>
      <Say>On videos unlike its training data, DFD-FCG's ranking drops to a coin flip. My system detects when it is outside what it knows
        and says “not sure”, instead of giving a confident wrong answer.</Say>
    </Section>
  );
}

function Live() {
  const f = UP.groups.ffpp, d = UP.groups.dfdc, s = UP.sure, st = UP.settings, tf = UP.two_face_summary;
  const ci = UP.significance.ffpp?.ci95;
  return (
    <Section id="live" n="The live checker" title="A public website, tested on the paper's own videos"
      lead={<>Upload a face video, get REAL / FAKE / NOT SURE with a short reason. Faces are found on the server's CPU; the detector runs on a
        free Hugging Face GPU, or on the CPU when the free GPU time is used up (same answer, slower).</>}>
      <Fig caption="CPU / GPU tags show where each step runs."><SystemFig /></Fig>
      <h3 className="mt-8 text-xl font-bold text-slate-900">Version 2: reads more, checks everyone, says when it is not sure</h3>
      <p className="mt-1 max-w-3xl text-sm text-slate-600">Version 1 used the research layout (2 clips, one face) and gave the research verdict on {pc(WC.ffpp.clean.same_verdict)} of
        normal FF++ videos. Version 2 uses {st.clips} clips from {st.window_s} s, every clear face, and “not sure” within {Math.round(st.cert_margin * 100)} points of the line.
        All settings were fixed on {UP.select.used} separate videos, then tested once on the {UP.videos} test videos.</p>
      <div className="mt-4 grid gap-4">
        <Fig><V2Fig /></Fig>
        <Fig><ProtocolFig select={UP.select.used} test={UP.videos} /></Fig>
      </div>
      <div className="mt-4">
        <Table head={["Same test videos", "Version 1", "Version 2"]} rows={[
          ["FF++ accuracy, as it is", pc(f.clean.v1.acc), <Up a={pc(f.clean.v1.acc)} b={pc(f.clean.v2.acc)} />],
          ["FF++ accuracy, strong blur", pc(f.GB_5.v1.acc), <Up a={pc(f.GB_5.v1.acc)} b={pc(f.GB_5.v2.acc)} />],
          ["FF++ accuracy, heavy compression", pc(f.VC_5.v1.acc), <Up a={pc(f.VC_5.v1.acc)} b={pc(f.VC_5.v2.acc)} />],
          ["DFDC AUROC, as it is", d.clean.v1.auroc?.toFixed(3) ?? "–", <Up a={d.clean.v1.auroc?.toFixed(3) ?? "0"} b={d.clean.v2.auroc?.toFixed(3) ?? "0"} />],
          ["Two-person fakes caught (normal video)", `${tf.clean.caught_v1} of ${tf.n_fake}`, <b className="text-real">{tf.clean.caught_v2} of {tf.n_fake}</b>],
          ["Right when it says “sure”", "–", <b>{pc(s.all.acc_sure)} ({pc(s.all.share)} of answers)</b>],
          ["Right when it says “not sure”", "–", pc(s.all.acc_unsure)],
        ]} note={<>Overall accuracy did not improve: FF++ average {pc(avg(f, "v1"))} → {pc(avg(f, "v2"))}{ci ? `, a change within noise (bootstrap 95% range ${(ci[0] * 100).toFixed(1)} to +${(ci[1] * 100).toFixed(1)} points)` : ""}.
          Version 2's gains are the two-person fakes, better DFDC ranking and the honest “not sure”. Details: <a className="text-brand underline" href={LINKS.study}>accuracy_study.md</a>.</>} />
      </div>
      <Say>I ran the website's own code on all {UP.videos} test videos. Accuracy at the line stayed the same, but it now catches the two-person
        fakes it used to miss, and when it says “sure” it is right {pc(s.all.acc_sure)} of the time.</Say>
    </Section>
  );
}

const NEW: [string, ReactNode, string, ReactNode][] = [
  ["A failure the paper does not report", "63% of real videos called FAKE under strong blur, while AUROC still reads 0.905.",
    "Everyone knows compression hurts detectors.",
    "In general, yes. The DFD-FCG paper reports this model as robust with AUROC. I show, with the authors' own damage code, that its decisions are not, and by how much."],
  ["The cause, measured exactly", "A 2-px blur adds +1.44 fake evidence to real faces, −0.17 to fakes. Exact split, checked on 5,976 clips.",
    "Explanation maps are unreliable.",
    "Heatmaps approximate. Mine is exact algebra: the heads are linear after a LayerNorm, so the score splits into a sum that I check on every clip."],
  ["A fix without retraining", "False alarms 22.5% → 8.7% on damage types left out of fitting; mixed AUROC 0.920 → 0.951.",
    "Isn't this just moving the threshold?",
    "No. A threshold or Platt scaling shifts every video equally (12.2%). My read-out shifts each video by its own amount, from its own features."],
  ["An answer that says when it is not sure", <>Right {pc(UP.sure.all.acc_sure)} of the time when “sure”, {pc(UP.sure.all.acc_unsure)} when “not sure”.</>,
    "Abstaining is an old idea.",
    "Yes (Chow 1970; Geifman and El-Yaniv 2017). New here: built on the quality-corrected score, the margin fixed on separate videos, checked on damaged and other-dataset video."],
  ["A public checker, tested on the paper's videos", <>Website code run on all {UP.videos} test videos; settings chosen on {UP.select.used} other videos.</>,
    "Anyone can make a website.",
    "The website is not the novelty. It is the proof that the method works end to end, with numbers measured on the paper's test set."],
];

function Novelty() {
  return (
    <Section id="novelty" n="Defend the novelty" tone="brand" title="What is new, the proof, and what to say when they push back"
      lead={<>Open with: “The paper says DFD-FCG is robust because AUROC stays high. I showed its decisions fail on low-quality real videos,
        found why inside the model, fixed it without retraining, and built a checker that says when it is not sure.”</>}>
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
          <div className="font-bold text-slate-900">Not mine (say it before they ask)</div>
          <ul className="mt-2 space-y-1 text-sm text-slate-600">
            <li>• The DFD-FCG model, training and weights (Han et al., CVPR 2025), and the damage code.</li>
            <li>• The FaceForensics++, FaceShifter and DFDC videos.</li>
            <li>• Ridge regression, Platt scaling and “saying not sure” are standard tools.</li>
            <li>• More clips and checking every face are common practice.</li>
          </ul>
        </Card>
        <Card>
          <div className="font-bold text-slate-900">Ideas that failed (also reported)</div>
          <ul className="mt-2 space-y-1 text-sm text-slate-600">
            <li>• Quality from the background, or from the face crop (15.1% false alarms on unseen types).</li>
            <li>• The blur test alone; removing a “quality direction” from the explanation pieces.</li>
            <li>• Mirrored clips, clip agreement as a doubt signal, and moving the 0.5 line: no test gain.</li>
          </ul>
        </Card>
      </div>
    </Section>
  );
}

const QA: [string, ReactNode][] = [
  ["What exactly is wrong in the paper?",
    "Its numbers are right; its robustness test is incomplete. AUROC only checks the order of scores. A user sees the decision at 0.5, and there strong blur flips 63% of real videos to FAKE while AUROC still reads 0.905."],
  ["Why not simply move the threshold?",
    "The right line depends on how damaged each video is, which you don't know. One global re-calibration still gives 12.2% false alarms; my per-video read-out gives 8.7%. For the live checker I also tested other lines on separate videos: the best one (0.46) was worse on the test videos."],
  ["Did you tune anything on the test set?",
    "No. The read-out is fitted on training-split videos at damage levels 2 and 4 and tested at 1, 3 and 5, with the tested damage type left out. The live-checker settings were fixed on 117 separate videos before the test run."],
  ["Your method catches fewer fakes. Isn't that worse?",
    "At the 0.5 line, yes (87.9% → 78.1%). At the same false-alarm rate it catches more (71.0% vs 67.2% at 5%), and balanced accuracy and mixed AUROC go up. The released model catches many fakes partly because it calls almost every damaged video FAKE."],
  ["You did not train anything. Is this research?",
    "Yes. The questions are measurement and cause: does the decision survive damage, and why not? The fix is training-free on purpose, so it works on the released weights."],
  ["Why should I trust your explanation?",
    "Because it is exact: the 240 pieces add up to the model's actual score within 0.0000003 on every one of 5,976 clips. A heatmap cannot be checked like that."],
  ["Why does the checker fail on some uploaded videos?",
    "DFD-FCG only learned 2018–19 face swaps in news-style video. On new kinds of video (phone clips, new apps, AI-generated video) its evidence is balanced and the score lands near 50%. On damaged DFDC video its AUROC is 0.51, a coin flip. That is why the checker says “not sure” there."],
  ["Isn't “not sure” a way to hide mistakes?",
    "No. The probable answer is still shown, and I report accuracy on all answers too. The label is honest only if “not sure” answers really are worse, and they are."],
  ["What did you build yourself?",
    "The test pipeline around the authors' code, the exact attribution engine, the evaluation, the self-calibration and strict mode, the live checker (backend, website, CPU fallback, version 2 study) and this guide."],
  ["What would you do next?",
    "Fine-tune the authors' side network on newer data (Celeb-DF, more DFDC, 2026 social-media and AI-generated videos), keeping the same test videos out, and fit the read-out on more kinds of damage."],
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
  const lines = [
    "DFD-FCG is a CVPR 2025 video deepfake detector built on CLIP. The paper says it is robust to blur and compression, because its AUROC stays high.",
    "I tested its real decisions at the 0.5 line, with the authors' own damage code. Under strong blur it calls 63% of real videos fake.",
    "An exact split of the score showed why: the model uses fine natural detail as proof of “real”. A 2-pixel blur adds +1.44 fake evidence to real faces and almost nothing to fakes.",
    "So I added one linear read-out that estimates, from the model's own features, how much fake evidence came only from poor quality, and subtracts it. No retraining.",
    "False alarms on damaged video drop from 22.5% to 8.7%, also for damage it never saw, and mixed-quality AUROC rises from 0.920 to 0.951. The cost: fewer fakes caught at the default line.",
    `I built a public checker and tested its code on all ${UP.videos} test videos. On videos unlike its training data the model is near a coin flip, so the checker says “not sure” there; when it says “sure” it is right ${pc(UP.sure.all.acc_sure)} of the time.`,
  ];
  return (
    <Section id="script" n="Practise this" title="The one-minute script">
      <Card className="p-6 md:p-8">
        <ol className="space-y-3">
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
