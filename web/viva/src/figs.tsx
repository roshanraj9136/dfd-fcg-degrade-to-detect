/* Hand-made SVG figures for the viva page. Colours: slate = original DFD-FCG, indigo = my addition,
   red = fake / problem, green = real / fixed. */
import type { ReactNode } from "react";

const C = {
  ink: "#0f172a", text: "#334155", muted: "#64748b", line: "#94a3b8", soft: "#e2e8f0", panel: "#f1f5f9",
  brand: "#4f46e5", brandSoft: "#eef2ff", real: "#16a34a", realSoft: "#dcfce7", fake: "#dc2626", fakeSoft: "#fee2e2",
  amber: "#d97706",
};

function Defs() {
  return (
    <defs>
      <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0L10 5L0 10z" fill={C.line} />
      </marker>
      <marker id="ahb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0L10 5L0 10z" fill={C.brand} />
      </marker>
      <marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0L10 5L0 10z" fill={C.fake} />
      </marker>
      <filter id="blur3"><feGaussianBlur stdDeviation="2.6" /></filter>
    </defs>
  );
}

function T({ x, y, children, size = 14, w = 400, color = C.text, anchor = "start", italic = false }:
  { x: number; y: number; children: ReactNode; size?: number; w?: number; color?: string; anchor?: "start" | "middle" | "end"; italic?: boolean }) {
  return <text x={x} y={y} fontSize={size} fontWeight={w} fill={color} textAnchor={anchor} fontStyle={italic ? "italic" : undefined}>{children}</text>;
}

function Box({ x, y, w, h, fill = "#fff", stroke = C.line, dash, sw = 1.5, r = 14 }:
  { x: number; y: number; w: number; h: number; fill?: string; stroke?: string; dash?: string; sw?: number; r?: number }) {
  return <rect x={x} y={y} width={w} height={h} rx={r} fill={fill} stroke={stroke} strokeWidth={sw} strokeDasharray={dash} />;
}

function Arrow({ d, color = "grey", dash, sw = 2 }: { d: string; color?: "grey" | "brand" | "red"; dash?: string; sw?: number }) {
  const stroke = color === "brand" ? C.brand : color === "red" ? C.fake : C.line;
  const m = color === "brand" ? "url(#ahb)" : color === "red" ? "url(#ahr)" : "url(#ah)";
  return <path d={d} fill="none" stroke={stroke} strokeWidth={sw} strokeDasharray={dash} markerEnd={m} strokeLinecap="round" strokeLinejoin="round" />;
}

/** A simple face; blurred when `blur`. */
export function Face({ x, y, s = 1, blur = false, tint = "#fcd9b6" }: { x: number; y: number; s?: number; blur?: boolean; tint?: string }) {
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`} filter={blur ? "url(#blur3)" : undefined}>
      <ellipse cx="0" cy="0" rx="30" ry="36" fill={tint} stroke="#c08457" strokeWidth="1.5" />
      <path d="M-30 -8 C-30 -40 30 -40 30 -8 C22 -26 -22 -26 -30 -8z" fill="#6b4f3a" />
      <circle cx="-11" cy="-2" r="3.2" fill="#1e293b" />
      <circle cx="11" cy="-2" r="3.2" fill="#1e293b" />
      <path d="M0 2 L-3 12 L3 12" fill="none" stroke="#c08457" strokeWidth="1.5" />
      <path d="M-10 20 Q0 27 10 20" fill="none" stroke="#b91c1c" strokeWidth="2.4" strokeLinecap="round" />
      {!blur && <g stroke="#e8b48a" strokeWidth=".8">{[-18, -14, 14, 18].map((dx) => <path key={dx} d={`M${dx} 8 l2 3`} />)}</g>}
    </g>
  );
}

function Verdict({ x, y, real, label }: { x: number; y: number; real: boolean; label: string }) {
  return (
    <g>
      <rect x={x} y={y} width="118" height="36" rx="18" fill={real ? C.realSoft : C.fakeSoft} stroke={real ? C.real : C.fake} strokeWidth="1.5" />
      <T x={x + 59} y={y + 23} size={15} w={800} color={real ? C.real : C.fake} anchor="middle">{label}</T>
    </g>
  );
}

/* ------------------------------------------------------------------ hero */
export function HeroFig() {
  return (
    <svg viewBox="0 0 560 300" className="w-full" role="img" aria-label="A real face is blurred; the original detector calls it fake, the corrected one real">
      <Defs />
      <Box x={10} y={40} w={150} h={170} fill="#fff" stroke={C.soft} />
      <Face x={85} y={118} s={1.25} />
      <T x={85} y={232} size={14} w={600} color={C.ink} anchor="middle">real video</T>
      <Arrow d="M168 125 H 214" />
      <T x={191} y={112} size={12} color={C.muted} anchor="middle">blur</T>
      <Box x={222} y={40} w={150} h={170} fill="#fff" stroke={C.soft} />
      <Face x={297} y={118} s={1.25} blur />
      <T x={297} y={232} size={14} w={600} color={C.ink} anchor="middle">same video, blurred</T>
      <Arrow d="M380 100 C 410 100, 410 78, 425 78" />
      <Arrow d="M380 150 C 410 150, 410 178, 425 178" color="brand" />
      <T x={430} y={50} size={12} color={C.muted}>DFD-FCG as released</T>
      <Verdict x={430} y={60} real={false} label="FAKE ✗" />
      <T x={430} y={152} size={12} color={C.brand} w={600}>with my add-on</T>
      <Verdict x={430} y={160} real label="REAL ✓" />
      <T x={280} y={285} size={13} color={C.muted} anchor="middle" italic>Under strong blur, 63% of real test videos are called FAKE. With the add-on: 11%.</T>
    </svg>
  );
}

/* ------------------------------------------------------------------ architecture: the model */
export function ModelArchFig() {
  const stripes = Array.from({ length: 24 }, (_, i) => 160 + i * 11);
  const taps = [2, 6, 10, 14, 18, 22];
  return (
    <svg viewBox="0 0 1240 610" className="w-full min-w-[900px]" role="img" aria-label="DFD-FCG architecture with my two additions">
      <Defs />
      {/* legend */}
      <rect x="16" y="14" width="16" height="16" rx="4" fill={C.panel} stroke={C.line} />
      <T x={40} y={27} size={13}>original DFD-FCG (as released by the authors)</T>
      <rect x="330" y="14" width="16" height="16" rx="4" fill={C.brandSoft} stroke={C.brand} strokeWidth="2" />
      <T x={354} y={27} size={13} color={C.brand} w={700}>my addition (no retraining)</T>

      {/* input */}
      <Box x={16} y={180} w={160} h={190} />
      {[0, 1, 2].map((k) => <rect key={k} x={44 + k * 10} y={200 + k * 10} width="84" height="84" rx="8" fill="#fff" stroke={C.line} />)}
      <Face x={106} y={262} s={0.95} />
      <T x={96} y={326} size={15} w={700} color={C.ink} anchor="middle">Input video</T>
      <T x={96} y={345} size={12.5} color={C.muted} anchor="middle">10 aligned face frames</T>
      <T x={96} y={361} size={12.5} color={C.muted} anchor="middle">over 3 seconds</T>
      <Arrow d="M178 275 H 204" />

      {/* CLIP */}
      <Box x={208} y={80} w={200} h={380} fill={C.panel} />
      <T x={308} y={108} size={16} w={800} color={C.ink} anchor="middle">CLIP ViT-L/14</T>
      <T x={308} y={127} size={12.5} color={C.muted} anchor="middle">image model, 24 layers</T>
      <T x={308} y={144} size={12.5} color={C.muted} anchor="middle">frozen (never changed)</T>
      {stripes.map((y, i) => <rect key={i} x={233} y={y} width="150" height="8" rx="3" fill={taps.includes(i) ? "#94a3b8" : "#cbd5e1"} />)}
      <T x={308} y={447} size={12} color={C.muted} anchor="middle">each frame = 16×16 patches</T>
      {taps.map((i) => <Arrow key={i} d={`M385 ${stripes[i] + 4} H 452`} sw={1.4} />)}

      {/* side network */}
      <Box x={458} y={80} w={270} h={380} fill="#f8fafc" />
      <T x={593} y={108} size={16} w={800} color={C.ink} anchor="middle">Side network</T>
      <T x={593} y={127} size={12.5} color={C.muted} anchor="middle">trained by the authors, reads every layer</T>
      <Box x={476} y={142} w={234} h={142} r={12} stroke="#cbd5e1" />
      <T x={593} y={166} size={14.5} w={700} color={C.ink} anchor="middle">Spatial module</T>
      <T x={593} y={185} size={12.5} anchor="middle">4 face-part queries</T>
      {["eyes", "nose", "lips", "skin"].map((p, k) => (
        <g key={p}>
          <rect x={486 + k * 55} y={197} width="50" height="26" rx="13" fill={C.panel} stroke="#cbd5e1" />
          <T x={511 + k * 55} y={214.5} size={12} w={600} anchor="middle">{p}</T>
        </g>
      ))}
      <T x={593} y={248} size={12} color={C.muted} anchor="middle">the FCG loss keeps each query</T>
      <T x={593} y={264} size={12} color={C.muted} anchor="middle">on its own face part</T>
      <Box x={476} y={298} w={234} h={146} r={12} stroke="#cbd5e1" />
      <T x={593} y={322} size={14.5} w={700} color={C.ink} anchor="middle">Temporal module</T>
      <T x={593} y={341} size={12.5} anchor="middle">how each patch changes</T>
      <T x={593} y={357} size={12.5} anchor="middle">from one frame to the next</T>
      {[0, 1].map((k) => (
        <g key={k} transform={`translate(${532 + k * 90} 370)`}>
          {Array.from({ length: 16 }, (_, i) => <rect key={i} x={(i % 4) * 9} y={Math.floor(i / 4) * 9} width="8" height="8" rx="1.5" fill={i === 5 || i === 6 ? (k ? "#a5b4fc" : "#c7d2fe") : "#e2e8f0"} />)}
        </g>
      ))}
      <Arrow d="M572 386 H 616" sw={1.5} />

      {/* features */}
      <rect x="752" y="170" width="146" height="232" rx="16" fill="none" stroke={C.brand} strokeWidth="1.5" strokeDasharray="5 5" />
      <Arrow d="M710 213 H 764" />
      <Arrow d="M710 371 H 764" />
      <Box x={766} y={190} w={118} h={46} r={23} />
      <T x={825} y={210} size={14} w={800} color={C.ink} anchor="middle">y_s</T>
      <T x={825} y={227} size={11.5} color={C.muted} anchor="middle">1024 numbers</T>
      <Box x={766} y={348} w={118} h={46} r={23} />
      <T x={825} y={368} size={14} w={800} color={C.ink} anchor="middle">y_t</T>
      <T x={825} y={385} size={11.5} color={C.muted} anchor="middle">256 numbers</T>

      {/* heads */}
      {[["Spatial head", 150, "d_s"], ["Fusion head", 260, "d_a"], ["Temporal head", 370, "d_t"]].map(([n, y, d]) => (
        <g key={n as string}>
          <Box x={930} y={y as number} w={132} h={46} r={12} fill={C.panel} />
          <T x={996} y={(y as number) + 28} size={13.5} w={700} color={C.ink} anchor="middle">{n}</T>
          <T x={1066} y={(y as number) + 15} size={12} color={C.muted}>{d}</T>
          <Arrow d={`M1062 ${(y as number) + 23} H 1124`} />
          <circle cx="1104" cy={(y as number) + 23} r="11" fill="#fff" stroke={C.brand} strokeWidth="2" />
          <T x={1104} y={(y as number) + 28} size={15} w={800} color={C.brand} anchor="middle">−</T>
        </g>
      ))}
      <Arrow d="M884 213 C 905 213, 905 173, 928 173" />
      <Arrow d="M884 371 C 905 371, 905 393, 928 393" />
      <Arrow d="M884 220 C 912 230, 905 283, 928 283" sw={1.5} />
      <Arrow d="M884 364 C 912 354, 905 283, 928 283" sw={1.5} />

      {/* decision */}
      <Box x={1128} y={150} w={100} h={266} fill={C.panel} />
      <T x={1178} y={240} size={13} anchor="middle">average of</T>
      <T x={1178} y={257} size={13} anchor="middle">the 3 scores</T>
      <T x={1178} y={281} size={13} anchor="middle">→ P(fake)</T>
      <T x={1178} y={312} size={14} w={800} color={C.ink} anchor="middle">≥ 0.5</T>
      <T x={1178} y={331} size={14} w={800} color={C.fake} anchor="middle">= FAKE</T>

      {/* my additions */}
      <path d="M1104 404 V 420" stroke={C.brand} strokeWidth="2" strokeDasharray="4 4" />
      <path d="M1104 184 V 272 M1104 294 V 382" stroke={C.brand} strokeWidth="2" strokeDasharray="4 4" />
      <Arrow d="M825 404 V 478" color="brand" />
      <T x={835} y={448} size={12} color={C.brand} w={600}>y_s, y_t</T>
      <Box x={740} y={482} w={330} h={116} fill={C.brandSoft} stroke={C.brand} sw={2.2} />
      <rect x="756" y="494" width="44" height="20" rx="10" fill={C.brand} />
      <T x={778} y={508} size={11} w={800} color="#fff" anchor="middle">NEW</T>
      <T x={808} y={509} size={15} w={800} color={C.brand}>Self-calibration</T>
      <T x={756} y={535} size={12.5} color={C.ink}>a small linear read-out predicts g: the fake</T>
      <T x={756} y={553} size={12.5} color={C.ink}>evidence that blur or compression added,</T>
      <T x={756} y={571} size={12.5} color={C.ink}>and subtracts it from every head score: d − g</T>
      <T x={756} y={589} size={11.5} color={C.muted}>3 × 1280 weights · fitted on training videos</T>
      <Arrow d="M1070 540 H 1104 V 418" color="brand" />

      <Box x={380} y={482} w={330} h={116} fill="#fff" stroke={C.brand} sw={1.8} dash="6 5" />
      <rect x="396" y="494" width="44" height="20" rx="10" fill={C.brand} />
      <T x={418} y={508} size={11} w={800} color="#fff" anchor="middle">NEW</T>
      <T x={448} y={509} size={15} w={800} color={C.brand}>Blur test (strict mode)</T>
      <T x={396} y={535} size={12.5} color={C.ink}>runs the model once more on the face</T>
      <T x={396} y={553} size={12.5} color={C.ink}>blurred by 2 px; real faces react strongly,</T>
      <T x={396} y={571} size={12.5} color={C.ink}>fakes hardly. Strict mode also subtracts</T>
      <T x={396} y={589} size={12.5} color={C.ink}>half of that reaction. Optional.</T>
      <Arrow d="M710 540 H 736" color="brand" dash="5 4" />
    </svg>
  );
}

/* ------------------------------------------------------------------ architecture: the deployed system */
export function SystemFig() {
  const step = (y: number, n: string, t: string, mine = false, where: "CPU" | "GPU" = "CPU") => (
    <g key={n}>
      <rect x={560} y={y} width={300} height={30} rx={8} fill={mine ? C.brandSoft : C.panel} stroke={mine ? C.brand : "#cbd5e1"} />
      <T x={574} y={y + 20} size={12.5} w={700} color={mine ? C.brand : C.ink}>{n}</T>
      <T x={596} y={y + 20} size={12.5} color={mine ? C.brand : C.text}>{t}</T>
      <rect x={814} y={y + 6} width={40} height={18} rx={9} fill={where === "GPU" ? C.amber : "#e2e8f0"} />
      <T x={834} y={y + 19} size={10.5} w={800} color={where === "GPU" ? "#ffffff" : C.text} anchor="middle">{where}</T>
    </g>
  );
  return (
    <svg viewBox="0 0 1240 500" className="w-full min-w-[900px]" role="img" aria-label="How the live system is deployed">
      <Defs />
      {/* user */}
      <Box x={16} y={180} w={150} h={120} />
      <circle cx="91" cy="214" r="16" fill={C.panel} stroke={C.line} strokeWidth="1.5" />
      <path d="M63 258 C 63 236, 119 236, 119 258" fill={C.panel} stroke={C.line} strokeWidth="1.5" />
      <T x={91} y={284} size={14} w={700} color={C.ink} anchor="middle">You (browser)</T>

      {/* vercel */}
      <Box x={230} y={60} w={250} h={120} />
      <T x={248} y={88} size={12} w={700} color={C.muted}>VERCEL</T>
      <T x={248} y={112} size={16} w={800} color={C.ink}>Checker website</T>
      <T x={248} y={133} size={12.5}>upload a video, see REAL / FAKE</T>
      <T x={248} y={152} size={12.5}>and how the answer was reached</T>
      <Box x={230} y={320} w={250} h={120} />
      <T x={248} y={348} size={12} w={700} color={C.muted}>VERCEL</T>
      <T x={248} y={372} size={16} w={800} color={C.ink}>Viva website (this page)</T>
      <T x={248} y={393} size={12.5}>plain explanation, figures,</T>
      <T x={248} y={412} size={12.5}>live numbers from the database</T>
      <Arrow d="M166 225 C 195 225, 195 120, 226 120" />
      <Arrow d="M166 255 C 195 255, 195 380, 226 380" />

      {/* HF space */}
      <Box x={540} y={30} w={340} h={260} fill="#fffbeb" stroke={C.amber} />
      <T x={558} y={56} size={12} w={700} color={C.amber}>HUGGING FACE SPACE · ZEROGPU</T>
      <T x={558} y={80} size={16} w={800} color={C.ink}>Checker backend (Gradio API)</T>
      {step(96, "1", "find and align every face")}
      {step(132, "2", "DFD-FCG, 8 clips per face", false, "GPU")}
      {step(168, "3", "self-calibration (my add-on)", true)}
      {step(204, "4", "blur test (my add-on)", true, "GPU")}
      {step(240, "5", "sure / not sure + evidence map", true)}
      <Arrow d="M480 104 H 536" />
      <T x={508} y={96} size={11.5} color={C.muted} anchor="middle">video</T>
      <Arrow d="M538 150 H 484" />
      <T x={511} y={168} size={11.5} color={C.muted} anchor="middle">answer</T>

      {/* weights */}
      <Box x={950} y={30} w={274} h={104} />
      <T x={968} y={56} size={12} w={700} color={C.muted}>HUGGING FACE · PRIVATE MODEL REPO</T>
      <T x={968} y={80} size={15} w={800} color={C.ink}>DFD-FCG weights (1.2 GB)</T>
      <T x={968} y={101} size={12.5}>private: the authors allow</T>
      <T x={968} y={119} size={12.5}>research use only</T>
      <Arrow d="M948 82 H 884" />
      <T x={916} y={74} size={11.5} color={C.muted} anchor="middle">at start</T>

      {/* neon */}
      <Box x={950} y={180} w={274} h={150} fill="#ecfdf5" stroke={C.real} />
      <T x={968} y={206} size={12} w={700} color={C.real}>NEON POSTGRES</T>
      <T x={968} y={230} size={15} w={800} color={C.ink}>Database</T>
      <T x={968} y={252} size={12.5}>• anonymous log of every check</T>
      <T x={980} y={270} size={12.5}>(scores only, never the video)</T>
      <T x={968} y={291} size={12.5}>• research results: 226 videos,</T>
      <T x={980} y={309} size={12.5}>15 quality conditions each</T>
      <Arrow d="M880 230 H 946" />
      <T x={913} y={222} size={11.5} color={C.muted} anchor="middle">log</T>

      {/* render */}
      <Box x={540} y={340} w={340} h={100} />
      <T x={558} y={366} size={12} w={700} color={C.muted}>RENDER</T>
      <T x={558} y={390} size={15} w={800} color={C.ink}>Results API (FastAPI)</T>
      <T x={558} y={411} size={12.5}>serves the research numbers from Neon</T>
      <Arrow d="M884 380 C 920 380, 920 300, 946 300" />
      <T x={914} y={346} size={11.5} color={C.muted} anchor="end">read-only</T>
      <Arrow d="M536 400 H 484" />
      <T x={510} y={392} size={11.5} color={C.muted} anchor="middle">numbers</T>

      {/* github */}
      <Box x={950} y={370} w={274} h={70} />
      <T x={968} y={396} size={12} w={700} color={C.muted}>GITHUB</T>
      <T x={968} y={420} size={14} w={700} color={C.ink}>all code, results and figures</T>
      <T x={620} y={482} size={12.5} color={C.muted} anchor="middle" italic>Every part runs on a free tier. Videos are deleted right after a check; only numbers are kept. GPU steps fall back to the CPU when free GPU time runs out.</T>
    </svg>
  );
}

/* ------------------------------------------------------------------ the problem: measured scores */
function jitter(i: number) {
  const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
  return (x - Math.floor(x)) * 2 - 1;
}

export type StripRow = { label: string; sub?: string; real: number[]; fake: number[]; mine?: boolean };

export function ScoreStrips({ rows }: { rows: StripRow[] }) {
  const X0 = 200, X1 = 860, H = 112;
  const px = (p: number) => X0 + (X1 - X0) * p;
  const height = 60 + rows.length * H;
  return (
    <svg viewBox={`0 0 1000 ${height}`} className="w-full min-w-[720px]" role="img" aria-label="Measured fake-scores of real and fake test videos">
      <Defs />
      <T x={px(0)} y={24} size={12} color={C.muted} anchor="middle">0 = surely real</T>
      <T x={px(1)} y={24} size={12} color={C.muted} anchor="middle">1 = surely fake</T>
      <line x1={px(0.5)} x2={px(0.5)} y1={34} y2={height - 14} stroke={C.ink} strokeWidth="1.5" strokeDasharray="5 5" />
      <T x={px(0.5)} y={24} size={12.5} w={700} color={C.ink} anchor="middle">decision line 0.5</T>
      <rect x={px(0.5)} y={34} width={px(1) - px(0.5)} height={height - 48} fill={C.fakeSoft} opacity=".35" />
      {rows.map((r, k) => {
        const y = 44 + k * H;
        const fa = r.real.filter((p) => p >= 0.5).length / r.real.length;
        const ca = r.fake.filter((p) => p >= 0.5).length / r.fake.length;
        return (
          <g key={r.label}>
            <rect x={8} y={y} width={984} height={H - 10} rx={14} fill={r.mine ? C.brandSoft : "#fff"} opacity={r.mine ? 0.6 : 0.0} />
            <T x={20} y={y + 36} size={15} w={800} color={r.mine ? C.brand : C.ink}>{r.label}</T>
            {r.sub && <T x={20} y={y + 56} size={12} color={C.muted}>{r.sub}</T>}
            <line x1={X0} x2={X1} y1={y + 30} y2={y + 30} stroke={C.soft} />
            <line x1={X0} x2={X1} y1={y + 70} y2={y + 70} stroke={C.soft} />
            {r.real.map((p, i) => <circle key={`r${i}`} cx={px(p)} cy={y + 30 + jitter(i + k * 97) * 11} r="4.2" fill={C.real} fillOpacity=".7" />)}
            {r.fake.map((p, i) => <circle key={`f${i}`} cx={px(p)} cy={y + 70 + jitter(i * 3 + k * 31) * 11} r="4.2" fill={C.fake} fillOpacity=".7" />)}
            <T x={X0 - 10} y={y + 34} size={12} color={C.real} w={700} anchor="end">real</T>
            <T x={X0 - 10} y={y + 74} size={12} color={C.fake} w={700} anchor="end">fake</T>
            <T x={878} y={y + 30} size={13} w={800} color={fa > 0.2 ? C.fake : C.real}>{Math.round(fa * 100)}% of real</T>
            <T x={878} y={y + 47} size={12} color={C.muted}>called FAKE</T>
            <T x={878} y={y + 72} size={13} w={800} color={C.ink}>{Math.round(ca * 100)}% of fakes</T>
            <T x={878} y={y + 89} size={12} color={C.muted}>caught</T>
          </g>
        );
      })}
    </svg>
  );
}

/* ------------------------------------------------------------------ simple bar charts */
export type Bar = { label: string; a: number; b?: number; note?: string; fmt?: (v: number) => string };

/** Horizontal bars, values 0..max. `a` = released (grey or red), `b` = mine (indigo). */
export function Bars({ bars, max = 100, unit = "%", aLabel = "DFD-FCG as released", bLabel = "with my add-on", aColor = C.line, lowerIsBetter = true }:
  { bars: Bar[]; max?: number; unit?: string; aLabel?: string; bLabel?: string; aColor?: string; lowerIsBetter?: boolean }) {
  const X0 = 250, W = 560, rowH = bars.some((b) => b.b !== undefined) ? 54 : 36;
  const h = 46 + bars.length * rowH;
  const fmt = (v: number) => (unit === "%" ? `${v}%` : v.toFixed(3));
  return (
    <svg viewBox={`0 0 900 ${h}`} className="w-full" role="img" aria-label={bars.map((b) => `${b.label}: ${b.a}${b.b !== undefined ? ` to ${b.b}` : ""}`).join("; ")}>
      <rect x={X0} y={8} width="14" height="14" rx="3" fill={aColor} />
      <T x={X0 + 20} y={20} size={12.5}>{aLabel}</T>
      {bars.some((b) => b.b !== undefined) && <>
        <rect x={X0 + 210} y={8} width="14" height="14" rx="3" fill={C.brand} />
        <T x={X0 + 230} y={20} size={12.5} color={C.brand} w={700}>{bLabel}</T>
        {lowerIsBetter !== undefined && <T x={X0 + W + 40} y={20} size={12} color={C.muted} anchor="end">{lowerIsBetter ? "lower is better" : "higher is better"}</T>}
      </>}
      {bars.map((b, i) => {
        const y = 38 + i * rowH;
        const f = b.fmt ?? fmt;
        return (
          <g key={b.label + (b.note ?? "")}>
            <T x={X0 - 14} y={y + 15} size={13.5} w={600} color={C.ink} anchor="end">{b.label}</T>
            {b.note && <T x={X0 - 14} y={y + 31} size={11.5} color={C.muted} anchor="end">{b.note}</T>}
            <rect x={X0} y={y + 2} width={Math.max(2, (W * b.a) / max)} height="18" rx="5" fill={aColor} />
            <T x={X0 + Math.max(2, (W * b.a) / max) + 8} y={y + 16} size={13} w={700} color={C.text}>{f(b.a)}</T>
            {b.b !== undefined && <>
              <rect x={X0} y={y + 24} width={Math.max(2, (W * b.b) / max)} height="18" rx="5" fill={C.brand} />
              <T x={X0 + Math.max(2, (W * b.b) / max) + 8} y={y + 38} size={13} w={800} color={C.brand}>{f(b.b)}</T>
            </>}
          </g>
        );
      })}
    </svg>
  );
}

/* ------------------------------------------------------------------ why: the blur test */
export function BlurTestFig() {
  const zeroX = 640, unit = 120;
  const row = (y: number, label: string, v: number, fake: boolean) => (
    <g>
      <Face x={70} y={y} s={0.9} tint={fake ? "#f5cfae" : "#fcd9b6"} />
      <T x={70} y={y + 56} size={13} w={700} color={fake ? C.fake : C.real} anchor="middle">{label}</T>
      <Arrow d={`M118 ${y} H 182`} />
      <T x={150} y={y - 10} size={12} color={C.muted} anchor="middle">2 px blur</T>
      <Face x={232} y={y} s={0.9} blur tint={fake ? "#f5cfae" : "#fcd9b6"} />
      <Arrow d={`M282 ${y} H 340`} />
      <T x={356} y={y - 6} size={13} color={C.text}>change in the</T>
      <T x={356} y={y + 12} size={13} color={C.text}>fake score:</T>
      <rect x={Math.min(zeroX, zeroX + v * unit)} y={y - 16} width={Math.abs(v * unit)} height="32" rx="6" fill={v > 0.5 ? C.fake : C.line} />
      <T x={zeroX + v * unit + (v >= 0 ? 10 : -10)} y={y + 6} size={20} w={800} color={v > 0.5 ? C.fake : C.ink} anchor={v >= 0 ? "start" : "end"}>{v >= 0 ? "+" : "−"}{Math.abs(v).toFixed(2)}</T>
    </g>
  );
  return (
    <svg viewBox="0 0 900 300" className="w-full min-w-[640px]" role="img" aria-label="A 2 pixel blur raises the fake score of real faces by 1.44 and changes fakes by minus 0.17">
      <Defs />
      <line x1={zeroX} x2={zeroX} y1={30} y2={270} stroke={C.ink} strokeWidth="1.5" />
      <T x={zeroX} y={22} size={12} color={C.muted} anchor="middle">no change</T>
      <T x={zeroX + 150} y={22} size={12} color={C.fake} anchor="middle">→ looks more fake</T>
      {row(90, "real face", 1.44, false)}
      {row(210, "deepfake", -0.17, true)}
      <T x={450} y={292} size={12.5} color={C.muted} anchor="middle" italic>Measured on the clean test videos (fusion head, average logit change). Deepfakes −0.81, Face2Face −0.14, FaceSwap +0.08, NeuralTextures +0.16.</T>
    </svg>
  );
}

/* ------------------------------------------------------------------ how I found it: exact attribution */
export function AttributionFig() {
  const raw = Array.from({ length: 40 }, (_, i) => jitter(i * 7 + 3) * (i < 8 ? 2 : 1));
  const P = raw.filter((v) => v > 0).reduce((a, b) => a + b, 0);
  const N = -raw.filter((v) => v < 0).reduce((a, b) => a + b, 0);
  const pieces = raw.map((v) => (v > 0 ? (v / P) * 250 : (v / N) * 150));   // pixels: 250 to FAKE, 150 to REAL
  const X0 = 390;
  let pos = X0, neg = X0;
  const segs = pieces.map((v, i) => {
    const w = Math.abs(v);
    const x = v >= 0 ? pos : neg - w;
    if (v >= 0) pos += w; else neg -= w;
    return <rect key={i} x={x} y={v >= 0 ? 70 : 108} width={w} height="30" fill={v >= 0 ? C.fake : C.real} fillOpacity={0.4 + 0.55 * Math.abs(jitter(i + 11))} stroke="#fff" strokeWidth="1" />;
  });
  return (
    <svg viewBox="0 0 900 236" className="w-full min-w-[640px]" role="img" aria-label="The model's score is an exact sum of 240 pieces">
      <Defs />
      <T x={X0} y={28} size={13.5} w={700} color={C.ink} anchor="middle">one video's fake score, split into its pieces</T>
      <T x={X0} y={48} size={12} color={C.muted} anchor="middle">(drawing: 40 of the 240 layer × face-part pieces)</T>
      <line x1={X0} x2={X0} y1={60} y2={186} stroke={C.ink} strokeWidth="1.5" />
      {segs}
      <T x={pos + 10} y={90} size={12.5} color={C.fake} w={700}>pieces pushing to FAKE</T>
      <T x={neg - 10} y={128} size={12.5} color={C.real} w={700} anchor="end">pieces pushing to REAL</T>
      <rect x={X0} y={158} width={100} height="22" rx="5" fill={C.ink} />
      <T x={X0 + 112} y={174} size={13} w={800} color={C.ink}>their sum = the model's own score</T>
      <T x={450} y={222} size={13} color={C.brand} w={700} anchor="middle">Exact: the pieces add up to the real output within 0.0000003, on all 5,976 clips.</T>
    </svg>
  );
}

/* ------------------------------------------------------------------ my fix */
export function FixFig() {
  return (
    <svg viewBox="0 0 1000 250" className="w-full min-w-[720px]" role="img" aria-label="How the self-calibration works">
      <Defs />
      <Box x={10} y={70} w={170} h={90} fill={C.panel} />
      <T x={95} y={104} size={14} w={800} color={C.ink} anchor="middle">head features</T>
      <T x={95} y={124} size={12.5} anchor="middle">y_s and y_t</T>
      <T x={95} y={142} size={12} color={C.muted} anchor="middle">(already computed)</T>
      <Arrow d="M182 115 H 228" color="brand" />
      <Box x={232} y={50} w={220} h={130} fill={C.brandSoft} stroke={C.brand} sw={2.2} />
      <T x={342} y={82} size={14.5} w={800} color={C.brand} anchor="middle">quality read-out</T>
      <T x={342} y={104} size={12.5} anchor="middle">one linear layer</T>
      <T x={342} y={122} size={12.5} anchor="middle">"how much fake evidence</T>
      <T x={342} y={140} size={12.5} anchor="middle">did low quality add?"</T>
      <T x={342} y={164} size={14} w={800} color={C.brand} anchor="middle">→ g</T>
      <Arrow d="M454 115 H 520" color="brand" />
      <Box x={10} y={196} w={170} h={44} fill={C.panel} />
      <T x={95} y={223} size={13.5} w={700} color={C.ink} anchor="middle">head scores d</T>
      <Arrow d="M182 218 H 540 V 150" />
      <circle cx="545" cy="115" r="24" fill="#fff" stroke={C.brand} strokeWidth="2.5" />
      <T x={545} y={123} size={24} w={800} color={C.brand} anchor="middle">−</T>
      <Arrow d="M571 115 H 640" />
      <Box x={644} y={78} w={160} h={74} />
      <T x={724} y={110} size={14} w={800} color={C.ink} anchor="middle">d − g</T>
      <T x={724} y={130} size={12.5} color={C.muted} anchor="middle">corrected scores</T>
      <Arrow d="M806 115 H 850" />
      <Box x={854} y={78} w={136} h={74} fill={C.panel} />
      <T x={922} y={108} size={13} w={700} color={C.ink} anchor="middle">same 0.5 line</T>
      <T x={922} y={128} size={12.5} anchor="middle">REAL / FAKE</T>
      <T x={500} y={28} size={13} color={C.muted} anchor="middle" italic>DFD-FCG itself is not changed. The add-on is 3 × 1280 weights and runs in under a millisecond.</T>
    </svg>
  );
}

/* ------------------------------------------------------------------ version 2 of the live checker */
/** What version 2 adds around the same model: every person, 10 seconds, and an honest "not sure". */
export function V2Fig() {
  const clip = (x: number, y: number, on: boolean, k: number) => (
    <rect key={k} x={x} y={y} width={22} height={30} rx={4} fill={on ? C.brandSoft : C.panel} stroke={on ? C.brand : "#cbd5e1"} />
  );
  return (
    <svg viewBox="0 0 1180 330" className="w-full min-w-[900px]" role="img" aria-label="What version 2 of the checker does">
      <Defs />
      {/* video with two people */}
      <Box x={16} y={40} w={230} h={170} />
      <T x={32} y={66} size={12} w={700} color={C.muted}>UPLOADED VIDEO</T>
      <rect x={32} y={78} width={198} height={116} rx={10} fill="#e0e7ff" />
      <Face x={92} y={138} s={0.95} />
      <Face x={172} y={138} s={0.95} tint="#f5d0a9" />
      <T x={131} y={238} size={12.5} color={C.muted} anchor="middle">first 10 seconds are read</T>

      {/* step 1: every face */}
      <Arrow d="M250 125 H 300" />
      <Box x={304} y={30} w={250} h={196} fill={C.brandSoft} stroke={C.brand} />
      <T x={320} y={56} size={12} w={800} color={C.brand}>1 · EVERY PERSON (NEW)</T>
      <T x={320} y={80} size={13} color={C.text}>Each clear face that stays in</T>
      <T x={320} y={98} size={13} color={C.text}>the video is tracked and aligned</T>
      <Face x={372} y={160} s={0.8} />
      <Face x={486} y={160} s={0.8} tint="#f5d0a9" />
      <T x={372} y={210} size={12} w={700} color={C.ink} anchor="middle">person 1</T>
      <T x={486} y={210} size={12} w={700} color={C.ink} anchor="middle">person 2</T>

      {/* step 2: 8 clips each */}
      <Arrow d="M558 125 H 600" />
      <Box x={604} y={30} w={270} h={196} />
      <T x={620} y={56} size={12} w={800} color={C.brand}>2 · 8 CLIPS EACH (WAS 2)</T>
      <T x={620} y={80} size={13} color={C.text}>DFD-FCG + my quality correction</T>
      <T x={620} y={98} size={13} color={C.text}>on 8 clips spread over 10 s</T>
      {Array.from({ length: 8 }, (_, i) => clip(620 + i * 30, 120, true, i))}
      {Array.from({ length: 8 }, (_, i) => clip(620 + i * 30, 168, true, 10 + i))}
      <T x={620} y={218} size={11.5} color={C.muted}>top row: person 1 · bottom row: person 2</T>

      {/* step 3: decide */}
      <Arrow d="M878 125 H 920" />
      <Box x={924} y={30} w={240} h={196} fill="#fffbeb" stroke={C.amber} />
      <T x={940} y={56} size={12} w={800} color={C.amber}>3 · DECIDE, HONESTLY (NEW)</T>
      <T x={940} y={80} size={13} color={C.text}>The more suspicious person</T>
      <T x={940} y={98} size={13} color={C.text}>decides (a swap can be on either)</T>
      <rect x={940} y={116} width={208} height={40} rx={20} fill={C.fakeSoft} stroke="#fca5a5" />
      <T x={1044} y={141} size={14} w={800} color={C.fake} anchor="middle">FAKE · sure</T>
      <rect x={940} y={166} width={208} height={40} rx={20} fill="#fef3c7" stroke="#fcd34d" />
      <T x={1044} y={191} size={14} w={800} color={C.amber} anchor="middle">probably FAKE · not sure</T>
      <T x={590} y={290} size={13} color={C.muted} anchor="middle" italic>
        Same model, same weights, same quality correction. Only how much of the video is read, which faces are checked,
      </T>
      <T x={590} y={310} size={13} color={C.muted} anchor="middle" italic>
        and when the answer is flagged as “not sure” are new. Nothing was retrained.
      </T>
    </svg>
  );
}

/** How the version-2 settings were chosen without looking at the reported test videos. */
export function ProtocolFig({ select, test }: { select: number; test: number }) {
  return (
    <svg viewBox="0 0 1180 230" className="w-full min-w-[900px]" role="img" aria-label="How the settings were chosen">
      <Defs />
      <Box x={16} y={30} w={300} h={150} fill={C.panel} />
      <T x={34} y={58} size={12} w={800} color={C.muted}>STEP 1 · SEPARATE VIDEOS</T>
      <T x={34} y={88} size={22} w={800} color={C.ink}>{select} videos</T>
      <T x={34} y={112} size={13}>FF++ test videos not in my results</T>
      <T x={34} y={131} size={13}>+ 20 new DFDC real/fake pairs</T>
      <T x={34} y={158} size={12.5} color={C.muted} italic>never used for any reported number</T>
      <Arrow d="M320 105 H 380" />
      <Box x={384} y={30} w={380} h={150} fill={C.brandSoft} stroke={C.brand} />
      <T x={402} y={58} size={12} w={800} color={C.brand}>STEP 2 · CHOOSE THE SETTINGS THERE</T>
      <T x={402} y={86} size={13.5} color={C.text}>• how many clips, how many seconds</T>
      <T x={402} y={108} size={13.5} color={C.text}>• one face or every face</T>
      <T x={402} y={130} size={13.5} color={C.text}>• mirrored clips: yes or no</T>
      <T x={402} y={152} size={13.5} color={C.text}>• where “sure” starts (the margin)</T>
      <Arrow d="M768 105 H 828" color="brand" />
      <Box x={832} y={30} w={332} h={150} fill="#ecfdf5" stroke={C.real} />
      <T x={850} y={58} size={12} w={800} color={C.real}>STEP 3 · TEST ONCE</T>
      <T x={850} y={88} size={22} w={800} color={C.ink}>{test} videos</T>
      <T x={850} y={112} size={13}>the same FF++, FaceShifter and DFDC</T>
      <T x={850} y={131} size={13}>test videos as all earlier results,</T>
      <T x={850} y={150} size={13}>3 website options each</T>
      <T x={590} y={214} size={13} color={C.muted} anchor="middle" italic>
        The test videos were run only after every setting was fixed, so the test numbers cannot be tuned to look good.
      </T>
    </svg>
  );
}

type CurvePt = { margin: number; all: { share: number | null; acc: number | null }; ffpp: { share: number | null; acc: number | null }; dfdc: { share: number | null; acc: number | null } };

/** Accuracy of the answers marked "sure" against how many answers are marked "sure", for every margin. */
export function CoverageFig({ curve, chosen }: { curve: CurvePt[]; chosen: number }) {
  const W = 760, H = 330, L = 64, R = 24, Tp = 20, B = 54;
  const x = (s: number) => L + (W - L - R) * s;
  const y0 = 0.5;
  const y = (a: number) => Tp + (H - Tp - B) * (1 - (a - y0) / (1 - y0));
  const line = (k: "ffpp" | "dfdc" | "all") => curve.filter((c) => c[k].share != null && c[k].acc != null && c[k].share! > 0.02)
    .map((c, i) => `${i ? "L" : "M"}${x(c[k].share!).toFixed(1)} ${y(c[k].acc!).toFixed(1)}`).join(" ");
  const pick = curve.reduce((a, c) => (Math.abs(c.margin - chosen) < Math.abs(a.margin - chosen) ? c : a), curve[0]);
  const series: ["ffpp" | "dfdc" | "all", string, string][] = [["ffpp", C.brand, "FF++ (the paper's dataset)"], ["dfdc", C.amber, "DFDC (another dataset)"], ["all", C.ink, "all test answers"]];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full min-w-[600px]" role="img" aria-label="Accuracy of sure answers against the share of sure answers">
      {[0.5, 0.6, 0.7, 0.8, 0.9, 1].map((a) => (
        <g key={a}>
          <line x1={L} x2={W - R} y1={y(a)} y2={y(a)} stroke={C.soft} />
          <T x={L - 10} y={y(a) + 4} size={12} color={C.muted} anchor="end">{Math.round(a * 100)}%</T>
        </g>
      ))}
      {[0, 0.25, 0.5, 0.75, 1].map((s) => (
        <T key={s} x={x(s)} y={H - B + 20} size={12} color={C.muted} anchor="middle">{Math.round(s * 100)}%</T>
      ))}
      <T x={(L + W - R) / 2} y={H - 10} size={13} w={600} color={C.text} anchor="middle">share of answers the checker marks “sure”</T>
      <text transform={`translate(16 ${(Tp + H - B) / 2}) rotate(-90)`} fontSize="13" fontWeight="600" fill={C.text} textAnchor="middle">accuracy of the “sure” answers</text>
      {series.map(([k, col]) => <path key={k} d={line(k)} fill="none" stroke={col} strokeWidth={k === "all" ? 2 : 3} strokeDasharray={k === "all" ? "6 5" : undefined} />)}
      {series.map(([k, col]) => pick[k].share != null && pick[k].acc != null && (
        <circle key={k} cx={x(pick[k].share!)} cy={y(pick[k].acc!)} r="7" fill={col} stroke="#fff" strokeWidth="2.5" />
      ))}
      {series.map(([k, col, lab], i) => (
        <g key={k} transform={`translate(${L + 16} ${H - B - 72 + i * 20})`}>
          <line x1="0" x2="26" y1="-4" y2="-4" stroke={col} strokeWidth={k === "all" ? 2 : 3} strokeDasharray={k === "all" ? "6 5" : undefined} />
          <T x={34} y={0} size={12.5} color={C.text}>{lab}</T>
        </g>
      ))}
      <T x={W - R - 8} y={y(0.6)} size={12} color={C.muted} anchor="end" italic>dots: the margin the website uses (±{Math.round(chosen * 100)} points)</T>
      <T x={W - R - 8} y={y(0.6) + 18} size={12} color={C.muted} anchor="end" italic>right end: every answer counted, no “not sure”</T>
    </svg>
  );
}
