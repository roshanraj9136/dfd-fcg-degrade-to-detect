// Hand-drawn SVG illustrations and icons (no external images).

type P = { className?: string };

export function Logo({ className = "h-9 w-9" }: P) {
  return (
    <svg viewBox="0 0 64 64" className={className} aria-hidden>
      <rect width="64" height="64" rx="16" fill="#4f46e5" />
      <circle cx="32" cy="28" r="12" fill="none" stroke="#fff" strokeWidth="4" />
      <path d="M18 50c3-7 8-10 14-10s11 3 14 10" fill="none" stroke="#fff" strokeWidth="4" strokeLinecap="round" />
      <circle cx="48" cy="16" r="7" fill="#22c55e" stroke="#fff" strokeWidth="3" />
    </svg>
  );
}

export function HeroArt({ className = "w-full" }: P) {
  return (
    <svg viewBox="0 0 420 300" className={className} aria-hidden>
      <defs>
        <linearGradient id="hg" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#eef2ff" />
          <stop offset="1" stopColor="#e0e7ff" />
        </linearGradient>
      </defs>
      {/* video frame */}
      <rect x="20" y="30" width="250" height="190" rx="18" fill="url(#hg)" stroke="#c7d2fe" strokeWidth="2" />
      <rect x="20" y="30" width="250" height="28" rx="14" fill="#c7d2fe" />
      <circle cx="40" cy="44" r="5" fill="#fff" /><circle cx="56" cy="44" r="5" fill="#fff" /><circle cx="72" cy="44" r="5" fill="#fff" />
      {/* face */}
      <ellipse cx="145" cy="138" rx="52" ry="62" fill="#fde7d6" stroke="#f3c9ab" strokeWidth="2" />
      <path d="M95 118c6-38 94-38 100 0" fill="#7c5a43" opacity=".85" />
      <ellipse cx="125" cy="135" rx="7" ry="5" fill="#334155" /><ellipse cx="165" cy="135" rx="7" ry="5" fill="#334155" />
      <path d="M145 140v18l-6 4" fill="none" stroke="#e0a982" strokeWidth="3" strokeLinecap="round" />
      <path d="M128 172c10 8 24 8 34 0" fill="none" stroke="#e11d48" strokeWidth="4" strokeLinecap="round" />
      {/* scan grid */}
      <g stroke="#6366f1" strokeWidth="1" opacity=".35">
        {Array.from({ length: 7 }, (_, i) => <line key={`v${i}`} x1={95 + i * 17} y1="76" x2={95 + i * 17} y2="200" />)}
        {Array.from({ length: 8 }, (_, i) => <line key={`h${i}`} x1="95" y1={76 + i * 17.7} x2="197" y2={76 + i * 17.7} />)}
      </g>
      <rect x="95" y="76" width="102" height="124" fill="none" stroke="#4f46e5" strokeWidth="3" rx="8" strokeDasharray="10 6" />
      {/* magnifier */}
      <circle cx="300" cy="150" r="46" fill="#fff" stroke="#4f46e5" strokeWidth="8" />
      <line x1="333" y1="184" x2="372" y2="226" stroke="#4f46e5" strokeWidth="14" strokeLinecap="round" />
      <path d="M280 152l14 14 26-30" fill="none" stroke="#16a34a" strokeWidth="9" strokeLinecap="round" strokeLinejoin="round" />
      {/* badges */}
      <g transform="translate(250 245)">
        <rect width="70" height="32" rx="16" fill="#f0fdf4" stroke="#86efac" />
        <text x="35" y="21" textAnchor="middle" fontSize="14" fontWeight="700" fill="#16a34a">REAL</text>
      </g>
      <g transform="translate(330 245)">
        <rect width="70" height="32" rx="16" fill="#fef2f2" stroke="#fca5a5" />
        <text x="35" y="21" textAnchor="middle" fontSize="14" fontWeight="700" fill="#dc2626">FAKE</text>
      </g>
    </svg>
  );
}

export function UploadIcon({ className = "h-12 w-12" }: P) {
  return (
    <svg viewBox="0 0 48 48" className={className} aria-hidden fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
      <rect x="6" y="10" width="36" height="28" rx="6" />
      <path d="M20 19l10 5-10 5z" fill="currentColor" stroke="none" />
      <path d="M24 2v8M20 6l4-4 4 4" />
    </svg>
  );
}

const stroke = { fill: "none", stroke: "currentColor", strokeWidth: 2.4, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };

export function StepIcon({ step, className = "h-6 w-6" }: { step: string; className?: string }) {
  switch (step) {
    case "upload":
      return <svg viewBox="0 0 24 24" className={className} {...stroke}><path d="M12 16V4M7 9l5-5 5 5" /><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" /></svg>;
    case "face":
      return <svg viewBox="0 0 24 24" className={className} {...stroke}><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" /><circle cx="9.5" cy="10.5" r=".6" fill="currentColor" /><circle cx="14.5" cy="10.5" r=".6" fill="currentColor" /><path d="M9.5 15c1.4 1 3.6 1 5 0" /></svg>;
    case "model":
      return <svg viewBox="0 0 24 24" className={className} {...stroke}><rect x="6" y="6" width="12" height="12" rx="2" /><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" /><path d="M10 10h4v4h-4z" /></svg>;
    case "quality":
      return <svg viewBox="0 0 24 24" className={className} {...stroke}><path d="M4 18h16" /><path d="M6 18V9M10 18V5M14 18v-7M18 18v-4" /></svg>;
    case "blur":
      return <svg viewBox="0 0 24 24" className={className} {...stroke}><circle cx="12" cy="12" r="8" strokeDasharray="2 3" /><circle cx="12" cy="12" r="3.5" /></svg>;
    case "gpu":
      return <svg viewBox="0 0 24 24" className={className} {...stroke}><rect x="3" y="7" width="18" height="10" rx="2" /><circle cx="9" cy="12" r="2.2" /><circle cx="15" cy="12" r="2.2" /><path d="M6 17v3M10 17v3M14 17v3M18 17v3" /></svg>;
    default:
      return <svg viewBox="0 0 24 24" className={className} {...stroke}><path d="M5 12l5 5 9-10" /></svg>;
  }
}

export function Gauge({ p, verdict, unsure = false }: { p: number; verdict: "REAL" | "FAKE"; unsure?: boolean }) {
  const fake = verdict === "FAKE";
  const conf = fake ? p : 1 - p;
  const R = 54, C = 2 * Math.PI * R;
  const color = unsure ? "#d97706" : fake ? "#dc2626" : "#16a34a";
  if (unsure) {
    // a near-50/50 score must not look like a confident verdict: amber ring at the fake score, "NOT SURE" in the middle
    return (
      <svg viewBox="0 0 140 140" className="h-40 w-40" role="img" aria-label={`Not sure, ${Math.round(p * 100)}% fake, leans ${verdict.toLowerCase()}`}>
        <circle cx="70" cy="70" r={R} fill="none" style={{ stroke: "var(--color-slate-200)" }} strokeWidth="14" />
        <circle cx="70" cy="70" r={R} fill="none" stroke={color} strokeWidth="14" strokeLinecap="round"
          strokeDasharray={`${C * p} ${C}`} transform="rotate(-90 70 70)" />
        <text x="70" y="64" textAnchor="middle" fontSize="15.5" fontWeight="800" fill={color}>NOT SURE</text>
        <text x="70" y="86" textAnchor="middle" fontSize="12.5" style={{ fill: "var(--color-slate-500)" }}>{Math.round(p * 100)}% fake</text>
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 140 140" className="h-40 w-40" role="img" aria-label={`${verdict}, ${Math.round(conf * 100)}% confidence`}>
      <circle cx="70" cy="70" r={R} fill="none" style={{ stroke: "var(--color-slate-200)" }} strokeWidth="14" />
      <circle cx="70" cy="70" r={R} fill="none" stroke={color} strokeWidth="14" strokeLinecap="round"
        strokeDasharray={`${C * conf} ${C}`} transform="rotate(-90 70 70)" />
      <text x="70" y="66" textAnchor="middle" fontSize="26" fontWeight="800" fill={color}>{verdict}</text>
      <text x="70" y="90" textAnchor="middle" fontSize="13" style={{ fill: "var(--color-slate-500)" }}>{Math.round(conf * 100)}% sure</text>
    </svg>
  );
}

/** A bar from 0 (real) to 1 (fake) with the 0.5 line, and a marker. */
export function ScoreBar({ p, label, muted = false }: { p: number; label: string; muted?: boolean }) {
  const x = 10 + 300 * Math.min(1, Math.max(0, p));
  const fake = p >= 0.5;
  return (
    <svg viewBox="0 0 320 52" className="w-full" role="img" aria-label={`${label}: ${Math.round(p * 100)}% fake`}>
      <defs>
        <linearGradient id={`sb-${label.replace(/\W/g, "")}`} x1="0" x2="1">
          <stop offset="0" stopColor="#bbf7d0" /><stop offset=".5" stopColor="#f1f5f9" /><stop offset="1" stopColor="#fecaca" />
        </linearGradient>
      </defs>
      <rect x="10" y="20" width="300" height="12" rx="6" fill={`url(#sb-${label.replace(/\W/g, "")})`} opacity={muted ? 0.6 : 1} />
      <line x1="160" y1="14" x2="160" y2="38" style={{ stroke: "var(--color-slate-400)" }} strokeDasharray="3 3" />
      <text x="12" y="50" fontSize="10" fill="#16a34a">real</text>
      <text x="160" y="50" fontSize="10" style={{ fill: "var(--color-slate-500)" }} textAnchor="middle">0.5</text>
      <text x="308" y="50" fontSize="10" fill="#dc2626" textAnchor="end">fake</text>
      <circle cx={x} cy="26" r="9" fill={fake ? "#dc2626" : "#16a34a"} style={{ stroke: "var(--color-white)" }} strokeWidth="3" opacity={muted ? 0.65 : 1} />
      <text x={x} y="10" fontSize="11" fontWeight="700" textAnchor="middle" fill={fake ? "#dc2626" : "#16a34a"} opacity={muted ? 0.75 : 1}>
        {Math.round(p * 100)}%
      </text>
    </svg>
  );
}

export function FaceRegions({ regions }: { regions: Record<string, number> }) {
  const max = Math.max(1e-6, ...Object.values(regions).map(Math.abs));
  const c = (k: string) => {
    const v = regions[k] ?? 0;
    const a = 0.15 + 0.75 * Math.min(1, Math.abs(v) / max);
    return v >= 0 ? `rgba(220,38,38,${a})` : `rgba(22,163,74,${a})`;
  };
  return (
    <svg viewBox="0 0 160 190" className="h-44 w-auto" aria-hidden>
      <ellipse cx="80" cy="98" rx="62" ry="80" fill={c("skin")} style={{ stroke: "var(--color-slate-300)" }} strokeWidth="2" />
      <ellipse cx="55" cy="80" rx="18" ry="10" fill={c("eyes")} style={{ stroke: "var(--color-white)" }} strokeWidth="2" />
      <ellipse cx="105" cy="80" rx="18" ry="10" fill={c("eyes")} style={{ stroke: "var(--color-white)" }} strokeWidth="2" />
      <path d="M80 85v30l-12 6h24z" fill={c("nose")} style={{ stroke: "var(--color-white)" }} strokeWidth="2" strokeLinejoin="round" />
      <ellipse cx="80" cy="145" rx="26" ry="11" fill={c("lips")} style={{ stroke: "var(--color-white)" }} strokeWidth="2" />
    </svg>
  );
}

export function Spinner({ className = "h-5 w-5" }: P) {
  return (
    <svg viewBox="0 0 24 24" className={`${className} animate-spin`} aria-hidden>
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeOpacity=".25" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

type Clip = { start: number; end: number; p: number; faces: number[] };

/** Fake score of every 3-second clip along the video (dot = middle of the clip), one line per person. */
export function Timeline({ clips, seconds, shown = 1 }: { clips: Clip[]; seconds: number; shown?: number }) {
  const W = 560, H = 200, L = 46, R = 18, T = 14, B = 34;
  const tmax = Math.max(1, seconds, ...clips.map((c) => c.end));
  const x = (t: number) => L + ((W - L - R) * t) / tmax;
  const y = (p: number) => T + (H - T - B) * (1 - p);
  const mid = (c: Clip) => (c.start + c.end) / 2;
  const nf = clips[0]?.faces.length ?? 1;
  const line = (ps: number[]) => ps.map((p, i) => `${i ? "L" : "M"}${x(mid(clips[i])).toFixed(1)} ${y(p).toFixed(1)}`).join(" ");
  const ticks = Array.from({ length: Math.floor(tmax) + 1 }, (_, i) => i).filter((t) => tmax <= 12 || t % 2 === 0);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto w-full min-w-[460px] max-w-[700px]" role="img" aria-label="Fake score of each clip along the video">
      <rect x={L} y={y(1)} width={W - L - R} height={y(0.5) - y(1)} style={{ fill: "var(--color-fake-soft)" }} rx="6" />
      <rect x={L} y={y(0.5)} width={W - L - R} height={y(0) - y(0.5)} style={{ fill: "var(--color-real-soft)" }} rx="6" />
      <line x1={L} x2={W - R} y1={y(0.5)} y2={y(0.5)} style={{ stroke: "var(--color-slate-400)" }} strokeDasharray="5 4" />
      <text x={L - 8} y={y(0.85)} fontSize="12" fill="#dc2626" textAnchor="end" fontWeight="600">fake</text>
      <text x={L - 8} y={y(0.5) + 4} fontSize="11" style={{ fill: "var(--color-slate-500)" }} textAnchor="end">50%</text>
      <text x={L - 8} y={y(0.15)} fontSize="12" fill="#16a34a" textAnchor="end" fontWeight="600">real</text>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={y(0)} y2={y(0) + 5} style={{ stroke: "var(--color-slate-300)" }} />
          <text x={x(t)} y={H - 10} fontSize="11" style={{ fill: "var(--color-slate-500)" }} textAnchor="middle">{t} s</text>
        </g>
      ))}
      {Array.from({ length: nf }, (_, f) => f).filter((f) => f !== shown - 1).map((f) => (
        <path key={f} d={line(clips.map((c) => c.faces[f]))} fill="none" style={{ stroke: "var(--color-slate-400)" }} strokeWidth="2.5" strokeDasharray="6 5" />
      ))}
      <path d={line(clips.map((c) => c.p))} fill="none" stroke="#4f46e5" strokeWidth="3" strokeLinejoin="round" />
      {clips.map((c, i) => (
        <circle key={i} cx={x(mid(c))} cy={y(c.p)} r="7" fill={c.p >= 0.5 ? "#dc2626" : "#16a34a"} style={{ stroke: "var(--color-white)" }} strokeWidth="2.5">
          <title>{`${c.start.toFixed(1)}–${c.end.toFixed(1)} s: ${Math.round(c.p * 100)}% fake`}</title>
        </circle>
      ))}
    </svg>
  );
}

/** Every person the checker looked at, with their score; the one that decided the answer is highlighted. */
export function People({ ps, shown }: { ps: number[]; shown: number }) {
  const w = 120;
  return (
    <svg viewBox={`0 0 ${w * ps.length} 150`} className="h-36 w-auto" role="img" aria-label={`${ps.length} people checked`}>
      {ps.map((p, i) => {
        const on = i === shown - 1;
        const col = p >= 0.5 ? "#dc2626" : "#16a34a";
        const cx = w * i + w / 2;
        return (
          <g key={i} opacity={on ? 1 : 0.55}>
            <circle cx={cx} cy="50" r="40" style={{ fill: on ? "var(--color-brand-soft)" : "var(--color-slate-50)", stroke: on ? "var(--color-brand)" : "var(--color-slate-300)" }} strokeWidth={on ? 4 : 2} />
            <circle cx={cx} cy="40" r="14" fill="#fde7d6" stroke="#e0a982" strokeWidth="2" />
            <path d={`M${cx - 24} 78c4-14 13-20 24-20s20 6 24 20`} fill="#fde7d6" stroke="#e0a982" strokeWidth="2" />
            <text x={cx} y="114" textAnchor="middle" fontSize="13" style={{ fill: "var(--color-slate-700)" }} fontWeight="600">Person {i + 1}</text>
            <text x={cx} y="134" textAnchor="middle" fontSize="13" fill={col} fontWeight="700">{Math.round(p * 100)}% fake</text>
          </g>
        );
      })}
    </svg>
  );
}

/** Small badge icon: a check in a shield (sure) or a question mark (not sure). */
export function SureIcon({ sure, className = "h-5 w-5" }: { sure: boolean; className?: string }) {
  return sure ? (
    <svg viewBox="0 0 24 24" className={className} {...stroke}><path d="M12 3l7 3v5c0 4.5-3 8.2-7 10-4-1.8-7-5.5-7-10V6z" /><path d="M8.5 12l2.5 2.5 4.5-5" /></svg>
  ) : (
    <svg viewBox="0 0 24 24" className={className} {...stroke}><circle cx="12" cy="12" r="9" /><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 .9-1 1.6v.6" /><circle cx="12" cy="17" r=".6" fill="currentColor" /></svg>
  );
}

/** Small pictures for the three ways to check: the video as it is, with strong blur, with heavy compression. */
export function OptionIcon({ kind, className = "h-11 w-11" }: { kind: string; className?: string }) {
  const face = (
    <g>
      <circle cx="24" cy="21" r="8" fill="#fde7d6" stroke="#e0a982" strokeWidth="1.5" />
      <circle cx="21" cy="20" r="1.2" fill="#334155" /><circle cx="27" cy="20" r="1.2" fill="#334155" />
      <path d="M21 24.5c1.8 1.4 4.2 1.4 6 0" fill="none" stroke="#e11d48" strokeWidth="1.4" strokeLinecap="round" />
      <path d="M12 38c2-6 6.5-8.5 12-8.5s10 2.5 12 8.5" fill="#c7d2fe" />
    </g>
  );
  const bg = kind === "blur" ? "#e0f2fe" : kind === "compress" ? "#fef3c7" : "#dcfce7";
  return (
    <svg viewBox="0 0 48 48" className={className} aria-hidden>
      <defs>
        <filter id="opt-blur" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="1.8" /></filter>
        <clipPath id="opt-clip"><rect x="6" y="6" width="36" height="36" rx="8" /></clipPath>
      </defs>
      <rect x="2" y="2" width="44" height="44" rx="12" fill={bg} />
      <rect x="6" y="6" width="36" height="36" rx="8" fill="#ffffff" />
      {kind === "blur" && <g filter="url(#opt-blur)" clipPath="url(#opt-clip)">{face}</g>}
      {kind === "compress" && (
        <g clipPath="url(#opt-clip)">
          {face}
          {/* big square blocks, like a video squeezed into a tiny file */}
          {[[12, 12, "#fde7d6"], [24, 12, "#fde7d6"], [12, 24, "#c7d2fe"], [24, 24, "#fde7d6"], [12, 30, "#c7d2fe"], [30, 30, "#c7d2fe"]].map(([x, y, c], i) => (
            <rect key={i} x={x as number} y={y as number} width="12" height="12" fill={c as string} opacity=".55" />
          ))}
          <g stroke="#ffffff" strokeWidth="1.2" opacity=".9">
            {[12, 18, 24, 30, 36].map((v) => <line key={`v${v}`} x1={v} x2={v} y1="6" y2="42" />)}
            {[12, 18, 24, 30, 36].map((v) => <line key={`h${v}`} y1={v} y2={v} x1="6" x2="42" />)}
          </g>
        </g>
      )}
      {kind !== "blur" && kind !== "compress" && face}
    </svg>
  );
}
