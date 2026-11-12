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

export function Gauge({ p, verdict }: { p: number; verdict: "REAL" | "FAKE" }) {
  const fake = verdict === "FAKE";
  const conf = fake ? p : 1 - p;
  const R = 54, C = 2 * Math.PI * R;
  const color = fake ? "#dc2626" : "#16a34a";
  return (
    <svg viewBox="0 0 140 140" className="h-40 w-40" role="img" aria-label={`${verdict}, ${Math.round(conf * 100)}% confidence`}>
      <circle cx="70" cy="70" r={R} fill="none" stroke="#e2e8f0" strokeWidth="14" />
      <circle cx="70" cy="70" r={R} fill="none" stroke={color} strokeWidth="14" strokeLinecap="round"
        strokeDasharray={`${C * conf} ${C}`} transform="rotate(-90 70 70)" />
      <text x="70" y="66" textAnchor="middle" fontSize="26" fontWeight="800" fill={color}>{verdict}</text>
      <text x="70" y="90" textAnchor="middle" fontSize="13" fill="#64748b">{Math.round(conf * 100)}% sure</text>
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
      <line x1="160" y1="14" x2="160" y2="38" stroke="#94a3b8" strokeDasharray="3 3" />
      <text x="12" y="50" fontSize="10" fill="#16a34a">real</text>
      <text x="160" y="50" fontSize="10" fill="#94a3b8" textAnchor="middle">0.5</text>
      <text x="308" y="50" fontSize="10" fill="#dc2626" textAnchor="end">fake</text>
      <circle cx={x} cy="26" r="9" fill={fake ? "#dc2626" : "#16a34a"} stroke="#fff" strokeWidth="3" opacity={muted ? 0.65 : 1} />
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
      <ellipse cx="80" cy="98" rx="62" ry="80" fill={c("skin")} stroke="#cbd5e1" strokeWidth="2" />
      <ellipse cx="55" cy="80" rx="18" ry="10" fill={c("eyes")} stroke="#fff" strokeWidth="2" />
      <ellipse cx="105" cy="80" rx="18" ry="10" fill={c("eyes")} stroke="#fff" strokeWidth="2" />
      <path d="M80 85v30l-12 6h24z" fill={c("nose")} stroke="#fff" strokeWidth="2" strokeLinejoin="round" />
      <ellipse cx="80" cy="145" rx="26" ry="11" fill={c("lips")} stroke="#fff" strokeWidth="2" />
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
