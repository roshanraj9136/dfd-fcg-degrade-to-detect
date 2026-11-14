// Simple SVG illustrations for the checker page (no external images). Colours use CSS variables so dark mode works.

type P = { className?: string };

/** The upload illustration: a video frame with a face, a moving scan line and REAL / FAKE chips. */
export function ScanArt({ className = "h-40 w-auto", active = false }: P & { active?: boolean }) {
  return (
    <svg viewBox="0 0 240 170" className={className} aria-hidden>
      <defs>
        <linearGradient id="sa-bg" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" style={{ stopColor: "var(--color-brand-soft)" }} />
          <stop offset="1" style={{ stopColor: "var(--color-white)" }} />
        </linearGradient>
        <linearGradient id="sa-line" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="#6366f1" stopOpacity="0" />
          <stop offset=".5" stopColor="#6366f1" stopOpacity=".55" />
          <stop offset="1" stopColor="#6366f1" stopOpacity="0" />
        </linearGradient>
        <clipPath id="sa-clip"><rect x="40" y="14" width="160" height="124" rx="16" /></clipPath>
      </defs>
      <rect x="40" y="14" width="160" height="124" rx="16" fill="url(#sa-bg)" style={{ stroke: "var(--color-slate-200)" }} strokeWidth="2" />
      <g clipPath="url(#sa-clip)">
        <ellipse cx="120" cy="80" rx="34" ry="42" fill="#fde7d6" stroke="#f1c7a6" strokeWidth="2" />
        <path d="M86 68c4-30 64-30 68 0" fill="#6b4f3a" opacity=".9" />
        <ellipse cx="107" cy="79" rx="4.5" ry="3.5" fill="#334155" />
        <ellipse cx="133" cy="79" rx="4.5" ry="3.5" fill="#334155" />
        <path d="M120 84v10l-4 3" fill="none" stroke="#e0a982" strokeWidth="2.4" strokeLinecap="round" />
        <path d="M109 104c7 5 15 5 22 0" fill="none" stroke="#e11d48" strokeWidth="3" strokeLinecap="round" />
        <rect x="40" y="0" width="160" height="34" fill="url(#sa-line)" className={active ? "scan-fast" : "scan"} />
      </g>
      <g fill="none" stroke="#4f46e5" strokeWidth="3.5" strokeLinecap="round">
        <path d="M80 40v-10h12M160 40v-10h-12M80 120v10h12M160 120v10h-12" />
      </g>
      <g transform="translate(46 144)">
        <rect width="62" height="24" rx="12" style={{ fill: "var(--color-real-soft)" }} stroke="#86efac" />
        <text x="31" y="16.5" textAnchor="middle" fontSize="12" fontWeight="800" style={{ fill: "var(--color-real)" }}>REAL</text>
      </g>
      <g transform="translate(132 144)">
        <rect width="62" height="24" rx="12" style={{ fill: "var(--color-fake-soft)" }} stroke="#fca5a5" />
        <text x="31" y="16.5" textAnchor="middle" fontSize="12" fontWeight="800" style={{ fill: "var(--color-fake)" }}>FAKE</text>
      </g>
    </svg>
  );
}

/** Big verdict emblem: a shield with a tick (REAL) or a warning mark (FAKE). */
export function VerdictEmblem({ fake, className = "h-16 w-16" }: P & { fake: boolean }) {
  const c = fake ? "#dc2626" : "#16a34a";
  return (
    <svg viewBox="0 0 64 64" className={className} aria-hidden>
      <path d="M32 4l22 8v16c0 14-9.5 25.5-22 32C19.5 53.5 10 42 10 28V12z" fill={c} />
      <path d="M32 9l17 6.2V28c0 11-7.3 20.4-17 26-9.7-5.6-17-15-17-26V15.2z" fill="#fff" opacity=".18" />
      {fake ? (
        <g fill="#fff"><rect x="29" y="18" width="6" height="20" rx="3" /><circle cx="32" cy="45" r="3.6" /></g>
      ) : (
        <path d="M21 32l8 8 15-17" fill="none" stroke="#fff" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" />
      )}
    </svg>
  );
}

/** A score from 0 (real) to 1 (fake): a rounded track filled towards the side it points to. */
export function DetectorBar({ p, strong = false }: { p: number; strong?: boolean }) {
  const v = Math.min(1, Math.max(0, p));
  const fake = v >= 0.5;
  const col = fake ? "#dc2626" : "#16a34a";
  const w = 200;
  const x = 4 + (w - 8) * v;
  return (
    <svg viewBox={`0 0 ${w} 16`} className="h-4 w-full" role="img" aria-label={`${Math.round(v * 100)}% fake`} preserveAspectRatio="none">
      <rect x="0" y="4" width={w} height="8" rx="4" style={{ fill: "var(--color-slate-100)" }} />
      <rect x={fake ? w / 2 : x} y="4" width={Math.abs(x - w / 2)} height="8" rx="4" fill={col} opacity={strong ? 1 : 0.75} />
      <line x1={w / 2} x2={w / 2} y1="1" y2="15" style={{ stroke: "var(--color-slate-400)" }} strokeWidth="1.2" />
    </svg>
  );
}

/** A ring filled to `v` (0..1), with the number in the middle. */
export function StatRing({ v, color, className = "h-24 w-24" }: P & { v: number; color: string }) {
  const R = 40, C = 2 * Math.PI * R;
  return (
    <svg viewBox="0 0 100 100" className={className} aria-hidden>
      <circle cx="50" cy="50" r={R} fill="none" style={{ stroke: "var(--color-slate-100)" }} strokeWidth="10" />
      <circle cx="50" cy="50" r={R} fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
        strokeDasharray={`${C * Math.min(1, Math.max(0, v))} ${C}`} transform="rotate(-90 50 50)" />
      <text x="50" y="57" textAnchor="middle" fontSize="21" fontWeight="800" style={{ fill: "var(--color-slate-900)" }}>
        {Math.round(v * 100)}%
      </text>
    </svg>
  );
}

/** The three steps of a check, as small pictures. */
export function FlowArt({ step, className = "h-16 w-16" }: P & { step: "face" | "detectors" | "answer" }) {
  return (
    <svg viewBox="0 0 64 64" className={className} aria-hidden>
      <rect width="64" height="64" rx="18" style={{ fill: "var(--color-brand-soft)" }} />
      {step === "face" && (
        <g>
          <ellipse cx="32" cy="33" rx="11" ry="13.5" fill="#fde7d6" stroke="#e0a982" strokeWidth="1.6" />
          <path d="M21 29c1.5-10 20.5-10 22 0" fill="#6b4f3a" />
          <circle cx="28" cy="33" r="1.4" fill="#334155" /><circle cx="36" cy="33" r="1.4" fill="#334155" />
          <path d="M28.5 39c2 1.4 5 1.4 7 0" fill="none" stroke="#e11d48" strokeWidth="1.6" strokeLinecap="round" />
          <path d="M15 22v-6h6M49 22v-6h-6M15 42v6h6M49 42v6h-6" fill="none" stroke="#4f46e5" strokeWidth="3" strokeLinecap="round" />
        </g>
      )}
      {step === "detectors" && (
        <g>
          <rect x="12" y="14" width="40" height="9" rx="4.5" fill="#4f46e5" />
          <rect x="12" y="27.5" width="40" height="9" rx="4.5" fill="#818cf8" />
          <rect x="12" y="41" width="40" height="9" rx="4.5" fill="#a5b4fc" />
          <circle cx="44" cy="18.5" r="2.4" fill="#fff" /><circle cx="36" cy="32" r="2.4" fill="#fff" /><circle cx="40" cy="45.5" r="2.4" fill="#fff" />
        </g>
      )}
      {step === "answer" && (
        <g>
          <path d="M32 11l16 6v12c0 10-7 18.5-16 23-9-4.5-16-13-16-23V17z" fill="#16a34a" />
          <path d="M24.5 31.5l5.5 5.5 10-11" fill="none" stroke="#fff" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
        </g>
      )}
    </svg>
  );
}

/** Small line icons for trust points. */
export function MiniIcon({ kind, className = "h-4 w-4" }: P & { kind: "target" | "layers" | "lock" | "bolt" }) {
  const s = { fill: "none", stroke: "currentColor", strokeWidth: 2.2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  return (
    <svg viewBox="0 0 24 24" className={className} {...s} aria-hidden>
      {kind === "target" && <><circle cx="12" cy="12" r="8.5" /><circle cx="12" cy="12" r="4.5" /><circle cx="12" cy="12" r="1" fill="currentColor" /></>}
      {kind === "layers" && <><path d="M12 3l9 5-9 5-9-5z" /><path d="M3 13l9 5 9-5" /></>}
      {kind === "lock" && <><rect x="5" y="10.5" width="14" height="10" rx="2.5" /><path d="M8.5 10.5V8a3.5 3.5 0 0 1 7 0v2.5" /></>}
      {kind === "bolt" && <path d="M13 3L5 14h6l-1 7 8-11h-6z" />}
    </svg>
  );
}

/** Dashed arrow between the steps (desktop). */
export function FlowArrow({ className = "h-6 w-12" }: P) {
  return (
    <svg viewBox="0 0 48 24" className={className} aria-hidden>
      <path d="M2 12h38" style={{ stroke: "var(--color-slate-300)" }} strokeWidth="2.5" strokeDasharray="4 5" strokeLinecap="round" />
      <path d="M36 6l8 6-8 6" fill="none" style={{ stroke: "var(--color-slate-300)" }} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
