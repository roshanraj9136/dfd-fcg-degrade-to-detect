import type { ReactNode } from "react";

export function Section({ id, kicker, title, lead, children }: {
  id: string; kicker: string; title: string; lead?: ReactNode; children: ReactNode;
}) {
  return (
    <section id={id} className="mx-auto max-w-6xl scroll-mt-20 px-5 py-16">
      <p className="mb-2 font-mono text-xs uppercase tracking-[0.25em] text-accent">{kicker}</p>
      <h2 className="text-3xl font-semibold tracking-tight text-white md:text-4xl">{title}</h2>
      {lead && <div className="mt-4 max-w-3xl text-lg leading-relaxed text-slate-300">{lead}</div>}
      <div className="mt-10">{children}</div>
    </section>
  );
}

export function Kpi({ label, value, sub, tone = "white" }: {
  label: string; value: ReactNode; sub?: ReactNode; tone?: "white" | "real" | "fake" | "accent" | "warn";
}) {
  const color = { white: "text-white", real: "text-real", fake: "text-fake", accent: "text-accent", warn: "text-warn" }[tone];
  return (
    <div className="card">
      <div className="text-xs uppercase tracking-widest muted">{label}</div>
      <div className={`kpi mt-2 ${color}`}>{value}</div>
      {sub && <div className="mt-2 text-sm muted">{sub}</div>}
    </div>
  );
}

export function Toggle<T extends string>({ value, options, onChange }: {
  value: T; options: { value: T; label: string }[]; onChange: (v: T) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((o) => (
        <button key={o.value} className={`btn ${o.value === value ? "btn-on" : ""}`} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Loading({ what = "results" }: { what?: string }) {
  return <div className="card animate-pulse muted">Loading {what}…</div>;
}

export function Verdict({ p }: { p: number }) {
  const fake = p >= 0.5;
  return (
    <span className={`rounded-md px-2 py-0.5 font-mono text-xs ${fake ? "bg-fake/15 text-fake" : "bg-real/15 text-real"}`}>
      {fake ? "FAKE" : "REAL"} {(100 * p).toFixed(1)}%
    </span>
  );
}
