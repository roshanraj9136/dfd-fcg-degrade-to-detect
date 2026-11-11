// Data access: static snapshots shipped with the site (always available) and the live API on Render.

export const API = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export type Metrics = {
  n_real: number; n_fake: number; auroc: number | null; ece: number;
  mean_real: number; mean_fake: number;
  "fpr@0.5": number; "tpr@0.5": number; "bacc@0.5": number;
  "fpr@tau5": number; "tpr@tau5": number; "bacc@tau5": number;
};
export type MethodResult = {
  all: Record<string, Metrics>; lopo: Record<string, Metrics>; fsh: Metrics; dfdc?: Metrics; tau5: number;
  cv_all?: Record<string, Metrics>; cv_lopo?: Record<string, Metrics>;
  pooled_all: Metrics; per_method_clean: Record<string, Metrics>; lam?: number; lam_curve?: Record<string, number>;
};
export type Conditions = Record<string, MethodResult>;
export type SummaryRow = Record<string, number | string>;
export type Summary = {
  table: SummaryRow[];
  main: string; strict: string; d2d_lam: number; d2d_lam_curve: Record<string, number>; strict_lam: number; alpha: number;
  drift_targets: Record<string, number[]>;
  probes: Record<string, string>;
  n: { test_real: number; test_fake: number; fit_real: number; fit_fake: number; fsh: number; dfdc_real: number; dfdc_fake: number };
  conditions: Record<string, string>;
  params: Record<string, number[]>;
  regions: string[];
};
export type CondScore = { p: number; p_fix: number; p_fix_unseen: number; p_strict: number; g: number[]; d: number[]; r: Record<string, number[]> };
export type VideoIndex = { vid: string; split: string; method: string; label: number; public: boolean; scores: Record<string, CondScore> };
export type VideoDetail = { landmarks: number[][]; regions: number[]; maps: Record<string, { s: number[]; t: number[]; sp: number[]; tp: number[] }> };

const cache = new Map<string, Promise<unknown>>();

export function getStatic<T>(name: string): Promise<T> {
  if (!cache.has(name)) {
    cache.set(name, fetch(`/data/${name}`).then((r) => {
      if (!r.ok) throw new Error(`${name}: ${r.status}`);
      return r.json();
    }));
  }
  return cache.get(name) as Promise<T>;
}

export async function apiHealth(): Promise<{ status: string; database?: string; videos?: number } | null> {
  if (!API) return null;
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 70000);
    const r = await fetch(`${API}/api/health`, { signal: ctl.signal });
    clearTimeout(t);
    return r.ok ? r.json() : null;
  } catch {
    return null;
  }
}

export type Decision = {
  p_released: number; p_selfcal: number; spurious_evidence: number[]; verdict_released: string; verdict_selfcal: string;
  response?: number[]; p_strict?: number; verdict_strict?: string; reading?: string;
};

export async function apiDecide(body: { d: number[]; y_s: number[]; y_t: number[]; d_probe?: Record<string, number[]> }): Promise<Decision> {
  const r = await fetch(`${API}/api/decide`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`API ${r.status}`);
  return r.json();
}

export async function videoDetail(vid: string): Promise<VideoDetail> {
  return getStatic<VideoDetail>(`maps/${vid.replace("/", "__")}.json`);
}

export const METHOD_LABEL: Record<string, string> = {
  original: "DFD-FCG (released)",
  platt: "Platt recalibration",
  qshift: "Quality-regression shift",
  en: "Background-quality nulling (negative result)",
  refit: "Quality-augmented head re-fit",
  "d2d-pB2": "Degrade-to-Detect, blur probe",
  "d2d-pJ50": "Degrade-to-Detect, JPEG probe",
  "d2d-lr": "Degrade-to-Detect, learned",
  selfcal: "Self-calibrated DFD-FCG (ours)",
  "selfcal+d2d": "Self-calibrated + blur probe, strict (ours)",
};
export const METHOD_COLOR: Record<string, string> = {
  original: "#f472b6", platt: "#a78bfa", qshift: "#fbbf24", en: "#64748b", refit: "#94a3b8",
  "d2d-pB2": "#22d3ee", "d2d-pJ50": "#0ea5e9", "d2d-lr": "#a3e635", selfcal: "#34d399", "selfcal+d2d": "#10b981",
};
export const NEEDS: Record<string, string> = {
  original: "–", platt: "fakes + reals", qshift: "reals only", en: "reals only", refit: "fakes + reals",
  "d2d-pB2": "1 number (λ)", "d2d-pJ50": "1 number (λ)", "d2d-lr": "10 numbers; fakes + reals",
  selfcal: "linear read-out of the model's own features (degraded training videos; no real/fake boundary is learned)",
  "selfcal+d2d": "same + 1 extra forward pass",
};
export const OURS = new Set(["selfcal", "selfcal+d2d"]);
export const FAKE_METHODS: Record<string, string> = {
  DF: "Deepfakes", F2F: "Face2Face", FS: "FaceSwap", NT: "NeuralTextures", FSh: "FaceShifter",
};
export const pct = (x: number | null | undefined, d = 1) => (x == null || Number.isNaN(x) ? "–" : `${(100 * x).toFixed(d)}%`);
export const f3 = (x: number | null | undefined) => (x == null || Number.isNaN(x) ? "–" : x.toFixed(3));
export const sgn = (x: number, d = 2) => `${x >= 0 ? "+" : ""}${x.toFixed(d)}`;
