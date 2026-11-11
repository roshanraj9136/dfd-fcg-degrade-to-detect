// Data access: static snapshots shipped with the site (always available) and the live API on Render.

export const API = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export type Metrics = {
  n_real: number; n_fake: number; auroc: number | null; ece: number;
  mean_real: number; mean_fake: number;
  "fpr@0.5": number; "tpr@0.5": number; "bacc@0.5": number;
  "fpr@tau5": number; "tpr@tau5": number; "bacc@tau5": number;
};
export type MethodResult = {
  all: Record<string, Metrics>; lopo: Record<string, Metrics>; fsh: Metrics; tau5: number;
  rank: Record<string, number> | null; pooled_all: Metrics; per_method_clean: Record<string, Metrics>;
};
export type Conditions = Record<string, MethodResult>;
export type SummaryRow = Record<string, number | string>;
export type Summary = {
  table: SummaryRow[];
  n: { test_real: number; test_fake: number; fit_real: number; fit_fake: number; fsh: number };
  conditions: Record<string, string>;
  params: Record<string, number[]>;
  en_pairs: number;
  regions: string[];
};
export type CondScore = { p: number; p_null: number; p_null_unseen: number; nuisance: number[]; d: number[] };
export type VideoIndex = { vid: string; split: string; method: string; label: number; public: boolean; scores: Record<string, CondScore> };
export type VideoDetail = { landmarks: number[][]; regions: number[]; maps: Record<string, { s: number[]; t: number[] }> };

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
    const t = setTimeout(() => ctl.abort(), 60000);
    const r = await fetch(`${API}/api/health`, { signal: ctl.signal });
    clearTimeout(t);
    return r.ok ? r.json() : null;
  } catch {
    return null;
  }
}

export async function apiNullify(body: { s: number[]; t: number[]; a: number[]; d: number[]; q_bg: number[] }) {
  const r = await fetch(`${API}/api/nullify`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`API ${r.status}`);
  return r.json() as Promise<{ p_original: number; p_corrected: number; nuisance_logit: number[]; verdict_original: string;
    verdict_corrected: string; top_quality_terms: { term: string; evidence: number }[] }>;
}

export async function videoDetail(vid: string): Promise<VideoDetail> {
  const safe = vid.replace("/", "__");
  return getStatic<VideoDetail>(`maps/${safe}.json`);
}

export const METHOD_LABEL: Record<string, string> = {
  original: "DFD-FCG (released)",
  platt: "Platt recalibration",
  qshift: "Quality-regression shift (face crop)",
  en: "Evidence Nulling (ours)",
  "en-face": "Evidence Nulling, face-crop quality (ablation)",
  "en-proj": "Attribution projection only (ablation)",
  refit: "Quality-augmented head re-fit",
};
export const METHOD_COLOR: Record<string, string> = {
  original: "#f472b6", platt: "#a78bfa", qshift: "#fbbf24", en: "#34d399", "en-face": "#22d3ee",
  "en-proj": "#64748b", refit: "#94a3b8",
};
export const FAKE_METHODS: Record<string, string> = {
  DF: "Deepfakes", F2F: "Face2Face", FS: "FaceSwap", NT: "NeuralTextures", FSh: "FaceShifter",
};
export const pct = (x: number | null | undefined, d = 1) => (x == null ? "–" : `${(100 * x).toFixed(d)}%`);
export const f3 = (x: number | null | undefined) => (x == null ? "–" : x.toFixed(3));
