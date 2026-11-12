import { Client, handle_file } from "@gradio/client";
import type { SpaceStatus } from "@gradio/client";

// The checker runs as a Hugging Face Space on ZeroGPU (a free GPU is attached for each check).
export const SPACE = ((import.meta.env.VITE_CHECKER_SPACE as string | undefined) || "roshan9136/dfd-fcg-checker").trim();
export const SPACE_URL = `https://huggingface.co/spaces/${SPACE}`;

export type Result = {
  verdict: "REAL" | "FAKE";
  p: number; p_original: number; p_strict: number; verdict_original: "REAL" | "FAKE";
  quality_shift: number; spurious_logit: number[]; blur_response: number[]; heads: number[];
  regions: Record<"eyes" | "nose" | "lips" | "skin", number>;
  parts: Record<string, number>;
  face_png: string; heat_png: string;
  sure?: boolean; agreement?: number; margin?: number;
  faces?: { p: number; p_original: number; p_strict: number }[];
  timeline?: { start: number; end: number; p: number; faces: number[] }[];
  meta: {
    fps: number; frames_used: number; clips: number; width: number; height: number; degrade: string; seconds: number;
    device?: string; faces_checked?: number; face_shown?: number; mirrored?: boolean; seconds_covered?: number;
  };
};
export type Health = { status: "ready" | "waking" | "error"; detail?: string; analysed_total?: number | null; error?: string };
export type Progress = { phase: "upload" | "queue" | "run" | "step"; position?: number; step?: string; frac?: number };

let client: Promise<Client> | null = null;
let lastStatus: SpaceStatus | null = null;
const listeners = new Set<(s: SpaceStatus) => void>();

export function onSpaceStatus(fn: (s: SpaceStatus) => void) {
  listeners.add(fn);
  if (lastStatus) fn(lastStatus);
  return () => { listeners.delete(fn); };
}

function connect(): Promise<Client> {
  if (!client) {
    client = Client.connect(SPACE, {
      events: ["data", "status"],
      status_callback: (s) => { lastStatus = s; listeners.forEach((f) => f(s)); },
    }).catch((e) => { client = null; throw e; });
  }
  return client;
}

export async function health(): Promise<Health> {
  try {
    const c = await connect();
    const r = await c.predict("/health", []);
    const h = (r.data as unknown[])[0] as { analysed_total?: number | null };
    return { status: "ready", analysed_total: h?.analysed_total ?? null };
  } catch (e) {
    const s = lastStatus;
    if (s && s.status === "error") return { status: "error", detail: s.detail, error: s.message };
    return { status: "error", error: friendly(e) };
  }
}

/** Upload one video and wait for the verdict. */
export async function check(file: Blob, name: string, degrade: string, onProgress: (p: Progress) => void): Promise<Result> {
  onProgress({ phase: "upload" });
  const c = await connect();
  const f = file instanceof File ? file : new File([file], name, { type: file.type || "video/mp4" });
  const job = c.submit("/analyze", [handle_file(f), degrade]);
  for await (const msg of job) {
    if (msg.type === "status") {
      if (msg.stage === "error") throw new Error(friendly(msg.message));
      if (msg.stage === "pending") {
        const pd = msg.progress_data?.[0];
        // the checker reports its steps as "<step id>: <text>" (face, gpu, cpu, quality)
        if (pd?.desc) onProgress({ phase: "step", step: pd.desc.split(":")[0].trim(), frac: pd.progress ?? undefined });
        else if (msg.position && msg.position > 0) onProgress({ phase: "queue", position: msg.position });
        else if (msg.position === 0) onProgress({ phase: "run" });
      }
    } else if (msg.type === "data") {
      return (msg.data as unknown[])[0] as Result;
    }
  }
  throw new Error("The check ended without a result. Please try again.");
}

function friendly(m: unknown): string {
  let s = typeof m === "string" ? m : m instanceof Error ? m.message : Array.isArray(m) ? m.map(String).join(" ") : String(m ?? "");
  s = s.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
  if (/quota/i.test(s)) {
    return "The free daily GPU time for your internet connection is used up (Hugging Face gives every visitor a few " +
      "minutes per day). Please try again later, or from another network.";
  }
  if (/no gpu was available/i.test(s)) return "All free GPUs are busy right now. Please try again in a minute.";
  if (!s || /could not resolve|failed to fetch|networkerror/i.test(s)) {
    return "Could not reach the checker. It may be waking up, which takes a minute or two. Please try again shortly.";
  }
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export const SAMPLES = [
  { id: "real", label: "Sample: real video", url: "https://raw.githubusercontent.com/aiiu-lab/DFD-FCG/main/resources/videos/000.mp4", name: "sample_real_000.mp4" },
  { id: "fake", label: "Sample: deepfake", url: "https://raw.githubusercontent.com/aiiu-lab/DFD-FCG/main/resources/videos/000_003.mp4", name: "sample_fake_000_003.mp4" },
];
