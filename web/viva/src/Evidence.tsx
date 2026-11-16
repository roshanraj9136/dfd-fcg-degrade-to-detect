import type { ReactNode } from "react";
import evidence from "./evidence.json";

const repo = "https://github.com/roshanraj9136/dfd-fcg-degrade-to-detect/blob/main/";
export function Source({ path, children }: { path: string; children?: ReactNode }) {
  return <a className="text-brand underline decoration-indigo-200 underline-offset-4 hover:decoration-brand" href={repo + path}>{children ?? path}</a>;
}

const labels: Record<string, string> = {
  dfd: "DFD-FCG · corrected", pe_ft: "Fine-tuned GenD-PE · raw", pe_ft_sc: "Fine-tuned GenD-PE · corrected",
  pe: "Released GenD-PE · raw", pe_sc: "Released GenD-PE · corrected",
};
const networks = [
  { title: "DFD-FCG", origin: "Authors’ released weights", input: "8 clips × 10 frames per face", view: "Spatial + temporal features", members: ["dfd"],
    detail: "Frozen CLIP ViT-L/14 and the authors’ trained side network produce three head logits. Our fitted ridge read-out predicts a shift for each head. Subtract it, apply sigmoid to each head, then average the probabilities over heads and clips." },
  { title: "Fine-tuned GenD-PE", origin: "Our adapted checkpoint", input: "Up to 24 frames per face", view: "Single-frame features", members: ["pe_ft", "pe_ft_sc"],
    detail: "PE means Perception Encoder, a foundation image model. We fine-tuned its LayerNorm parameters and GenD’s classifier head using the GenD recipe; the other encoder weights stay frozen. One run supplies both raw and self-calibrated scores. The class-token read-out is fitted separately for this checkpoint." },
  { title: "Released GenD-PE", origin: "Authors’ released weights", input: "Up to 24 frames per face", view: "Single-frame features", members: ["pe", "pe_sc"],
    detail: "This network keeps the released GenD-PE weights. The Perception Encoder summarizes each face frame, and a linear head scores it. We retain that raw score and also subtract a fitted quality shift from its frame logits. Frame probabilities are averaged for each score. Both scores share the same network run." },
];

export function LiveArchitecture() {
  const f = evidence.fusion;
  return <div className="rounded-3xl border border-slate-200 bg-white p-5 md:p-7">
    <h3 className="text-xl font-bold text-slate-900">Follow one uploaded video</h3>
    <p className="mt-2 max-w-3xl text-sm text-slate-600">Three network instances produce five inputs to a learned combiner. Raw and corrected GenD scores are both retained; this is not a majority vote. All fitting happens before deployment. An upload runs the fixed models; it does not train them.</p>
    <div className="mt-5 rounded-2xl bg-slate-100 p-4">
      <h4 className="font-bold">1. Find and align faces</h4>
      <p className="mt-1 text-sm text-slate-600">Sample up to the first 10 seconds (a 360-frame cap can shorten this). S3FD detects faces; 2D-FAN locates landmarks. Track up to two clear faces and align them into 150 × 150 crops. Each network applies its own input transform.</p>
    </div>
    <p className="my-3 text-center text-sm text-slate-500" aria-hidden="true">↓ Same face crops, three network paths ↓</p>
    <div className="grid gap-4 lg:grid-cols-3">
      {networks.map((net) => <div key={net.title} className="flex flex-col rounded-2xl border border-slate-200 p-4">
        <p className="text-xs text-slate-500">{net.origin}</p>
        <h4 className="mt-1 font-bold text-slate-900">{net.title}</h4>
        <p className="mt-2 text-sm text-slate-600">{net.input}<br />{net.view}</p>
        <ul className="mt-4 space-y-2">
          {net.members.map((id) => <li key={id} className={`rounded-lg border px-3 py-2 text-sm ${id === "dfd" || id.endsWith("_sc") ? "border-indigo-200 bg-brand-soft text-indigo-900" : "border-slate-200 bg-slate-50"}`}>
            <span className="block font-semibold">{id === "dfd" || id.endsWith("_sc") ? "Self-calibrated score" : "Raw score"}</span>
            <code className="text-xs">{id}</code>
          </li>)}
        </ul>
        <details className="mt-4 text-sm">
          <summary className="cursor-pointer font-semibold text-brand">How this path works</summary>
          <p className="mt-2 leading-relaxed text-slate-600">{net.detail}</p>
        </details>
      </div>)}
    </div>
    <p className="my-3 text-center text-sm text-slate-500" aria-hidden="true">↓ Five scores per face ↓</p>
    <div className="grid gap-4 md:grid-cols-2">
      <div className="rounded-2xl border border-indigo-200 bg-brand-soft p-4">
        <h4 className="font-bold text-indigo-900">2. Combine the five scores</h4>
        <p className="mt-1 text-sm text-slate-700">Convert each probability to a logit, multiply by its learned weight, add the intercept, then apply sigmoid. The logistic regression was fitted on {f.fitted_on_validation_videos} validation videos, each clean, blurred and compressed.</p>
      </div>
      <div className="rounded-2xl border border-slate-200 p-4">
        <h4 className="font-bold text-slate-900">3. Decide from the most suspicious face</h4>
        <p className="mt-1 text-sm text-slate-600">Take the largest combined fake probability across tracked faces. At 0.5 or above: FAKE; below: REAL. Confidence is max(p, 1 − p), not a guarantee of correctness. Invalid scores return an analysis error.</p>
      </div>
    </div>
    <details className="mt-5 border-t border-slate-200 pt-4">
      <summary className="cursor-pointer font-semibold text-brand">See the actual combiner weights and equation</summary>
      <p className="mt-3 break-words rounded-lg bg-slate-50 p-3 text-sm">z = {f.intercept.toFixed(3)} + Σ weightᵢ × logit(pᵢ); p(face is fake) = sigmoid(z)</p>
      <dl className="mt-3 grid gap-2 text-sm sm:grid-cols-2">
        {f.members.map((id, i) => <div key={id} className="flex justify-between gap-3 border-b border-slate-100 py-2"><dt>{labels[id]}</dt><dd className="font-semibold tabular-nums">{f.coef[i].toFixed(3)}</dd></div>)}
      </dl>
      <p className="mt-3 text-sm text-slate-600">A negative weight is possible because the inputs are correlated. These weights describe their joint fit, not the standalone quality of each detector.</p>
    </details>
    <p className="mt-5 text-sm text-slate-600"><b>Explanation branch:</b> the spatial heatmap and blur probe help inspect DFD-FCG. The blur-probe strict score is not an input to the live five-score combiner, and the heatmap does not explain the entire ensemble.</p>
    <p className="mt-3 text-xs text-slate-500">Sources: <Source path="service/app/fusion.json">live configuration</Source>, <Source path="service/app/pipeline.py">video pipeline</Source>, <Source path="service/app/gend_live.py">GenD scoring</Source>.</p>
  </div>;
}

export function DeploymentFlow() {
  return <details className="mt-4 rounded-2xl border border-slate-200 bg-white p-5">
    <summary className="cursor-pointer font-semibold text-brand">Where the website, models and results run</summary>
    <ol className="mt-4 grid gap-4 text-sm md:grid-cols-3">
      <li><h4 className="font-bold text-slate-900">Checker → Hugging Face</h4><p className="mt-1 text-slate-600">The React checker on Vercel sends the video to the Gradio Space. Face preparation runs on CPU; model inference uses ZeroGPU with CPU fallback. The response includes the verdict, confidence, timeline and heatmap.</p></li>
      <li><h4 className="font-bold text-slate-900">Study → saved result files</h4><p className="mt-1 text-slate-600">Offline evaluation produces the research tables and ensemble metrics. This viva page bundles those results at build time; loading it does not run the detectors.</p></li>
      <li><h4 className="font-bold text-slate-900">Results API → Neon</h4><p className="mt-1 text-slate-600">The separate Render API reads stored research results from Neon for the original research explorer. It is not the inference engine. The checker backend can also write anonymous analysis logs.</p></li>
    </ol>
    <p className="mt-4 text-xs text-slate-500">Sources: <Source path="service/space/app.py">Space wrapper</Source>, <Source path="web/checker/src/App.tsx">checker</Source>, <Source path="web/api/main.py">results API</Source>. DFD-FCG weights are research-only; the checker is free and non-commercial.</p>
  </details>;
}

export function ExtraWork() {
  const clip = evidence.experiments.round3_clip, paired = evidence.experiments.paired_v1;
  const train = evidence.paired_training;
  const percent = (n: number) => (n * 100).toFixed(2) + "%";
  return <section id="extra" className="scroll-mt-20 py-12">
    <p className="text-sm font-semibold text-brand">Beyond the initial self-calibration study</p>
    <h2 className="mt-1 text-2xl font-extrabold tracking-tight text-slate-900 md:text-3xl">Extra work completed, and what actually shipped</h2>
    <p className="mt-3 max-w-3xl text-slate-600">The later work includes reliability improvements, broader model comparisons and a paired training experiment. Completing an experiment is a contribution to the evidence, even when it does not justify replacing the live model.</p>
    <div className="mt-6 grid gap-4 md:grid-cols-2">
      <article className="rounded-2xl border border-green-200 bg-white p-5">
        <p className="text-sm font-semibold text-green-800">Deployed · reliability</p>
        <h3 className="mt-2 text-lg font-bold">Safer scoring and face tracking</h3>
        <p className="mt-2 text-sm text-slate-600">Added finite-score and missing-output checks, stable sigmoid calculations, a displacement gate for the main face track, and correct sampling when networks share a forward pass. Also normalized the Windows build for Linux deployment.</p>
        <p className="mt-2 text-sm text-slate-600">The recorded release passed 41 regression tests, model parity checks and live real/fake samples. These checks verify behavior; they do not establish a new accuracy gain.</p>
        <p className="mt-3 text-sm"><Source path="results/reliability_followup.md">Release record and verification</Source></p>
      </article>
      <article className="rounded-2xl border border-slate-200 bg-white p-5">
        <p className="text-sm font-semibold text-amber-800">Completed · no replacement</p>
        <h3 className="mt-2 text-lg font-bold">Reweighting and GenD-CLIP comparison</h3>
        <p className="mt-2 text-sm text-slate-600">Tested {evidence.reweight.options} validation-only reweighting options to reduce FF++ false alarms. None qualified: fewer false alarms came with more missed fakes.</p>
        <p className="mt-2 text-sm text-slate-600">Compared {clip.options} fusion subsets including CLIP candidates on {clip.validation_videos} validation videos. The selected fusion excluded CLIP. Newer-set balanced accuracy then fell from {percent(clip.live_new_bacc)} to {percent(clip.candidate_new_bacc)}, failing the release rule.</p>
        <p className="mt-3 text-sm"><Source path="results/ensemble/round3/selection.json">Reweighting results</Source> · <Source path="results/ensemble/round3_clip/test.json">CLIP comparison</Source></p>
      </article>
      <article className="rounded-2xl border border-slate-200 bg-white p-5">
        <p className="text-sm font-semibold text-amber-800">Completed · no replacement</p>
        <h3 className="mt-2 text-lg font-bold">Paired real/fake training</h3>
        <p className="mt-2 text-sm text-slate-600">Prepared 96 missing real counterparts and ran a 1,000-step continuation with {train.train_videos.toLocaleString("en-US")} training videos, {train.pairs} verified FF++ pairs and {train.validation_videos} validation videos.</p>
        <p className="mt-2 text-sm text-slate-600">Validation AUROC stayed best at step zero ({train.validation[0].auroc.toFixed(4)}). The retained checkpoint was identical in all 104 tensors. The subsequent {paired.options}-option fusion comparison also failed the release rule. This specific continuation did not help; it does not disprove paired training in general.</p>
        <p className="mt-3 text-sm"><Source path="results/ensemble/paired_v1/README.md">Training outcome</Source> · <Source path="results/ensemble/PAIRED_PLAN.md">Frozen protocol</Source></p>
      </article>
      <article className="rounded-2xl border border-slate-200 bg-white p-5">
        <p className="text-sm font-semibold text-brand">Documented · reproducible artifacts</p>
        <h3 className="mt-2 text-lg font-bold">Data checks and paper draft</h3>
        <p className="mt-2 text-sm text-slate-600">Added split-overlap checks, saved all candidate results and frozen selections, and kept an audit of prepared pairs. Later comparisons exclude the disputed synth validation cohort. These checks do not establish identity-disjoint evaluation.</p>
        <p className="mt-2 text-sm text-slate-600">A seven-page paper draft and scripts that generate its numbers are in the repository. It is a draft, not a peer-reviewed publication; author details and several references still need completion.</p>
        <p className="mt-3 text-sm"><Source path="paper/README.md">Paper status</Source> · <Source path="results/reliability_followup.md">Data and split audit</Source></p>
      </article>
    </div>
    <details className="mt-4 rounded-2xl border border-slate-200 bg-white p-5">
      <summary className="cursor-pointer font-semibold text-brand">Additional transfer experiment: the correction on PE and CLIP</summary>
      <p className="mt-3 text-sm text-slate-600">The same self-calibration recipe was fitted separately for released PE, fine-tuned PE and released CLIP. Strong-blur FF++ results below show both the reduced false alarms and the lost fake recall. These are individual-model scores, not the ensemble.</p>
      <div className="mt-3 overflow-x-auto"><table className="w-full min-w-[520px] text-left text-sm">
        <caption className="sr-only">Self-calibration transfer under strong blur, raw to corrected scores</caption>
        <thead><tr className="border-b border-slate-200"><th className="py-2">Network</th><th>False alarms</th><th>Fakes caught</th></tr></thead>
        <tbody>{Object.entries(evidence.blur_transfer).map(([id, row]) => <tr key={id} className="border-b border-slate-100">
          <th className="py-3 font-medium">{({ pe: "Released GenD-PE", pe_ft: "Fine-tuned GenD-PE", clip: "Released GenD-CLIP" } as Record<string, string>)[id]}</th>
          <td>{percent(row.raw.false_alarm)} → {percent(row.corrected.false_alarm)}</td>
          <td>{percent(row.raw.caught)} → {percent(row.corrected.caught)}</td>
        </tr>)}</tbody>
      </table></div>
      <p className="mt-3 text-sm"><Source path="results/ensemble/selfcal_transfer.json">Full transfer results</Source>. GenD-CLIP is not in the deployed ensemble.</p>
    </details>
    <p className="mt-4 rounded-2xl border-l-4 border-amber-500 bg-amber-50 p-4 text-sm text-slate-700">The live five-score model and its published accuracy remain unchanged by these follow-ups. CLIP and paired comparisons reuse previously reported test videos; they are follow-up benchmarks, not fresh blind holdouts.</p>
  </section>;
}
