import type { Conditions, Summary } from "../data";
import { pct } from "../data";
import { Section } from "../components/ui";

export function Method({ summary, cond }: { summary: Summary; cond: Conditions }) {
  const curve = Object.entries(summary.lam_curve ?? {}).sort((a, b) => Number(a[0]) - Number(b[0]));
  const main = cond[summary.main];
  return (
    <Section id="method" kicker="Contribution" title="Degrade-to-Detect: stress-test the input, read the response"
      lead={<>A deepfake detector usually looks at a video once. We look twice: once at the input and once at a copy
        we degrade ourselves with a fixed, mild probe. Authentic faces lose fine detail and the model moves towards
        FAKE; manipulated faces lose their fragile artifacts and the model moves towards REAL. Subtracting the response
        pushes each class back to its own side of the threshold.</>}>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="card space-y-4">
          <h3 className="text-lg font-semibold text-white">The rule</h3>
          <div className="eq">r_h = d_h(probe(x)) − d_h(x)&nbsp;&nbsp;&nbsp; S_h = d_h − λ · r_h&nbsp;&nbsp;&nbsp; P = ⅓ Σ_h σ(S_h)</div>
          <p className="text-sm muted">d_h is the logit difference (FAKE − REAL) of DFD-FCG's spatial, temporal and fusion
            heads, exactly as in the released model, so the threshold stays at 0.5. The probe is a Gaussian blur with
            σ = 2 px on the 150 × 150 aligned face (a JPEG q50 probe is also reported). One extra forward pass; no
            retraining; nothing about the test video's history is needed.</p>
          <h3 className="pt-2 text-lg font-semibold text-white">One number is fitted</h3>
          <p className="text-sm muted">λ is chosen on FF++ <i>train-split</i> videos only ({summary.n.fit_real} real,{" "}
            {summary.n.fit_fake} fake, degradation severities 2 and 4) to maximise balanced accuracy at 0.5. Selected:
            <b className="text-white"> λ = {summary.lam}</b>. In the “unseen type” evaluation λ is re-selected without the
            tested degradation type. A learned variant (logistic regression on the 3 logits and the 6 responses of two
            probes, 10 numbers) is reported alongside.</p>
          {curve.length > 0 && (
            <table className="grid-table">
              <thead><tr><th>λ</th>{curve.map(([l]) => <th key={l}>{l}</th>)}</tr></thead>
              <tbody><tr><td className="muted">train-split balanced acc.</td>{curve.map(([l, v]) => <td key={l} className={Number(l) === summary.lam ? "text-real" : ""}>{pct(v)}</td>)}</tr></tbody>
            </table>
          )}
        </div>
        <div className="card space-y-4">
          <h3 className="text-lg font-semibold text-white">Why it is an extension of DFD-FCG, not a new model</h3>
          <ul className="list-disc space-y-2 pl-5 text-sm muted">
            <li>It uses the released checkpoint unchanged and the paper's own three heads and fusion rule.</li>
            <li>It was found through the exact additive explanation of those heads (our earlier work): the per-term
              changes under degradation showed that the evidence for REAL and for FAKE react oppositely.</li>
            <li>It answers the gap in the paper's robustness study, which measures AUROC only: the response restores
              a usable threshold under degradation instead of only ranking.</li>
            <li>The response is itself an explanation: for every verdict we report whether the evidence collapsed
              (fragile, typical of manipulation) or grew (fine detail removed, typical of authentic video).</li>
          </ul>
          {main && (
            <div className="rounded-xl border border-real/30 bg-real/5 p-4 text-sm">
              Clean test videos: AUROC {main.all.clean.auroc?.toFixed(3)} (released {cond.original.all.clean.auroc?.toFixed(3)}),
              false alarms {pct(main.all.clean["fpr@0.5"])}, fakes caught {pct(main.all.clean["tpr@0.5"])}. FaceShifter, a
              manipulation DFD-FCG never saw: AUROC {main.fsh.auroc?.toFixed(3)} (released {cond.original.fsh.auroc?.toFixed(3)}).
            </div>
          )}
          <p className="text-xs muted">Closest prior idea: RIGID (He et al., 2024) compares foundation-model embeddings of an
            image and its noised copy to detect generated images without training. Here the probe response is read off a
            trained video deepfake detector's own decision heads, and used to repair its calibration.</p>
        </div>
      </div>
    </Section>
  );
}
