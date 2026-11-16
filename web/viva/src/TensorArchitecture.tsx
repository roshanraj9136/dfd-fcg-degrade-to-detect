import { useState } from "react";
import type { ReactNode } from "react";
import { Source } from "./Evidence";

const asset = "/architecture/";
const pdf = asset + "architecture-explained.pdf";

function Shape({ children }: { children: ReactNode }) {
  return <code className="rounded-md bg-slate-100 px-2 py-1 text-sm font-semibold text-slate-900">{children}</code>;
}

function Easy({ children }: { children: ReactNode }) {
  return <p lang="hi-Latn" className="mt-4 border-l-4 border-teal-600 bg-teal-50 px-4 py-3 text-sm leading-relaxed text-teal-950"><b>Easy Hinglish: </b>{children}</p>;
}

export function ArchitectureHero() {
  return <section id="top" className="grid items-center gap-8 py-12 lg:grid-cols-[1fr_1fr] lg:py-16">
    <div>
      <p className="font-semibold text-brand">Roshan Raj’s B.Tech project</p>
      <h1 className="mt-4 text-4xl font-extrabold leading-[1.08] tracking-tight text-slate-950 sm:text-5xl">How our deepfake<br className="hidden sm:block" /> checker works.</h1>
      <p className="mt-5 max-w-xl text-lg leading-relaxed text-slate-600">Follow one video from face crops to a REAL or FAKE verdict. Understand every input, feature vector and connection—and exactly what we added to the published models.</p>
      <div className="mt-7 flex flex-wrap gap-3">
        <a className="btn-primary" href="#dimensions">See dimensions & diagrams</a>
        <a className="btn-ghost" href={pdf} download>Download architecture PDF</a>
      </div>
      <p className="mt-4 text-sm text-slate-500">Simple explanations, easy Hinglish recaps and links to the code.</p>
    </div>
    <div className="architecture-board rounded-3xl border border-indigo-200 p-5 sm:p-7" aria-label="Overview: a face video goes through three networks, producing five scores for one verdict">
      <div className="flex items-center gap-4 border-b border-indigo-200 pb-4">
        <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-white text-3xl" aria-hidden="true">▤</div>
        <div><p className="font-bold">One uploaded video</p><p className="text-sm text-slate-600">Find faces → align → sample frames</p></div>
      </div>
      <div className="mt-5 space-y-3">
        {[
          ["DFD-FCG", "10-frame clips · appearance + motion", "1 score"],
          ["Our fine-tuned GenD-PE", "Individual frames · raw + corrected", "2 scores"],
          ["Released GenD-PE", "Individual frames · raw + corrected", "2 scores"],
        ].map(([title, sub, count]) => <div key={title} className="flex items-center justify-between gap-3 rounded-xl bg-white p-4 shadow-sm">
          <div><p className="font-semibold text-slate-900">{title}</p><p className="mt-1 text-xs text-slate-600">{sub}</p></div><span className="shrink-0 rounded-lg bg-indigo-50 px-2 py-1 text-sm font-bold text-brand">{count}</span>
        </div>)}
      </div>
      <div className="py-2 text-center text-indigo-500" aria-hidden="true">↓</div>
      <div className="rounded-xl bg-brand px-4 py-3 text-center text-white"><b>5 score logits → learned combiner</b><p className="mt-1 text-sm text-indigo-100">One fake probability per face → video verdict</p></div>
      <p className="mt-4 text-center text-xs text-slate-600">3 network instances · 2 detector families · 5 score inputs</p>
    </div>
  </section>;
}

export function TensorArchitecture() {
  const [diagram, setDiagram] = useState<"dfd" | "ensemble">("dfd");
  const [zoom, setZoom] = useState(false);
  const file = diagram === "dfd" ? "dfd-dimensions" : "ensemble-dimensions";
  return <section id="dimensions" className="scroll-mt-24 py-12">
    <p className="font-semibold text-brand">Read the architecture, one shape at a time</p>
    <h2 className="mt-2 text-3xl font-extrabold tracking-tight text-slate-950 sm:text-4xl">Which vector? What size? Where does it go?</h2>
    <p className="mt-4 max-w-3xl leading-relaxed text-slate-600">A vector is a list of learned numbers. A 1024-dimensional feature means 1024 numbers describing the input; it does not mean 1024 pixels. A tensor is an array with several dimensions.</p>

    <dl className="mt-6 grid gap-x-8 gap-y-4 border-y border-slate-200 py-5 sm:grid-cols-3">
      <div><dt className="font-bold">B = clips in this batch</dt><dd className="mt-1 text-sm text-slate-600">One DFD forward pass processes B clips. The face has 8 clips by default; they can be processed in smaller batches.</dd></div>
      <div><dt className="font-bold">M = sampled GenD frames</dt><dd className="mt-1 text-sm text-slate-600">Up to 24 available frames from the face. GenD scores each frame individually.</dd></div>
      <div><dt className="font-bold">A scalar = one number</dt><dd className="mt-1 text-sm text-slate-600">After averaging, each member gives one fake probability for the face.</dd></div>
    </dl>

    <div className="mt-7 rounded-2xl border border-slate-200 bg-white">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 p-4">
        <div className="flex flex-wrap gap-2" role="group" aria-label="Choose architecture diagram">
          {([ ["dfd", "1. Inside DFD-FCG"], ["ensemble", "2. GenD + final fusion"] ] as const).map(([id, label]) => <button key={id} aria-pressed={diagram === id} onClick={() => { setDiagram(id); setZoom(false); }} className={`rounded-lg px-4 py-2 text-sm font-semibold ${diagram === id ? "bg-brand text-white" : "bg-slate-100 text-slate-700 hover:bg-slate-200"}`}>{label}</button>)}
        </div>
        <button className="rounded-lg border border-slate-300 px-3 py-2 text-sm font-semibold text-slate-700" aria-pressed={zoom} onClick={() => setZoom(!zoom)}>{zoom ? "Fit to page" : "Enlarge diagram"}</button>
      </div>
      <figure>
        <div className="overflow-x-auto rounded-b-2xl" tabIndex={zoom ? 0 : undefined} aria-label="Architecture diagram; scroll horizontally when enlarged">
          <img className={zoom ? "block w-[1600px] max-w-none" : "block h-auto w-full"} src={`${asset}${file}.svg`} width="1600" height={diagram === "dfd" ? "1770" : "1640"} alt={diagram === "dfd" ? "DFD-FCG: B by 10 by 3 by 224 by 224 input, 1024 spatial and 256 temporal features, 1280 concatenation, three heads and our quality correction. Full explanation below." : "Two GenD-PE networks: M by 3 by 224 by 224 input, 1024 features, raw and corrected scores; five score logits feed the final combiner. Full explanation below."} />
        </div>
        <figcaption className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-slate-200 p-4 text-sm">
          <a href={`${asset}${file}.svg`} target="_blank" rel="noreferrer" className="font-semibold text-brand underline">Open full-size image</a>
          <a href={`${asset}${file}.png`} download className="font-semibold text-brand underline">Download PNG</a>
          <a href={pdf} download className="font-semibold text-brand underline">Download both diagrams as PDF</a>
        </figcaption>
      </figure>
    </div>

    <div className="mt-10 space-y-10">
      <article>
        <h3 className="text-xl font-bold">1. From face pixels to CLIP tokens</h3>
        <p className="mt-3 max-w-3xl text-slate-600">The aligned crop is 150 × 150 RGB. DFD selects 10 frames per clip, resizes each to 224 × 224, normalizes it and changes the channel order.</p>
        <div className="mt-4 flex flex-wrap items-center gap-3"><Shape>[N, 150, 150, 3]</Shape><span aria-hidden="true">→</span><Shape>[B, 10, 3, 224, 224]</Shape></div>
        <p className="mt-4 max-w-3xl text-slate-600">Read the input as <b>B clips × 10 frames × 3 colour channels × 224 height × 224 width</b>. CLIP ViT-L/14 divides a frame into 14 × 14 pixel patches: 224 ÷ 14 = 16 per side, so 16 × 16 = 256 patches. Add one CLS summary token: <b>257 tokens, each 1024 values long</b>.</p>
        <p className="mt-4"><Shape>[B, 10, 257, 1024]</Shape></p>
        <p className="mt-3 text-sm text-slate-600">Example: processing 2 clips together means input [2, 10, 3, 224, 224]. B is 2 in that call; the model still processes 10 frames per clip.</p>
        <Easy>Video se face crop nikaalte hain. Ek clip mein 10 frames hain. Har frame ko 256 chhote patches mein baantte hain. Har patch ki 1024-number representation banti hai; side branches CLS ko hata kar patches use karti hain.</Easy>
      </article>

      <article>
        <h3 className="text-xl font-bold">2. DFD makes two feature vectors</h3>
        <div className="mt-4 grid gap-6 md:grid-cols-2">
          <div className="border-l-4 border-indigo-500 pl-5"><h4 className="font-bold">Spatial y_s: 1024 numbers</h4><p className="mt-2 text-sm leading-relaxed text-slate-600">Four learned queries attend to lips, skin, eyes and nose. Their weighted patch features have shape <b>[B, 10, 4, 1024]</b>. Average across 10 frames and 4 queries, then sum the 24 layers: <b>[B, 1024]</b>.</p><p className="mt-2 text-sm text-slate-600">It summarizes appearance. The four queries are pooled; they do not make a 4096-number concatenation.</p></div>
          <div className="border-l-4 border-teal-600 pl-5"><h4 className="font-bold">Temporal y_t: 256 numbers</h4><p className="mt-2 text-sm leading-relaxed text-slate-600">Compare frames at each patch using Q, K and V. Learned temporal and spatial operations produce <b>[B, 1, 16, 16]</b>. Flatten the patch grid and sum the 24 layers: <b>[B, 256]</b>.</p><p className="mt-2 text-sm text-slate-600">It summarizes changes across frames. Here 256 = 16 × 16 patch responses; it is not the frame count.</p></div>
        </div>
        <details className="mt-5 rounded-xl border border-slate-200 bg-white p-4">
          <summary className="cursor-pointer font-semibold text-brand">Technical detail: Q / K / V and temporal concatenation</summary>
          <p className="mt-3 text-sm leading-relaxed text-slate-600">Q, K and V mean query, key and value projections. CLIP has 16 attention heads, each 64 values wide (16 × 64 = 1024). For each of Q, K and V, the branch builds a 10 × 10 frame-affinity matrix at each of 256 patches: <b>[B × 256, 16, 10, 10]</b>.</p>
          <p className="mt-3 text-sm leading-relaxed text-slate-600">Concatenate the three affinity tensors along their channel axis: <b>[B × 256, 48, 10, 10]</b>, because 16 + 16 + 16 = 48. A temporal convolution gives <b>[B × 256, 1, 10, 10]</b>. Rearrange to <b>[B, 16, 16, 100]</b>; apply the residual projection (100 → 10 → 100), rearrange to <b>[B, 100, 16, 16]</b>, then spatial convolution gives <b>[B, 1, 16, 16]</b>.</p>
        </details>
      </article>

      <article id="concatenation" className="scroll-mt-24">
        <h3 className="text-xl font-bold">3. Concatenation: append the two lists</h3>
        <div className="mt-5 flex flex-wrap items-center gap-3 text-lg font-bold"><span className="rounded-xl bg-indigo-100 px-4 py-3 text-indigo-900">Spatial · 1024</span><span>+</span><span className="rounded-xl bg-teal-100 px-4 py-3 text-teal-900">Temporal · 256</span><span>=</span><span className="rounded-xl bg-slate-900 px-4 py-3 text-white">Combined · 1280</span></div>
        <p className="mt-4 text-slate-600">Toy example: <code>[s1, s2, s3]</code> and <code>[t1, t2]</code> become <code>[s1, s2, s3, t1, t2]</code>. No values are added or averaged. The batch axis stays B; the feature axis gets longer.</p>
        <div className="mt-5 grid gap-5 md:grid-cols-2">
          <div className="rounded-xl border border-indigo-200 bg-white p-5"><h4 className="font-bold">Inside the authors’ classifier</h4><p className="mt-2 text-sm leading-relaxed text-slate-600">Apply LayerNorm to each vector, then concatenate. Linear(1280 → 2) produces REAL and FAKE logits. The spatial and temporal vectors also have their own 2-logit heads: three heads in total.</p><pre className="mt-3 overflow-x-auto rounded-lg bg-slate-50 p-3 text-xs leading-relaxed">{`s = LayerNorm(y_s)         # [B, 1024]\nt = LayerNorm(y_t)         # [B, 256]\nfused = cat([s, t], dim=-1) # [B, 1280]\nfused_logits = head(fused) # [B, 2]`}</pre></div>
          <div className="rounded-xl border border-teal-200 bg-teal-50 p-5"><h4 className="font-bold text-teal-900">Inside our quality correction</h4><p className="mt-2 text-sm leading-relaxed text-slate-700">Concatenate the raw vectors before those LayerNorms. Standardize using the read-out’s fitted mean and standard deviation. Ridge regression predicts one shift per head.</p><pre className="mt-3 overflow-x-auto rounded-lg bg-white p-3 text-xs leading-relaxed">{`u = cat([y_s, y_t], dim=-1) # [B, 1280]\nu_std = (u - mean) / std   # [B, 1280]\nG = u_std @ W.T + b        # [B, 3]\n# W: [3, 1280]; b: [3]`}</pre></div>
        </div>
        <p className="mt-4 max-w-3xl text-slate-600">For each head, subtract REAL logit from FAKE logit. Stack those differences into D: <b>[B, 3]</b>. Subtract the predicted shifts G, apply sigmoid to each head, average the three probabilities, then average over clips. Result: <b>one corrected DFD probability per face</b>.</p>
        <Easy>1024 spatial numbers ke baad 256 temporal numbers rakh diye: total 1280. Classifier normalized features jodta hai; hamara read-out raw features jod kar standardize karta hai. Read-out teen shifts deta hai—har head ke liye ek—jinhe logits se subtract karte hain.</Easy>
      </article>

      <article>
        <h3 className="text-xl font-bold">4. GenD: a 1024-number vector for each frame</h3>
        <p className="mt-3 max-w-3xl leading-relaxed text-slate-600">Both GenD-PE instances take <b>[M, 3, 224, 224]</b>. Perception Encoder produces <b>[M, 1024]</b>. L2 normalization gives each feature vector length 1; the linear head maps 1024 → 2 logits. FAKE minus REAL gives <b>[M]</b>.</p>
        <div className="mt-4 grid gap-5 sm:grid-cols-2"><div className="rounded-xl bg-white p-5 ring-1 ring-slate-200"><h4 className="font-bold">Raw score</h4><p className="mt-2 text-sm text-slate-600">Sigmoid each frame’s logit difference, then average the frame probabilities. One scalar.</p></div><div className="rounded-xl bg-teal-50 p-5 ring-1 ring-teal-200"><h4 className="font-bold">Corrected score</h4><p className="mt-2 text-sm text-slate-700">Standardize the L2-normalized features. A separately fitted 1024 → 1 read-out predicts a shift for each frame. Subtract it, apply sigmoid, then average. One scalar.</p></div></div>
        <p className="mt-4 text-sm text-slate-600">The two instances have different weights and separately fitted read-outs. Within each instance, raw and corrected scores share one forward pass. The PE model name contains “336”, but this live implementation explicitly resizes inputs to 224 with dynamic image sizing.</p>
        <Easy>GenD har frame ko alag dekhta hai. Ek network run se do answers milte hain: original score aur quality correction ke baad score. Released aur fine-tuned GenD mila kar chaar scores dete hain.</Easy>
      </article>

      <article>
        <h3 className="text-xl font-bold">5. The final vector has only five numbers</h3>
        <p className="mt-3 max-w-3xl text-slate-600">Take the five face probabilities, convert each to a logit, and stack them in this exact order:</p>
        <pre className="mt-4 overflow-x-auto rounded-xl bg-slate-900 p-5 text-sm leading-loose text-slate-100">{`x = [logit(p_dfd), logit(p_pe_ft), logit(p_pe_ft_sc),\n     logit(p_pe), logit(p_pe_sc)]                 # [5]\nz = dot(weights, x) + intercept                 # scalar\np_face = sigmoid(z)                             # scalar`}</pre>
        <p className="mt-4 max-w-3xl leading-relaxed text-slate-600">This is logistic regression fitted on validation videos. DFD’s 1280 features and GenD’s 1024 features stay inside their own paths. The final combiner receives <b>five score logits</b>. Take the highest face probability across up to two faces; ≥ 0.5 means FAKE, otherwise REAL.</p>
        <Easy>Final stage par bade feature vectors nahi jodte. DFD ka 1 score + fine-tuned GenD ke 2 + released GenD ke 2 = total 5. Learned weights se ek final probability banti hai. Ye majority voting nahi hai.</Easy>
      </article>
    </div>

    <details className="mt-8 rounded-xl border border-slate-200 bg-white p-5">
      <summary className="cursor-pointer font-semibold text-brand">Three different normalizations—what do they mean?</summary>
      <dl className="mt-4 space-y-3 text-sm text-slate-600"><div><dt className="font-bold text-slate-900">LayerNorm</dt><dd>Normalizes values within a feature vector, with learned scale and bias. DFD uses it before the heads; our GenD fine-tune updates the encoder’s LayerNorm parameters and classifier.</dd></div><div><dt className="font-bold text-slate-900">L2 normalization</dt><dd>Divides a GenD feature vector by its length. The output still has 1024 values, but its Euclidean length is 1.</dd></div><div><dt className="font-bold text-slate-900">Read-out standardization</dt><dd>Uses a mean and standard deviation fitted separately for every feature on the fitting data. Those statistics stay fixed for uploads.</dd></div></dl>
    </details>
    <p className="mt-5 text-sm leading-relaxed text-slate-500">Checked against <a href="https://github.com/aiiu-lab/DFD-FCG/blob/main/src/model/clip/svl.py" className="text-brand underline">the authors’ DFD architecture</a>, <Source path="qa/engine.py">feature extraction</Source>, <Source path="service/app/model.json">DFD read-out dimensions</Source>, <Source path="service/app/gend_live.py">live GenD</Source> and <Source path="service/app/fusion.json">fusion parameters</Source>.</p>
  </section>;
}

const papers = [
  { name: "DFD-FCG", title: "Towards More General Video-based Deepfake Detection through Facial Component Guided Adaptation for Foundation Model", authors: "Han, Huang, Hua & Chen · CVPR 2025", url: "https://openaccess.thecvf.com/content/CVPR2025/html/Han_Towards_More_General_Video-based_Deepfake_Detection_through_Facial_Component_Guided_CVPR_2025_paper.html", role: "Our base video detector. Supplies CLIP-based spatial and temporal side branches, facial-component queries and three classification heads.", ours: "We study its quality sensitivity, extend exact attribution and fit a quality-shift read-out while keeping the deployed detector frozen." },
  { name: "GenD", title: "Deepfake Detection that Generalizes Across Benchmarks", authors: "Yermakov, Cech, Matas & Fritz · WACV 2026", url: "https://openaccess.thecvf.com/content/WACV2026/papers/Yermakov_Deepfake_Detection_that_Generalizes_Across_Benchmarks_WACV_2026_paper.pdf", role: "Our frame-detector family. Supplies released weights and parameter-efficient adaptation using LayerNorm and a classifier on normalized visual features.", ours: "We fine-tune a PE version, fit separate quality read-outs and combine released and adapted scores with DFD. A later paired-training continuation did not improve the selected checkpoint." },
  { name: "CLIP", title: "Learning Transferable Visual Models From Natural Language Supervision", authors: "Radford et al. · ICML 2021", url: "https://proceedings.mlr.press/v139/radford21a.html", role: "The pretrained visual backbone inside DFD-FCG: ViT-L/14, 24 layers, width 1024. CLIP pretraining supplies visual representations.", ours: "We reuse this backbone through DFD-FCG. Current detection does not use text prompts. CLIP is not an extra fourth network in the live ensemble." },
  { name: "Perception Encoder (PE)", title: "Perception Encoder: The best visual embeddings are not at the output of the network", authors: "Bolya et al. · 2025 · arXiv:2504.13181", url: "https://arxiv.org/abs/2504.13181", role: "The visual backbone inside our two GenD-PE instances. Its frame representation feeds GenD’s real/fake classifier.", ours: "We use GenD’s PE-based checkpoint and adapt one instance. We did not train this foundation model from scratch." },
];

export function PaperMap() {
  return <section id="papers" className="scroll-mt-24 py-12">
    <p className="font-semibold text-brand">Where each part comes from</p>
    <h2 className="mt-2 text-3xl font-extrabold tracking-tight">Four papers. Clear roles.</h2>
    <p className="mt-3 max-w-3xl text-slate-600">DFD-FCG and GenD are the detector methods. CLIP and Perception Encoder provide their visual backbones. Our project extends and evaluates these existing models.</p>
    <div className="mt-6 divide-y divide-slate-200 rounded-2xl border border-slate-200 bg-white">
      {papers.map((p) => <article key={p.name} className="grid gap-4 p-5 sm:p-6 lg:grid-cols-[1fr_1.25fr]">
        <div><h3 className="text-lg font-bold">{p.name}</h3><p className="mt-1 text-sm text-slate-500">{p.authors}</p><a href={p.url} target="_blank" rel="noreferrer" className="mt-3 block text-sm font-semibold leading-relaxed text-brand underline decoration-indigo-200 underline-offset-4">{p.title}</a></div>
        <div className="text-sm leading-relaxed"><p className="text-slate-600"><b className="text-slate-900">Used for: </b>{p.role}</p><p className="mt-3 text-slate-600"><b className="text-teal-800">Our work: </b>{p.ours}</p></div>
      </article>)}
    </div>
    <Easy>Base video detector DFD-FCG hai, frame detector GenD hai. DFD ke andar CLIP aur GenD-PE ke andar Perception Encoder hai. Hamara kaam quality-bias study, correction, adaptation, evaluation aur final checker hai.</Easy>
  </section>;
}
