// 16x16 evidence map drawn over a schematic face built from the video's mean landmarks
// (FF++ frames are not redistributed; only the two samples bundled upstream show pixels).

const JAW = [...Array(17).keys()];
const BROW_L = [17, 18, 19, 20, 21];
const BROW_R = [22, 23, 24, 25, 26];
const NOSE = [27, 28, 29, 30, 33, 32, 31, 33, 34, 35];
const EYE_L = [36, 37, 38, 39, 40, 41, 36];
const EYE_R = [42, 43, 44, 45, 46, 47, 42];
const LIPS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48];

function color(v: number, vmax: number) {
  const a = Math.min(1, Math.abs(v) / (vmax || 1));
  return v >= 0 ? `rgba(244,114,182,${0.85 * a})` : `rgba(52,211,153,${0.85 * a})`;
}

export function FaceMap({ values, landmarks, image, title, vmax }: {
  values: number[]; landmarks: number[][]; image?: string; title: string; vmax?: number;
}) {
  const S = 150, G = 16, step = S / G;
  const m = vmax ?? Math.max(1e-6, ...values.map((v) => Math.abs(v)));
  const path = (idx: number[]) => idx.map((i, k) => `${k ? "L" : "M"}${landmarks[i][0]},${landmarks[i][1]}`).join(" ");
  const sum = values.reduce((a, b) => a + b, 0);
  return (
    <figure className="flex flex-col items-center">
      <svg viewBox={`0 0 ${S} ${S}`} className="w-full max-w-[260px] rounded-xl border border-edge bg-black/50">
        {image && <image href={image} x={0} y={0} width={S} height={S} opacity={0.75} />}
        {values.map((v, i) => (
          <rect key={i} x={(i % G) * step} y={Math.floor(i / G) * step} width={step} height={step} fill={color(v, m)} />
        ))}
        {landmarks.length === 68 && (
          <g fill="none" stroke="rgba(226,232,240,.55)" strokeWidth={0.8}>
            {[JAW, BROW_L, BROW_R, NOSE, EYE_L, EYE_R, LIPS].map((p, k) => <path key={k} d={path(p)} />)}
          </g>
        )}
      </svg>
      <figcaption className="mt-2 text-center text-xs muted">
        {title} <span className="font-mono">(sum {sum >= 0 ? "+" : ""}{sum.toFixed(2)})</span>
      </figcaption>
    </figure>
  );
}

export function Legend() {
  return (
    <div className="flex items-center gap-4 text-xs muted">
      <span className="flex items-center gap-1"><i className="inline-block h-3 w-3 rounded-sm bg-fake/80" /> pushes towards FAKE</span>
      <span className="flex items-center gap-1"><i className="inline-block h-3 w-3 rounded-sm bg-real/80" /> pushes towards REAL</span>
    </div>
  );
}
