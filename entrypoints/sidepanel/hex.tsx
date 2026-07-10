// Animated Enclave hexagon logo states, shared by the first-run card (HexDraw) and the
// cache→GPU loading state (HexSpinner). Geometry matches Logo.tsx / public/logo.svg.

export const HEX_POINTS = '64,8 114,36 114,92 64,120 14,92 14,36';
export const SPARK_PATH =
  'M64 36 C65.5 50 78 62.5 92 64 C78 65.5 65.5 78 64 92 C62.5 78 50 65.5 36 64 C50 62.5 62.5 50 64 36 Z';

/** The hexagon, stroke-drawn once when the first-run card mounts ("sealing the enclave"), with
 *  the spark fading in after. Static under prefers-reduced-motion. */
export function HexDraw() {
  return (
    <svg width={44} height={44} viewBox="0 0 128 128" fill="none" aria-hidden="true">
      <polygon
        className="onboard-hex text-zinc-400 dark:text-zinc-500"
        points={HEX_POINTS}
        pathLength={1}
        stroke="currentColor"
        strokeWidth="6"
        strokeLinejoin="round"
      />
      <path className="onboard-spark text-zinc-800 dark:text-zinc-200" d={SPARK_PATH} fill="currentColor" />
    </svg>
  );
}

/** Enclave-branded loader: a dash orbits the hexagon while the spark breathes. Shown while model
 *  weights load from the local cache onto the GPU. Static under prefers-reduced-motion. */
export function HexSpinner({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center gap-2" role="status" aria-label={label}>
      <svg width={40} height={40} viewBox="0 0 128 128" fill="none" aria-hidden="true">
        <polygon
          className="text-zinc-200 dark:text-zinc-700"
          points={HEX_POINTS}
          stroke="currentColor"
          strokeWidth="6"
          strokeLinejoin="round"
        />
        <polygon
          className="hex-spinner-dash text-zinc-700 dark:text-zinc-300"
          points={HEX_POINTS}
          pathLength={1}
          stroke="currentColor"
          strokeWidth="6"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        <path className="hex-spinner-spark text-zinc-800 dark:text-zinc-200" d={SPARK_PATH} fill="currentColor" />
      </svg>
      <span className="text-[11px] text-zinc-400 dark:text-zinc-500">{label}</span>
    </div>
  );
}
