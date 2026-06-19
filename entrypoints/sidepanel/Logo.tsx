export function Logo({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 128 128" fill="none" aria-hidden="true">
      <defs>
        <linearGradient id="encHex" x1="20" y1="8" x2="108" y2="120" gradientUnits="userSpaceOnUse">
          <stop stopColor="#3f3f46" />
          <stop offset="1" stopColor="#1c1c1f" />
        </linearGradient>
        <linearGradient id="encSpark" x1="40" y1="36" x2="88" y2="92" gradientUnits="userSpaceOnUse">
          <stop stopColor="#ffffff" />
          <stop offset="1" stopColor="#d4d4d8" />
        </linearGradient>
      </defs>
      <polygon
        points="64,8 114,36 114,92 64,120 14,92 14,36"
        fill="url(#encHex)"
        stroke="#52525b"
        strokeWidth="5"
        strokeLinejoin="round"
      />
      <polygon
        points="64,24 100,44 100,84 64,104 28,84 28,44"
        fill="none"
        stroke="#ffffff"
        strokeOpacity="0.1"
        strokeWidth="2.5"
        strokeLinejoin="round"
      />
      <path
        d="M64 36 C65.5 50 78 62.5 92 64 C78 65.5 65.5 78 64 92 C62.5 78 50 65.5 36 64 C50 62.5 62.5 50 64 36 Z"
        fill="url(#encSpark)"
      />
    </svg>
  );
}
