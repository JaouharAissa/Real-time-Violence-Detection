export function PieChart({
  value,
  size = 120,
  stroke = 14,
  primaryClass = 'text-rose-400',
  secondaryClass = 'text-emerald-400',
}: {
  value: number // 0..1 (primary slice)
  size?: number
  stroke?: number
  primaryClass?: string
  secondaryClass?: string
}) {
  const v = Math.max(0, Math.min(1, value))
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const primary = c * v
  const secondary = c - primary

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <g transform={`rotate(-90 ${size / 2} ${size / 2})`}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="currentColor"
          strokeWidth={stroke}
          className={`${secondaryClass} opacity-70`}
          strokeDasharray={`${secondary} ${primary}`}
          strokeDashoffset={primary}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="currentColor"
          strokeWidth={stroke}
          className={primaryClass}
          strokeDasharray={`${primary} ${secondary}`}
          strokeLinecap="round"
        />
      </g>
    </svg>
  )
}

