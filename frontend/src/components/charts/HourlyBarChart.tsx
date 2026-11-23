import type { HourlyActivityPoint } from '../../api/types'

export function HourlyBarChart({
  points,
}: {
  points: HourlyActivityPoint[]
}) {
  const max = Math.max(1, ...points.map((p) => p.total))

  return (
    <div className="flex items-end gap-1">
      {points.map((p) => {
        const h = Math.round((p.total / max) * 100)
        const violentH = Math.round((p.violent / max) * 100)
        return (
          <div key={p.hour} className="group flex w-4 flex-col items-center gap-1">
            <div className="relative h-24 w-full overflow-hidden rounded-md bg-slate-800">
              <div
                className="absolute bottom-0 left-0 right-0 bg-slate-600"
                style={{ height: `${h}%` }}
              />
              <div
                className="absolute bottom-0 left-0 right-0 bg-rose-500/80"
                style={{ height: `${violentH}%` }}
              />
            </div>
            <div className="text-[10px] text-slate-500">{String(p.hour).padStart(2, '0')}</div>
          </div>
        )
      })}
    </div>
  )
}

