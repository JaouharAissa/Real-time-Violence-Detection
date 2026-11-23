export function ProgressBar({ value }: { value: number }) {
  const v = Math.max(0, Math.min(100, value))
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
      <div className="h-full bg-indigo-500 transition-all" style={{ width: `${v}%` }} />
    </div>
  )
}

