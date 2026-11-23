export function Spinner({ className }: { className?: string }) {
  return (
    <div
      className={[
        'h-4 w-4 animate-spin rounded-full border-2 border-slate-500 border-t-transparent',
        className,
      ].join(' ')}
      aria-label="Loading"
      role="status"
    />
  )
}

