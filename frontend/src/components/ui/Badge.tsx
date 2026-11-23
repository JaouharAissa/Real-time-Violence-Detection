import type { HTMLAttributes, PropsWithChildren } from 'react'

type Variant = 'success' | 'warning' | 'danger' | 'neutral'

export function Badge({
  children,
  className,
  variant = 'neutral',
  ...props
}: PropsWithChildren<HTMLAttributes<HTMLSpanElement> & { variant?: Variant }>) {
  const base =
    'inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset'
  const variants: Record<Variant, string> = {
    success: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
    warning: 'bg-amber-50 text-amber-800 ring-amber-200',
    danger: 'bg-rose-50 text-rose-700 ring-rose-200',
    neutral: 'bg-slate-100 text-slate-700 ring-slate-200',
  }

  return (
    <span className={[base, variants[variant], className].join(' ')} {...props}>
      {children}
    </span>
  )
}

