import type { ButtonHTMLAttributes, PropsWithChildren } from 'react'

type Variant = 'primary' | 'secondary' | 'danger' | 'ghost'

export function Button({
  children,
  className,
  variant = 'primary',
  ...props
}: PropsWithChildren<
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }
>) {
  const base =
    'inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60'
  const variants: Record<Variant, string> = {
    primary:
      'bg-indigo-600 text-white shadow-sm hover:bg-indigo-500 focus:ring-indigo-500 focus:ring-offset-white',
    secondary:
      'bg-slate-100 text-slate-900 ring-1 ring-inset ring-slate-200 hover:bg-slate-200 focus:ring-slate-400 focus:ring-offset-white',
    danger:
      'bg-rose-600 text-white shadow-sm hover:bg-rose-500 focus:ring-rose-500 focus:ring-offset-white',
    ghost:
      'bg-transparent text-slate-800 hover:bg-slate-100 focus:ring-slate-400 focus:ring-offset-white',
  }

  return (
    <button className={[base, variants[variant], className].join(' ')} {...props}>
      {children}
    </button>
  )
}

