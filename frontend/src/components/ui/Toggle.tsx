import type { InputHTMLAttributes } from 'react'

export function Toggle({
  className,
  ...props
}: Omit<InputHTMLAttributes<HTMLInputElement>, 'type'>) {
  return (
    <label className={['inline-flex cursor-pointer items-center', className].join(' ')}>
      <input type="checkbox" className="peer sr-only" {...props} />
      <span className="h-6 w-11 rounded-full bg-slate-700 transition peer-checked:bg-indigo-500 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-indigo-400/60" />
      <span className="ml-3 text-sm text-slate-200">{props.checked ? 'On' : 'Off'}</span>
    </label>
  )
}

