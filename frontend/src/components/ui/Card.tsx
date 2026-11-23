import type { HTMLAttributes, PropsWithChildren } from 'react'

export function Card({
  children,
  className,
  ...props
}: PropsWithChildren<HTMLAttributes<HTMLDivElement>>) {
  return (
    <div
      className={[
        'rounded-2xl border border-slate-200/70 bg-white/80 shadow-sm backdrop-blur',
        className,
      ].join(' ')}
      {...props}
    >
      {children}
    </div>
  )
}

export function CardHeader({
  children,
  className,
  ...props
}: PropsWithChildren<HTMLAttributes<HTMLDivElement>>) {
  return (
    <div className={['border-b border-slate-200/70 p-4', className].join(' ')} {...props}>
      {children}
    </div>
  )
}

export function CardBody({
  children,
  className,
  ...props
}: PropsWithChildren<HTMLAttributes<HTMLDivElement>>) {
  return (
    <div className={['p-4', className].join(' ')} {...props}>
      {children}
    </div>
  )
}

