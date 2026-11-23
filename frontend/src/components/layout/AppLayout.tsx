import { NavLink, Outlet } from 'react-router-dom'

const navItems = [
  { to: '/', label: 'Dashboard' },
  { to: '/upload', label: 'Upload vidéo' },
  { to: '/live', label: 'Live Stream' },
  { to: '/alerts', label: 'Historique' },
  { to: '/settings', label: 'Paramètres' },
] as const

export function AppLayout() {
  return (
    <div className="min-h-full text-slate-900">
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[260px,1fr]">
          <aside className="rounded-2xl border border-slate-200/70 bg-white/70 p-4 shadow-sm backdrop-blur">
            <div className="mb-4">
              <div className="text-sm font-semibold tracking-wide text-slate-900">
                Surveillance intelligente
              </div>
              <div className="text-xs text-slate-500">
                Violence detection • Two-Stream + LSTM
              </div>
            </div>

            <nav className="space-y-1">
              {navItems.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === '/'}
                  className={({ isActive }) =>
                    [
                      'block rounded-xl px-3 py-2 text-sm transition',
                      isActive
                        ? 'bg-indigo-600 text-white shadow-sm'
                        : 'text-slate-700 hover:bg-slate-100 hover:text-slate-900',
                    ].join(' ')
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </nav>

            <div className="mt-6 rounded-xl border border-slate-200/70 bg-slate-50 p-3 text-xs text-slate-600">
              API: <span className="font-semibold text-slate-900">localhost:8000</span>
            </div>
          </aside>

          <main className="rounded-2xl border border-slate-200/70 bg-white/70 p-5 shadow-sm backdrop-blur sm:p-6">
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  )
}

