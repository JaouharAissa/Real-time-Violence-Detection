import { useEffect, useMemo, useState } from 'react'
import type { AlertItem, DashboardStats } from '../api/types'
import { fetchDashboardStats, fetchRecentAlerts } from '../api/alertsApi'
import { Card, CardBody, CardHeader } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { Spinner } from '../components/ui/Spinner'
import { HourlyBarChart } from '../components/charts/HourlyBarChart'
import { PieChart } from '../components/charts/PieChart'

function formatPct(x: number) {
  if (!Number.isFinite(x)) return '—'
  return `${Math.round(x * 100)}%`
}

function formatDateTime(iso: string) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString()
}

export function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [recent, setRecent] = useState<AlertItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const activity = useMemo(() => {
    if (stats?.activityByHour?.length) return stats.activityByHour
    return Array.from({ length: 24 }, (_, hour) => ({ hour, total: 0, violent: 0 }))
  }, [stats])

  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoading(true)
      setError(null)
      try {
        const data = await fetchDashboardStats()
        if (!cancelled) setStats(data)
      } catch (e) {
        if (!cancelled) {
          setError((e as Error).message || 'Dashboard endpoint not available yet')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    async function poll() {
      try {
        const items = await fetchRecentAlerts()
        if (!cancelled) setRecent(items)
      } catch {
        // ignore: backend might not expose this yet
      }
    }
    void poll()
    const id = window.setInterval(poll, 5000)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])

  const totalAnalyses = stats?.totalAnalyses ?? '—'
  const violenceRate = stats ? formatPct(stats.violenceRate) : '—'
  const alertsToday = stats?.alertsToday ?? '—'
  const recentAlerts = stats?.recentAlerts?.length ? stats.recentAlerts : recent
  const violenceRateValue = stats?.violenceRate ?? 0

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="bg-gradient-to-r from-fuchsia-600 via-indigo-600 to-cyan-500 bg-clip-text text-xl font-bold text-transparent">
            Dashboard
          </h1>
          <p className="mt-1 text-sm text-slate-600">
            Vue globale des analyses et alertes en temps réel.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-600">
          {loading ? <Spinner /> : null}
          {error ? <Badge variant="warning">Endpoints dashboard non prêts</Badge> : null}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card className="border-fuchsia-200 bg-gradient-to-br from-fuchsia-500/20 via-white to-indigo-500/10">
          <CardBody>
            <div className="text-xs text-slate-600">Total analyses</div>
            <div className="mt-1 text-3xl font-semibold text-slate-900">{totalAnalyses}</div>
          </CardBody>
        </Card>
        <Card className="border-cyan-200 bg-gradient-to-br from-cyan-500/20 via-white to-emerald-500/10">
          <CardBody className="flex items-center justify-between gap-4">
            <div>
              <div className="text-xs text-slate-600">Taux de violence</div>
              <div className="mt-1 text-3xl font-semibold text-slate-900">{violenceRate}</div>
              <div className="mt-1 text-xs text-slate-600">
                Violence (rouge) vs Non-Violence (vert)
              </div>
            </div>
            <div className="shrink-0">
              <PieChart value={violenceRateValue} />
            </div>
          </CardBody>
        </Card>
        <Card className="border-indigo-200 bg-gradient-to-br from-indigo-500/20 via-white to-rose-500/10">
          <CardBody>
            <div className="text-xs text-slate-600">Alertes du jour</div>
            <div className="mt-1 text-3xl font-semibold text-slate-900">{alertsToday}</div>
          </CardBody>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-semibold text-slate-900">Activité par heure</div>
                <div className="text-xs text-slate-600">
                  Total (gris) vs violence (rouge)
                </div>
              </div>
            </div>
          </CardHeader>
          <CardBody>
            <HourlyBarChart points={activity} />
            {error ? (
              <p className="mt-3 text-xs text-slate-600">
                Astuce: expose un endpoint `GET /dashboard/stats` côté FastAPI pour alimenter cette
                section (sinon la page reste vide).
              </p>
            ) : null}
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold text-slate-900">Dernières alertes</div>
              <div className="text-xs text-slate-600">Actualisation ~5s</div>
            </div>
          </CardHeader>
          <CardBody>
            <div className="space-y-2">
              {recentAlerts?.length ? (
                recentAlerts.slice(0, 8).map((a) => (
                  <div
                    key={a.id}
                    className="flex items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white/80 px-3 py-2 shadow-sm"
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <Badge variant={String(a.label).toLowerCase().includes('viol') ? 'danger' : 'success'}>
                          {a.label}
                        </Badge>
                        <span className="text-xs text-slate-600">{a.sourceType}</span>
                      </div>
                      <div className="mt-1 truncate text-xs text-slate-600">
                        {formatDateTime(a.createdAt)}
                      </div>
                    </div>
                    <div className="shrink-0 text-sm font-semibold text-slate-900">
                      {Math.round((a.confidence ?? 0) * 100)}%
                    </div>
                  </div>
                ))
              ) : (
                <div className="rounded-xl border border-dashed border-slate-300 p-6 text-center text-sm text-slate-600">
                  Aucune alerte à afficher pour le moment.
                </div>
              )}
            </div>
          </CardBody>
        </Card>
      </div>
    </div>
  )
}

