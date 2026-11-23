import { useEffect, useMemo, useState } from 'react'
import type { AlertItem } from '../api/types'
import { fetchAlerts } from '../api/alertsApi'
import { Badge } from '../components/ui/Badge'
import { Button } from '../components/ui/Button'
import { Card, CardBody, CardHeader } from '../components/ui/Card'
import { Input } from '../components/ui/Input'
import { Select } from '../components/ui/Select'
import { Spinner } from '../components/ui/Spinner'
import { downloadCsv } from '../lib/csv'

function formatDateTime(iso: string) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString()
}

export function AlertsHistoryPage() {
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [from, setFrom] = useState('')
  const [to, setTo] = useState('')
  const [label, setLabel] = useState<'all' | 'Violence' | 'Non-Violence'>('all')

  const [items, setItems] = useState<AlertItem[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const totalPages = useMemo(() => Math.max(1, Math.ceil(total / pageSize)), [total, pageSize])

  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoading(true)
      setError(null)
      try {
        const res = await fetchAlerts({
          page,
          pageSize,
          from: from || undefined,
          to: to || undefined,
          label: label === 'all' ? undefined : label,
        })
        if (cancelled) return
        setItems(res.items)
        setTotal(res.total)
      } catch (e) {
        if (cancelled) return
        setError(
          (e as Error).message ||
            'Endpoint historique non disponible (prévoir GET /alerts)',
        )
        setItems([])
        setTotal(0)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [page, pageSize, from, to, label])

  function exportCsv() {
    downloadCsv(
      `alerts_${new Date().toISOString().slice(0, 10)}.csv`,
      items.map((a) => ({
        id: a.id,
        createdAt: a.createdAt,
        sourceType: a.sourceType,
        label: a.label,
        confidence: a.confidence,
      })),
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="bg-gradient-to-r from-rose-600 via-fuchsia-600 to-indigo-600 bg-clip-text text-xl font-bold text-transparent">
            Historique des alertes
          </h1>
          <p className="mt-1 text-sm text-slate-600">
            Toutes les analyses passées avec filtres, pagination, et export CSV.
          </p>
        </div>
        <Button onClick={exportCsv} disabled={!items.length} variant="secondary">
          Export CSV
        </Button>
      </div>

      <Card>
        <CardHeader>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
              <div>
                <div className="mb-1 text-xs text-slate-600">Du</div>
                <Input
                  type="date"
                  value={from}
                  onChange={(e) => {
                    setPage(1)
                    setFrom(e.target.value)
                  }}
                />
              </div>
              <div>
                <div className="mb-1 text-xs text-slate-600">Au</div>
                <Input
                  type="date"
                  value={to}
                  onChange={(e) => {
                    setPage(1)
                    setTo(e.target.value)
                  }}
                />
              </div>
              <div>
                <div className="mb-1 text-xs text-slate-600">Résultat</div>
                <Select
                  value={label}
                  onChange={(e) => {
                    setPage(1)
                    setLabel(e.target.value as typeof label)
                  }}
                >
                  <option value="all">Tous</option>
                  <option value="Violence">Violence</option>
                  <option value="Non-Violence">Non-Violence</option>
                </Select>
              </div>
              <div>
                <div className="mb-1 text-xs text-slate-600">Page size</div>
                <Select
                  value={String(pageSize)}
                  onChange={(e) => {
                    setPage(1)
                    setPageSize(Number(e.target.value))
                  }}
                >
                  <option value="10">10</option>
                  <option value="20">20</option>
                  <option value="50">50</option>
                </Select>
              </div>
            </div>

            <div className="flex items-center gap-2 text-xs text-slate-600">
              {loading ? <Spinner /> : null}
              <span>
                {total ? `${total} résultats` : '—'}
              </span>
            </div>
          </div>
        </CardHeader>

        <CardBody>
          {error ? (
            <div className="mb-4 rounded-xl border border-amber-300 bg-amber-50 p-3 text-sm text-amber-800">
              {error}
            </div>
          ) : null}

          <div className="overflow-hidden rounded-2xl border border-slate-200 shadow-sm">
            <table className="w-full text-left text-sm">
              <thead className="bg-gradient-to-r from-indigo-50 to-fuchsia-50 text-xs text-slate-700">
                <tr>
                  <th className="px-4 py-3">Date</th>
                  <th className="px-4 py-3">Source</th>
                  <th className="px-4 py-3">Label</th>
                  <th className="px-4 py-3 text-right">Confidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 bg-white/80">
                {items.length ? (
                  items.map((a) => {
                    const isViolent = String(a.label).toLowerCase().includes('viol')
                    return (
                      <tr key={a.id} className="hover:bg-indigo-50/70">
                        <td className="px-4 py-3 text-slate-800">
                          {formatDateTime(a.createdAt)}
                        </td>
                        <td className="px-4 py-3 text-slate-700">{a.sourceType}</td>
                        <td className="px-4 py-3">
                          <Badge variant={isViolent ? 'danger' : 'success'}>{a.label}</Badge>
                        </td>
                        <td className="px-4 py-3 text-right font-semibold text-slate-900">
                          {Math.round((a.confidence ?? 0) * 100)}%
                        </td>
                      </tr>
                    )
                  })
                ) : (
                  <tr>
                    <td colSpan={4} className="px-4 py-10 text-center text-slate-600">
                      Aucun résultat.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="text-xs text-slate-600">
              Page <span className="font-semibold text-indigo-700">{page}</span> /{' '}
              <span className="font-semibold text-indigo-700">{totalPages}</span>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
              >
                Précédent
              </Button>
              <Button
                variant="secondary"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
              >
                Suivant
              </Button>
            </div>
          </div>
        </CardBody>
      </Card>
    </div>
  )
}

