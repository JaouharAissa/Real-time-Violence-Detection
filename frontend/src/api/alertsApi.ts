import { api } from './client'
import type { AlertItem, DashboardStats } from './types'

export async function fetchDashboardStats(): Promise<DashboardStats> {
  const res = await api.get('/dashboard/stats')
  return res.data as DashboardStats
}

export async function fetchRecentAlerts(): Promise<AlertItem[]> {
  const res = await api.get('/alerts/recent')
  return res.data as AlertItem[]
}

export async function fetchAlerts(params: {
  page: number
  pageSize: number
  from?: string
  to?: string
  label?: string
}): Promise<{ items: AlertItem[]; total: number }> {
  const res = await api.get('/alerts', {
    params: {
      ...params,
      from: params.from,
    },
  })
  return res.data as { items: AlertItem[]; total: number }
}

