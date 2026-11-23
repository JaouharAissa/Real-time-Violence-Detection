export type ViolenceLabel = 'Violence' | 'Non-Violence' | string

export type PredictResponse = {
  label?: ViolenceLabel
  prediction?: ViolenceLabel
  confidence: number
  // Optional: base64-encoded image or a URL depending on your backend.
  optical_flow_visualization?: string | null
  // Optional: extra info from model.
  meta?: Record<string, unknown>
}

export type AlertSourceType = 'upload' | 'live' | 'batch' | string

export type AlertItem = {
  id: string
  createdAt: string
  sourceType: AlertSourceType
  label: ViolenceLabel
  confidence: number
}

export type HourlyActivityPoint = { hour: number; total: number; violent: number }

export type DashboardStats = {
  totalAnalyses: number
  violenceRate: number
  alertsToday: number
  activityByHour: HourlyActivityPoint[]
  recentAlerts: AlertItem[]
}

export type AppSettings = {
  confidenceThreshold: number
  notificationsEnabled: boolean
  notificationEmail: string
  cameraSources: { id: string; name: string; url: string }[]
}

