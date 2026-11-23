import type { AppSettings } from '../api/types'

const KEY = 'violence_app_settings_v1'

export const DEFAULT_SETTINGS: AppSettings = {
  confidenceThreshold: 0.7,
  notificationsEnabled: true,
  notificationEmail: '',
  cameraSources: [
    { id: 'cam-1', name: 'Caméra 1', url: '0' },
  ],
}

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return DEFAULT_SETTINGS
    const parsed = JSON.parse(raw) as Partial<AppSettings>
    return {
      ...DEFAULT_SETTINGS,
      ...parsed,
      cameraSources: Array.isArray(parsed.cameraSources)
        ? parsed.cameraSources
        : DEFAULT_SETTINGS.cameraSources,
    }
  } catch {
    return DEFAULT_SETTINGS
  }
}

export function saveSettings(next: AppSettings) {
  localStorage.setItem(KEY, JSON.stringify(next))
}

