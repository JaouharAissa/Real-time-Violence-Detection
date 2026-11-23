import { useMemo, useState } from 'react'
import { Card, CardBody, CardHeader } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { Toggle } from '../components/ui/Toggle'
import type { AppSettings } from '../api/types'
import { loadSettings, saveSettings } from '../lib/settings'

function uid() {
  return `id_${Math.random().toString(16).slice(2)}_${Date.now()}`
}

export function SettingsPage() {
  const initial = useMemo(() => loadSettings(), [])
  const [settings, setSettings] = useState<AppSettings>(initial)
  const [hint, setHint] = useState<string | null>(null)

  function persist(next: AppSettings) {
    setSettings(next)
    saveSettings(next)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Paramètres</h1>
        <p className="mt-1 text-sm text-slate-600">
          Seuil de confiance, notifications, et gestion des sources caméra.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <div className="text-sm font-semibold text-slate-900">Détection</div>
          </CardHeader>
          <CardBody className="space-y-4">
            <div>
              <div className="flex items-center justify-between">
                <div className="text-sm text-slate-700">Seuil de confiance</div>
                <div className="text-sm font-semibold text-slate-900">
                  {Math.round(settings.confidenceThreshold * 100)}%
                </div>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={settings.confidenceThreshold}
                onChange={(e) =>
                  persist({
                    ...settings,
                    confidenceThreshold: Number(e.target.value),
                  })
                }
                className="mt-3 w-full accent-indigo-600"
              />
              <div className="mt-2 text-xs text-slate-600">
                Utilisé par l’upload et le live stream (envoyé au backend).
              </div>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <div className="text-sm font-semibold text-slate-900">Notifications</div>
          </CardHeader>
          <CardBody className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-slate-700">Activer notifications email</div>
                <div className="mt-1 text-xs text-slate-600">
                  Le backend envoie l’email quand une alerte “Violence” est créée (via SMTP).
                </div>
              </div>
              <Toggle
                checked={settings.notificationsEnabled}
                onChange={(e) =>
                  persist({ ...settings, notificationsEnabled: e.target.checked })
                }
              />
            </div>

            <div>
              <div className="mb-1 text-xs text-slate-500">Email destinataire</div>
              <Input
                placeholder="owner@example.com"
                value={settings.notificationEmail}
                onChange={(e) =>
                  persist({ ...settings, notificationEmail: e.target.value })
                }
              />
              <div className="mt-2 text-xs text-slate-600">
                À copier dans ton `.env` Docker: <span className="font-mono">NOTIFY_EMAIL_TO</span>
              </div>
            </div>
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="text-sm font-semibold text-slate-900">Sources caméra</div>
              <div className="text-xs text-slate-600">
                Exemple: <span className="font-mono">0</span> (webcam locale), RTSP/URL, etc.
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                onClick={() => {
                  setHint(null)
                  const idx = settings.cameraSources.findIndex((s) => s.url.trim() === '0')
                  if (idx >= 0) {
                    const next = [...settings.cameraSources]
                    next[idx] = { ...next[idx], name: 'Webcam PC', url: '0' }
                    persist({ ...settings, cameraSources: next })
                    setHint('Webcam PC (0) déjà présente — renommée.')
                    return
                  }
                  persist({
                    ...settings,
                    cameraSources: settings.cameraSources.concat({
                      id: uid(),
                      name: 'Webcam PC',
                      url: '0',
                    }),
                  })
                  setHint('Webcam PC (0) ajoutée.')
                }}
              >
                Ajouter Webcam PC (0)
              </Button>
              <Button
                variant="secondary"
                onClick={() =>
                  persist({
                    ...settings,
                    cameraSources: settings.cameraSources.concat({
                      id: uid(),
                      name: `Caméra ${settings.cameraSources.length + 1}`,
                      url: '',
                    }),
                  })
                }
              >
                Ajouter
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardBody className="space-y-3">
          {hint ? (
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
              {hint}
            </div>
          ) : null}
          {settings.cameraSources.length ? (
            settings.cameraSources.map((s, idx) => (
              <div
                key={s.id}
                className="grid grid-cols-1 gap-3 rounded-2xl border border-slate-200 bg-white/60 p-4 shadow-sm md:grid-cols-[1fr,1.5fr,110px]"
              >
                <div>
                  <div className="mb-1 text-xs text-slate-500">Nom</div>
                  <Input
                    value={s.name}
                    onChange={(e) => {
                      const next = [...settings.cameraSources]
                      next[idx] = { ...s, name: e.target.value }
                      persist({ ...settings, cameraSources: next })
                    }}
                  />
                </div>
                <div>
                  <div className="mb-1 text-xs text-slate-500">URL / ID</div>
                  <Input
                    placeholder="0, 1, rtsp://..., http://..."
                    value={s.url}
                    onChange={(e) => {
                      const next = [...settings.cameraSources]
                      next[idx] = { ...s, url: e.target.value }
                      persist({ ...settings, cameraSources: next })
                    }}
                  />
                </div>
                <div className="flex items-end">
                  <Button
                    variant="danger"
                    className="w-full"
                    onClick={() =>
                      persist({
                        ...settings,
                        cameraSources: settings.cameraSources.filter((x) => x.id !== s.id),
                      })
                    }
                    disabled={settings.cameraSources.length <= 1}
                    title={
                      settings.cameraSources.length <= 1
                        ? 'Garde au moins une source'
                        : 'Supprimer'
                    }
                  >
                    Supprimer
                  </Button>
                </div>
              </div>
            ))
          ) : (
            <div className="rounded-xl border border-dashed border-slate-300 p-6 text-center text-sm text-slate-600">
              Aucune source. Clique sur “Ajouter”.
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  )
}

