import { useMemo, useState } from 'react'
import type { PredictResponse } from '../api/types'
import { predictVideo } from '../api/violenceApi'
import { Card, CardBody, CardHeader } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { ProgressBar } from '../components/ui/ProgressBar'
import { Badge } from '../components/ui/Badge'
import { Input } from '../components/ui/Input'
import { loadSettings } from '../lib/settings'

function asImageSrc(value: string) {
  const v = value.trim()
  if (v.startsWith('http://') || v.startsWith('https://')) return v
  if (v.startsWith('data:image/')) return v
  // assume raw base64 (png/jpeg)
  return `data:image/png;base64,${v}`
}

export function UploadVideoPage() {
  const settings = useMemo(() => loadSettings(), [])
  const [file, setFile] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function onSubmit() {
    if (!file) return
    setBusy(true)
    setProgress(0)
    setResult(null)
    setError(null)
    try {
      const res = await predictVideo(
        file,
        { threshold: settings.confidenceThreshold },
        (pct) => setProgress(pct),
      )
      setResult(res)
    } catch (e) {
      setError((e as Error).message || 'Upload failed')
    } finally {
      setBusy(false)
    }
  }

  const label = result?.label
  const effectiveLabel = label ?? result?.prediction
  const isViolent = effectiveLabel
    ? String(effectiveLabel).toLowerCase().includes('viol')
    : false
  const confidencePct =
    result?.confidence != null ? Math.round(result.confidence * 100) : null

  return (
    <div className="space-y-6">
      <div>
        <h1 className="bg-gradient-to-r from-violet-600 via-fuchsia-600 to-rose-500 bg-clip-text text-xl font-bold text-transparent">
          Upload vidéo
        </h1>
        <p className="mt-1 text-sm text-slate-600">
          Upload d’une vidéo MP4 pour classification Violence / Non-Violence.
        </p>
      </div>

      <Card>
        <CardHeader>
          <div className="text-sm font-semibold text-slate-900">Nouvelle analyse</div>
        </CardHeader>
        <CardBody className="space-y-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-[1fr,220px]">
            <div>
              <div className="text-xs text-slate-600">Fichier vidéo</div>
              <Input
                type="file"
                accept="video/mp4"
                onChange={(e) => {
                  const f = e.target.files?.[0] ?? null
                  setFile(f)
                }}
              />
              <div className="mt-2 text-xs text-slate-600">
                Seuil confiance (paramètres):{' '}
                <span className="font-semibold text-indigo-700">
                  {Math.round(settings.confidenceThreshold * 100)}%
                </span>
              </div>
            </div>

            <div className="flex items-end">
              <Button
                onClick={() => void onSubmit()}
                disabled={!file || busy}
                className="w-full"
              >
                {busy ? 'Analyse en cours…' : 'Lancer l’analyse'}
              </Button>
            </div>
          </div>

          {busy ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs text-slate-600">
                <span>Upload</span>
                <span>{progress}%</span>
              </div>
              <ProgressBar value={progress} />
            </div>
          ) : null}

          {error ? (
            <div className="rounded-xl border border-rose-300 bg-rose-50 p-3 text-sm text-rose-700">
              {error}
            </div>
          ) : null}
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <div className="text-sm font-semibold text-slate-900">Résultat</div>
        </CardHeader>
        <CardBody>
          {result ? (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <div className="rounded-2xl border border-indigo-200 bg-gradient-to-br from-indigo-500/10 to-fuchsia-500/10 p-4">
                <div className="flex items-center justify-between">
                  <Badge variant={isViolent ? 'danger' : 'success'}>
                    {effectiveLabel ?? '—'}
                  </Badge>
                  <div className="text-sm font-semibold text-slate-900">
                    {confidencePct != null ? `${confidencePct}%` : '—'}
                  </div>
                </div>
                <div className="mt-3 text-xs text-slate-600">
                  API attendue: <span className="font-mono">POST /predict</span> (multipart form-data{' '}
                  <span className="font-mono">file</span>) →{' '}
                  <span className="font-mono">
                    {'{ label, confidence, optical_flow_visualization? }'}
                  </span>
                </div>
              </div>

              <div className="rounded-2xl border border-cyan-200 bg-gradient-to-br from-cyan-500/10 to-emerald-500/10 p-4">
                <div className="text-xs font-semibold text-slate-800">
                  Optical Flow visualization
                </div>
                {result.optical_flow_visualization ? (
                  <img
                    src={asImageSrc(result.optical_flow_visualization)}
                    alt="Optical flow"
                    className="mt-3 w-full rounded-xl border border-slate-200"
                  />
                ) : (
                  <div className="mt-3 rounded-xl border border-dashed border-slate-300 p-6 text-center text-sm text-slate-600">
                    Aucune visualisation reçue (champ `optical_flow_visualization`).
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-slate-300 p-6 text-center text-sm text-slate-600">
              Lance une analyse pour afficher le résultat ici.
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  )
}

