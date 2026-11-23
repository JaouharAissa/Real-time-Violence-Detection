import { useEffect, useMemo, useRef, useState } from 'react'
import { io, type Socket } from 'socket.io-client'
import { ENV } from '../lib/env'
import { loadSettings } from '../lib/settings'
import { Button } from '../components/ui/Button'
import { Card, CardBody, CardHeader } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { Select } from '../components/ui/Select'

type StreamFrameMessage = {
  frame?: string // base64 jpg/png or data URL
  label?: string
  confidence?: number
  ts?: string
}

function asImageSrc(value: string) {
  const v = value.trim()
  if (v.startsWith('http://') || v.startsWith('https://')) return v
  if (v.startsWith('data:image/')) return v
  return `data:image/jpeg;base64,${v}`
}

export function LiveStreamPage() {
  const initial = useMemo(() => loadSettings(), [])
  const defaultSourceId =
    initial.cameraSources.find((s) => s.url.trim() === '0')?.id ??
    initial.cameraSources[0]?.id ??
    ''
  const [settings, setSettings] = useState(initial)
  const [selectedSourceId, setSelectedSourceId] = useState(defaultSourceId)
  const selectedSource = settings.cameraSources.find((s) => s.id === selectedSourceId)

  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState<'disconnected' | 'connecting' | 'connected'>(
    'disconnected',
  )
  const [lastMsg, setLastMsg] = useState<StreamFrameMessage | null>(null)
  const [error, setError] = useState<string | null>(null)

  const socketRef = useRef<Socket | null>(null)

  useEffect(() => {
    return () => {
      socketRef.current?.disconnect()
      socketRef.current = null
    }
  }, [])

  useEffect(() => {
    function refresh() {
      const next = loadSettings()
      setSettings(next)
      const hasSelected = next.cameraSources.some((s) => s.id === selectedSourceId)
      if (!hasSelected) {
        const nextDefault =
          next.cameraSources.find((s) => s.url.trim() === '0')?.id ??
          next.cameraSources[0]?.id ??
          ''
        setSelectedSourceId(nextDefault)
      }
    }
    window.addEventListener('storage', refresh)
    const id = window.setInterval(refresh, 1000)
    return () => {
      window.removeEventListener('storage', refresh)
      window.clearInterval(id)
    }
  }, [selectedSourceId])

  function start() {
    if (running) return
    const sourceUrl = selectedSource?.url?.trim() ?? ''
    if (!sourceUrl) {
      setError('Source caméra vide. Choisis "Webcam PC (0)" ou renseigne une URL/ID.')
      return
    }
    setError(null)
    setStatus('connecting')

    const streamBase = ENV.streamUrl.replace(/\/+$/, '')
    const socketPath =
      ENV.streamSocketPath.trim().length > 0
        ? ENV.streamSocketPath.trim()
        : streamBase.endsWith('/stream') || streamBase.endsWith('/stream/')
          ? '/stream/socket.io'
          : '/socket.io'

    const socket = io(streamBase, {
      path: socketPath,
      transports: ['websocket'],
      autoConnect: true,
    })
    socketRef.current = socket

    socket.on('connect', () => {
      setStatus('connected')
      setRunning(true)
      socket.emit('start', {
        source: sourceUrl,
        threshold: settings.confidenceThreshold,
      })
    })

    socket.on('disconnect', () => {
      setStatus('disconnected')
      setRunning(false)
    })

    socket.on('connect_error', (e) => {
      setError(e?.message ?? 'WebSocket connection error')
      setStatus('disconnected')
      setRunning(false)
    })

    socket.on('frame', (msg: StreamFrameMessage) => {
      setLastMsg(msg)
    })

    socket.on('status', (msg: { running?: boolean; error?: string } | undefined) => {
      if (msg?.error) setError(msg.error)
    })
  }

  function stop() {
    const socket = socketRef.current
    if (!socket) return
    try {
      socket.emit('stop')
    } catch {
      // ignore
    }
    socket.disconnect()
    socketRef.current = null
    setRunning(false)
    setStatus('disconnected')
  }

  const label = lastMsg?.label ?? '—'
  const confidencePct =
    lastMsg?.confidence != null ? Math.round(lastMsg.confidence * 100) : null
  const isViolent = String(label).toLowerCase().includes('viol')

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="bg-gradient-to-r from-cyan-600 via-blue-600 to-indigo-600 bg-clip-text text-xl font-bold text-transparent">
            Live Stream
          </h1>
          <p className="mt-1 text-sm text-slate-600">
            Connexion WebSocket au serveur de stream (OpenCV) avec overlay en temps réel.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge
            variant={
              status === 'connected'
                ? 'success'
                : status === 'connecting'
                  ? 'warning'
                  : 'neutral'
            }
          >
            {status}
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[360px,1fr]">
        <Card>
          <CardHeader>
            <div className="text-sm font-semibold text-slate-900">Contrôles</div>
          </CardHeader>
          <CardBody className="space-y-4">
            <div>
              <div className="mb-1 text-xs text-slate-600">Source caméra</div>
              <Select
                value={selectedSourceId}
                onChange={(e) => setSelectedSourceId(e.target.value)}
                disabled={running}
              >
                {settings.cameraSources.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </Select>
              <div className="mt-2 text-xs text-slate-600">
                URL/ID: <span className="font-semibold text-indigo-700">{selectedSource?.url ?? '—'}</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <Button
                onClick={start}
                disabled={running || !(selectedSource?.url?.trim()?.length)}
              >
                Start
              </Button>
              <Button onClick={stop} disabled={!running} variant="secondary">
                Stop
              </Button>
            </div>

            <div className="rounded-xl border border-indigo-200 bg-gradient-to-br from-indigo-500/10 to-cyan-500/10 p-3">
              <div className="text-xs text-slate-600">Dernière prédiction</div>
              <div className="mt-2 flex items-center justify-between">
                <Badge variant={isViolent ? 'danger' : 'success'}>{label}</Badge>
                <div className="text-sm font-semibold text-slate-900">
                  {confidencePct != null ? `${confidencePct}%` : '—'}
                </div>
              </div>
              <div className="mt-2 text-xs text-slate-600">
                Seuil: {Math.round(settings.confidenceThreshold * 100)}%
              </div>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white/70 p-3 text-xs text-slate-600">
              Events attendus côté stream server:
              <div className="mt-1 text-slate-700">
                - emit <span className="font-mono">start</span> {'{ source, threshold }'}
              </div>
              <div className="text-slate-700">
                - on <span className="font-mono">frame</span> {'{ frame, label, confidence }'}
              </div>
            </div>

            {error ? (
              <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-200">
                {error}
              </div>
            ) : null}
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold text-slate-900">Stream</div>
              <div className="text-xs text-slate-600">
                {ENV.streamUrl.replace(/^https?:\/\//, '')}
              </div>
            </div>
          </CardHeader>
          <CardBody>
            <div className="relative overflow-hidden rounded-2xl border border-slate-200 bg-black">
              {lastMsg?.frame ? (
                <>
                  <img
                    src={asImageSrc(lastMsg.frame)}
                    alt="Live frame"
                    className="block w-full"
                  />
                  <div className="absolute left-3 top-3 flex items-center gap-2 rounded-xl bg-slate-950/70 px-3 py-2 text-xs text-slate-100 ring-1 ring-white/20 backdrop-blur">
                    <Badge variant={isViolent ? 'danger' : 'success'}>{label}</Badge>
                    <span className="font-semibold">
                      {confidencePct != null ? `${confidencePct}%` : '—'}
                    </span>
                  </div>
                </>
              ) : (
                <div className="flex aspect-video items-center justify-center p-10 text-center text-sm text-slate-300">
                  Démarre le stream pour afficher les frames.
                </div>
              )}
            </div>
          </CardBody>
        </Card>
      </div>
    </div>
  )
}

