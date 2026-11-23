import type { AxiosProgressEvent } from 'axios'
import { api } from './client'
import type { PredictResponse } from './types'

export async function healthCheck() {
  const res = await api.get('/health')
  return res.data as { status?: string } | unknown
}

export async function predictVideo(
  file: File,
  params?: { threshold?: number },
  onUploadProgress?: (progressPct: number) => void,
) {
  const form = new FormData()
  form.append('file', file)
  if (params?.threshold != null) form.append('threshold', String(params.threshold))

  const res = await api.post('/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (evt: AxiosProgressEvent) => {
      const total = evt.total ?? 0
      const loaded = evt.loaded ?? 0
      if (!total) return
      onUploadProgress?.(Math.round((loaded / total) * 100))
    },
  })
  return res.data as PredictResponse
}

