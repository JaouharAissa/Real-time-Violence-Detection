function readEnv(key: string, fallback: string) {
  const value = import.meta.env[key]
  return typeof value === 'string' && value.length > 0 ? value : fallback
}

export const ENV = {
  apiBaseUrl: readEnv('VITE_API_BASE_URL', 'http://localhost:8000'),
  streamUrl: readEnv('VITE_STREAM_URL', 'http://localhost:8001'),
  // Optional override for Socket.IO path (must match server `SOCKETIO_PATH`).
  // Example prod behind ingress: /stream/socket.io
  streamSocketPath: readEnv('VITE_SOCKETIO_PATH', ''),
}

