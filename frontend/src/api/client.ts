import axios from 'axios'
import { ENV } from '../lib/env'

export const api = axios.create({
  baseURL: ENV.apiBaseUrl,
  timeout: 120_000,
})

api.interceptors.response.use(
  (res) => res,
  (error) => {
    const status = error?.response?.status
    const message =
      error?.response?.data?.detail ||
      error?.response?.data?.message ||
      error?.message ||
      'Request failed'
    return Promise.reject(Object.assign(new Error(String(message)), { status }))
  },
)

