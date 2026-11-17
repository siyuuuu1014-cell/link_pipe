import axios from 'axios'

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api/v1',
  timeout: 600_000,
})

export function absoluteAsset(url?: string | null) {
  if (!url) return ''
  if (/^https?:/.test(url)) return url
  const apiBase = import.meta.env.VITE_API_BASE_URL as string | undefined
  if (apiBase?.startsWith('http')) return `${new URL(apiBase).origin}${url}`
  return url
}
