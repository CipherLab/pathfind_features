export const API_BASE = (import.meta as any).env?.VITE_API_URL || 'http://127.0.0.1:8000'

export async function jget<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`)
  if (!r.ok) throw new Error(r.statusText)
  return r.json() as Promise<T>
}

export async function jpost(path: string, body: any) {
  const r = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(r.statusText)
  return r.json()
}
