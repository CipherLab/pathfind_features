import * as React from 'react'
import { useEffect, useMemo, useState } from 'react'
import { jget } from '../../lib/api'

type RunFS = { name: string }
type Artifact = { name: string; size?: number; modified?: number }

type Props = {
  label: string
  value: string
  onChange: (v: string) => void
  // File extensions to include, e.g. ['.parquet'] or ['.json']
  includeExts: string[]
  // Predefined options that should always show first
  standardOptions?: string[]
  // Optional additional filename filter (receives base filename)
  filterName?: (name: string) => boolean
}

function hasAllowedExt(name: string, includeExts: string[]) {
  const lower = name.toLowerCase()
  return includeExts.some(ext => lower.endsWith(ext.toLowerCase()))
}

export default function ArtifactPicker({ label, value, onChange, includeExts, standardOptions = [], filterName }: Props){
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [options, setOptions] = useState<{ label: string; value: string }[]>([])

  useEffect(() => {
    let ignore = false
    async function load() {
      setLoading(true)
      setError(null)
      try {
        const runs = await jget<RunFS[]>(`/runs/list-fs`)
        const out: { label: string; value: string }[] = []
        // Fetch artifacts for each run sequentially to avoid overloading the API
        for (const r of runs || []) {
          try {
            const arts = await jget<Artifact[]>(`/runs/${encodeURIComponent(r.name)}/artifacts`)
            for (const a of arts || []) {
              if (!a?.name) continue
              if (!hasAllowedExt(a.name, includeExts)) continue
              if (filterName && !filterName(a.name)) continue
              const full = `pipeline_runs/${r.name}/${a.name}`
              out.push({ label: `${r.name}/${a.name}`, value: full })
            }
          } catch {
            // ignore individual run errors
          }
        }
        if (!ignore) setOptions(out.sort((a, b) => a.label.localeCompare(b.label)))
      } catch (e: any) {
        if (!ignore) setError(e?.message || 'Failed to load artifacts')
      } finally {
        if (!ignore) setLoading(false)
      }
    }
    load()
    return () => { ignore = true }
  }, [includeExts.join('|')])

  const allOptions = useMemo(() => {
    const std = (standardOptions || []).map(v => ({ label: v, value: v }))
    const dedup = new Map<string, { label: string; value: string }>()
    for (const o of [...std, ...options]) {
      if (!dedup.has(o.value)) dedup.set(o.value, o)
    }
    return Array.from(dedup.values())
  }, [standardOptions, options])

  return (
    <label>
      {label}
      <div className="col">
        <select
          title={label}
          value={allOptions.find(o => o.value === value)?.value || ''}
          onChange={e => onChange(e.target.value)}
        >
          <option value="">(choose from suggestions)</option>
          {allOptions.length > 0 && (
            <optgroup label="Suggestions">
              {allOptions.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </optgroup>
          )}
        </select>
        <input
          className="input mt4"
          placeholder="Or type a custom path..."
          value={value}
          onChange={e => onChange(e.target.value)}
        />
        {loading && <div className="muted">Loading artifacts…</div>}
        {error && <div className="error">{error}</div>}
      </div>
    </label>
  )
}
