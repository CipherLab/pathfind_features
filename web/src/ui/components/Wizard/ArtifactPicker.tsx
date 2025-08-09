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

  // Whether the current value is already present in the suggestions list
  const valueInOptions = useMemo(() => {
    if (!value) return false
    return allOptions.some(o => o.value === value)
  }, [allOptions, value])

  return (
    <label className="flex flex-col gap-2">
      <span className="text-sm font-medium">{label}</span>
      <select
        className="rounded-md border border-slate-600 bg-slate-800 p-2 text-sm"
        title={label}
        value={value}
        onChange={e => onChange(e.target.value)}
      >
        <option value="">(choose from suggestions)</option>
  {!valueInOptions && value && (
          <option value={value}>{value}</option>
        )}
        {allOptions.length > 0 && (
          <optgroup label="Suggestions">
            {allOptions.map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </optgroup>
        )}
      </select>
      {loading && <div className="text-xs text-slate-400">Loading artifacts…</div>}
      {error && <div className="text-xs text-red-500">{error}</div>}
    </label>
  )
}
