import * as React from 'react'
import { useEffect, useMemo, useState } from 'react'
import { jget, jpost } from '../../lib/api'

export type Mode = 'parquet' | 'features'

type RunFS = { name: string }
 type Artifact = { name: string; size?: number; modified?: number }

 type Row = {
  path: string
  run: string
  name: string
  size?: number
  modified?: number
  kind: 'parquet' | 'features.json' | 'other'
  // Metrics populated by /inspect
  rows?: number | null
  columns?: number | null
  num_features?: number | null
  num_targets?: number | null
  has_era?: boolean | null
  features_medium_count?: number | null
  features_valid?: boolean | null
  added_count?: number | null
  removed_count?: number | null
}

type Props = {
  mode: Mode
  onSelect: (value: string)=> void
  onClose: ()=> void
}

export default function GlobalPickerModal({ mode, onSelect, onClose }: Props){
  const [rows, setRows] = useState<Row[]>([])
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string|undefined>()
  const [sortKey, setSortKey] = useState<string>(mode==='parquet'? 'rows':'features_medium_count')
  const [sortDir, setSortDir] = useState<'asc'|'desc'>('desc')

  useEffect(()=>{
    let ignore=false
    async function load(){
      setLoading(true); setErr(undefined)
      try{
        const runs = await jget<RunFS[]>(`/runs/list-fs`)
        const out: Row[] = []
        for (const r of runs||[]){
          try{
            const arts = await jget<Artifact[]>(`/runs/${encodeURIComponent(r.name)}/artifacts`)
            for (const a of arts||[]){
              const lower = (a.name||'').toLowerCase()
              const isParquet = lower.endsWith('.parquet')
              const isFeatures = a.name === 'features.json' // exact filename only
              const kind = isParquet? 'parquet' : (isFeatures? 'features.json':'other')
              if (mode==='parquet' && !isParquet) continue
              if (mode==='features' && !isFeatures) continue
              out.push({
                path: `pipeline_runs/${r.name}/${a.name}`,
                run: r.name,
                name: a.name,
                size: a.size,
                modified: a.modified,
                kind,
              })
            }
          } catch {}
        }
        // Also include baseline v5.0 defaults at top
        if (mode==='features'){
          out.unshift({ path: 'v5.0/features.json', run: 'v5.0', name: 'features.json', kind: 'features.json' })
        } else if (mode==='parquet'){
          out.unshift({ path: 'v5.0/train.parquet', run: 'v5.0', name: 'train.parquet', kind: 'parquet' })
          out.unshift({ path: 'v5.0/validation.parquet', run: 'v5.0', name: 'validation.parquet', kind: 'parquet' })
          out.unshift({ path: 'v5.0/live.parquet', run: 'v5.0', name: 'live.parquet', kind: 'parquet' })
        }
        // Inspect sequentially to avoid overloading server
        const enriched: Row[] = []
        for (const row of out){
          try{
            const info = await jpost('/inspect', { path: row.path })
            if (info.type==='parquet'){
              enriched.push({
                ...row,
                rows: info.rows ?? null,
                columns: info.columns ?? null,
                num_features: info.num_features ?? null,
                num_targets: info.num_targets ?? null,
                has_era: info.has_era ?? null,
              })
            } else if (info.type==='features.json'){
              enriched.push({
                ...row,
                features_valid: info.features?.valid_for_pipeline ?? null,
                features_medium_count: info.features?.medium_count ?? null,
                added_count: (info.features_diff?.added||[]).length,
                removed_count: (info.features_diff?.removed||[]).length,
              })
            } else {
              enriched.push(row)
            }
          } catch {
            enriched.push(row)
          }
        }
        if (!ignore) setRows(enriched)
      } catch(e:any){ if (!ignore) setErr(e?.message || 'Failed to load') }
      finally{ if (!ignore) setLoading(false) }
    }
    load(); return ()=>{ ignore=true }
  }, [mode])

  const sorted = useMemo(()=>{
    const r = [...rows]
    r.sort((a,b)=>{
      const av = (a as any)[sortKey]
      const bv = (b as any)[sortKey]
      const ax = (av===undefined || av===null)? -Infinity : av
      const bx = (bv===undefined || bv===null)? -Infinity : bv
      const cmp = ax>bx? 1 : ax<bx? -1 : 0
      return sortDir==='asc'? cmp : -cmp
    })
    return r
  }, [rows, sortKey, sortDir])

  const topPaths = useMemo(()=> new Set(sorted.slice(0,3).map(r=> r.path)), [sorted])

  function toggleSort(key:string){
    if (sortKey===key) setSortDir(d=> d==='asc'? 'desc':'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="max-h-[85vh] w-[90vw] max-w-5xl overflow-hidden rounded-lg border border-slate-700 bg-slate-900">
        <div className="flex items-center justify-between border-b border-slate-700 p-3">
          <div className="text-sm font-semibold text-slate-100">Select {mode==='parquet'? 'data parquet':'features.json'}</div>
          <button className="btn btn-sm" onClick={onClose}>Close</button>
        </div>
        <div className="p-3 text-xs text-slate-400">Showing valid options across all runs. Click a row to select. Click headers to sort.</div>
        {err && <div className="px-3 pb-2 text-xs text-red-500">{err}</div>}
        <div className="max-h-[65vh] overflow-auto p-3">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 bg-slate-900">
              <tr>
                <th className="p-2">Select</th>
                <th className="cursor-pointer p-2" onClick={()=>toggleSort('run')}>Run</th>
                <th className="cursor-pointer p-2" onClick={()=>toggleSort('name')}>Name</th>
                <th className="cursor-pointer p-2" onClick={()=>toggleSort('rows')}>Rows</th>
                <th className="cursor-pointer p-2" onClick={()=>toggleSort('num_features')}>#Feat</th>
                <th className="cursor-pointer p-2" onClick={()=>toggleSort('features_medium_count')}>feat.medium</th>
                <th className="cursor-pointer p-2" onClick={()=>toggleSort('added_count')}>+Δ</th>
                <th className="cursor-pointer p-2" onClick={()=>toggleSort('removed_count')}>-Δ</th>
                <th className="p-2">Era</th>
              </tr>
            </thead>
            <tbody>
              {loading && (
                <tr><td colSpan={9} className="p-3 text-center text-slate-400">Loading…</td></tr>
              )}
              {!loading && sorted.map((r)=> (
                <tr key={r.path} className={`border-b border-slate-800 ${topPaths.has(r.path)? 'bg-blue-900/20':''}`}>
                  <td className="p-2"><button className="btn btn-xs" onClick={()=> { onSelect(r.path); onClose() }}>Select</button></td>
                  <td className="p-2 text-slate-300">{r.run}</td>
                  <td className="p-2 text-slate-300">
                    <span className="inline-block max-w-[4em] truncate align-baseline" title={r.path}>{r.path}</span>
                  </td>
                  <td className="p-2 text-slate-300">{r.rows ?? ''}</td>
                  <td className="p-2 text-slate-300">{r.num_features ?? ''}</td>
                  <td className="p-2 text-slate-300">{r.features_medium_count ?? ''}</td>
                  <td className="p-2 text-slate-300">{r.added_count ?? ''}</td>
                  <td className="p-2 text-slate-300">{r.removed_count ?? ''}</td>
                  <td className="p-2 text-slate-300">{r.has_era===null || r.has_era===undefined? '': String(r.has_era)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
