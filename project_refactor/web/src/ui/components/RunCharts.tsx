import * as React from 'react'
import { useEffect, useMemo, useState } from 'react'
import { API_BASE } from '../lib/api'

type Item = { name: string }

function parsePerf(txt: string){
  const lines = (txt||'').split(/\r?\n/)
  const metrics: Record<string, number> = {}
  for (const ln of lines){
    const m = ln.match(/(sharpe|cor|corr|ic|spearman|pearson|rmse|mae)\s*[:=]\s*([-+]?[0-9]*\.?[0-9]+)/i)
    if (m){
      const key = m[1].toLowerCase()
      const val = parseFloat(m[2])
      if (!Number.isNaN(val)) metrics[key] = val
    }
  }
  return metrics
}

export default function RunCharts(){
  const [items, setItems] = useState<Item[]>([])
  const [sel, setSel] = useState(0)
  const [perf, setPerf] = useState('')

  useEffect(()=>{
    let ignore=false
    const load = async()=>{
      try{ const res = await fetch(`${API_BASE}/runs/list-fs`); if(!res.ok) return; const data = await res.json(); if(!ignore) setItems(Array.isArray(data)? data: []) }catch{}
    }
    load()
    const id = setInterval(load, 5000)
    return ()=>{ ignore=true; clearInterval(id) }
  }, [])

  useEffect(()=>{
    let ignore=false
    const it = items[sel]
    if (!it) { setPerf(''); return }
    const load = async()=>{
      try{ const r = await fetch(`${API_BASE}/runs/${encodeURIComponent((it as any).name)}/performance`); if(!r.ok) { if(!ignore) setPerf(''); return } const j = await r.json(); if(!ignore) setPerf(j.content||'') }catch{}
    }
    load()
    const id = setInterval(load, 7000)
    return ()=>{ ignore=true; clearInterval(id) }
  }, [items, sel])

  const metrics = useMemo(()=> parsePerf(perf), [perf])

  return (
    <div className="card">
      <div className="bold">Recent Performance</div>
      <div className="row mt8">
        <label>Run
          <select value={String(sel)} onChange={e=> setSel(parseInt(e.target.value,10))}>
            {items.map((it, i)=> <option key={(it as any).name} value={i}>{(it as any).name}</option>)}
          </select>
        </label>
      </div>
      <div className="mt8">
        {Object.keys(metrics).length? (
          <table className="table"><thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
              {Object.entries(metrics).map(([k,v])=> (
                <tr key={k}><td>{k}</td><td>{v}</td></tr>
              ))}
            </tbody>
          </table>
        ): <div className="muted">No performance metrics found for this run.</div>}
      </div>
    </div>
  )
}
