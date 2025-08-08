import * as React from 'react'
import { useEffect, useMemo, useState } from 'react'
import { API_BASE } from '../lib/api'

type Active = {
  id: string
  status: 'PENDING'|'RUNNING'|'SUCCESS'|'ERROR'|'CANCELLED'
  created_at: number
  params?: { run_name?: string }
  run_dir?: string
}

export default function ActiveRuns({ onHasRunning }: { onHasRunning?: (v:boolean)=>void }){
  const [items, setItems] = useState<Active[]>([])
  useEffect(()=>{
    let ignore=false
    const load = async()=>{
      try {
        const res = await fetch(`${API_BASE}/runs`)
        if (!res.ok) return
        const data: Active[] = await res.json()
        if (!ignore) setItems(Array.isArray(data)? data: [])
      } catch {}
    }
    load()
    const id = setInterval(load, 2000)
    return ()=>{ ignore=true; clearInterval(id) }
  }, [])

  const running = useMemo(()=> items.some(i=> i.status==='RUNNING' || i.status==='PENDING'), [items])
  useEffect(()=>{ onHasRunning?.(running) }, [running, onHasRunning])

  if (!items.length) {
    return (
      <div className="card">
        <div className="bold">Active Runs</div>
        <div className="muted">No active runs</div>
      </div>
    )
  }
  return (
    <div className="card">
      <div className="bold">Active Runs</div>
      <table className="table mt8">
        <thead><tr><th>Status</th><th>Run Name</th><th>Started</th><th>Dir</th></tr></thead>
        <tbody>
          {items
            .slice()
            .sort((a,b)=> (a.created_at||0) - (b.created_at||0))
            .map(it=> (
              <tr key={it.id}>
                <td>
                  <span className={`badge ${it.status.toLowerCase()}`}>
                    {it.status}
                  </span>
                </td>
                <td>{it.params?.run_name || ''}</td>
                <td>{it.created_at ? new Date(it.created_at*1000).toLocaleString(): ''}</td>
                <td title={it.run_dir || ''} className="ellipsis">{it.run_dir || ''}</td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  )
}
