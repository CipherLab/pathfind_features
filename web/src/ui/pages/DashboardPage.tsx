import * as React from 'react'
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import RunList from '../components/RunList'
import RunCharts from '../components/RunCharts'
import ActiveRuns from '../components/ActiveRuns'

const API = (import.meta as any).env?.VITE_API_URL || 'http://127.0.0.1:8000'

export type FsRun = { name: string; path: string; status: string; started?: string }

export default function DashboardPage(){
  const [runs, setRuns] = useState<FsRun[]>([])
  const [q, setQ] = useState('')
  const [sort, setSort] = useState<'name'|'status'|'started'>('started')

  useEffect(()=>{
    let ignore = false
    const load = async()=>{
      const res = await fetch(`${API}/runs/list-fs`)
      const data = await res.json()
      if(!ignore) setRuns(data)
    }
    load()
    const id = setInterval(load, 4000)
    return ()=>{ ignore = true; clearInterval(id) }
  },[])

  const filtered = useMemo(()=>{
    if (!Array.isArray(runs)) return []
    const f = runs.filter(r => r.name.toLowerCase().includes(q.toLowerCase()))
    return f.sort((a,b)=>{
      if(sort==='name') return a.name.localeCompare(b.name)
      if(sort==='status') return a.status.localeCompare(b.status)
      return (b.started||'').localeCompare(a.started||'')
    })
  }, [runs,q,sort])

  return (
    <div>
  <ActiveRuns />
      <div className="row">
        <input className="input" placeholder="Search runs" value={q} onChange={e=>setQ(e.target.value)} />
        <label>
          Sort
          <select className="input" title="Sort runs" value={sort} onChange={e=>setSort(e.target.value as any)}>
          <option value="started">Start time</option>
          <option value="name">Name</option>
          <option value="status">Status</option>
          </select>
        </label>
        <Link to="/wizard"><button className="btn btn-primary">Create new run</button></Link>
      </div>
      <div className="card mt8">
        <RunList runs={filtered as any} />
      </div>
      <div className="mt8">
        <RunCharts />
      </div>
    </div>
  )
}