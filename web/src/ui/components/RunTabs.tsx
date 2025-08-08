import * as React from 'react'
import { useEffect, useState } from 'react'
import { API_BASE } from '../lib/api'

function Tabs({tab,setTab}:{tab:number; setTab:(n:number)=>void}){
  return (
    <div className="tabs">
      {[['Performance',0],['Logs',1],['Artifacts',2]].map(([label,idx])=> (
        <button key={label as string} onClick={()=>setTab(idx as number)} className={`tab ${tab===idx? 'active':''}`}>{label as string}</button>
      ))}
    </div>
  )
}

export default function RunTabs({ name }: { name: string }){
  const [tab, setTab] = useState(0)
  const [perf, setPerf] = useState('')
  const [logs, setLogs] = useState('')
  const [arts, setArts] = useState<{name:string; size?:number; modified?:number}[]>([])

  useEffect(()=>{
    let ignore=false
    async function load(){
      try{ const p = await fetch(`${API_BASE}/runs/fs/${encodeURIComponent(name)}/performance`).then(r=> r.ok? r.json(): {content:''}); if(!ignore) setPerf(p.content||'') }catch{}
      try{ const l = await fetch(`${API_BASE}/runs/fs/${encodeURIComponent(name)}/logs`).then(r=> r.json()); if(!ignore) setLogs(l.content||'') }catch{}
      try{ const a = await fetch(`${API_BASE}/runs/fs/${encodeURIComponent(name)}/artifacts`).then(r=> r.json()); if(!ignore) setArts(a||[]) }catch{}
    }
    load()
    const id = setInterval(load, 5000)
    return ()=>{ ignore=true; clearInterval(id) }
  }, [name])

  return (
    <div>
      <Tabs tab={tab} setTab={setTab} />
      {tab===0 && (
        <div className="card mt8">
          <pre className="pre">{perf || 'No performance report yet'}</pre>
        </div>
      )}
      {tab===1 && (
        <div className="card mt8">
          <pre className="pre">{logs || 'No logs yet'}</pre>
        </div>
      )}
      {tab===2 && (
        <div className="card mt8">
          <table className="table"><thead><tr><th>Name</th><th>Size</th><th>Modified</th></tr></thead>
          <tbody>
            {arts.map(a=> <tr key={a.name}><td>{a.name}</td><td>{a.size ?? ''}</td><td>{a.modified ? new Date(a.modified*1000).toLocaleString(): ''}</td></tr>)}
          </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
