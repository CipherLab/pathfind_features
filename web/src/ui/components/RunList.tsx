import * as React from 'react'
import { Link } from 'react-router-dom'

export type Run = { name: string; status: string; started?: string; path: string }

export function RunListItem({ run }: { run: Run }){
  return (
    <tr>
  <td><span className={`badge ${run.status?.toLowerCase?.()}`}>{run.status}</span></td>
      <td><Link to={`/runs/${encodeURIComponent(run.name)}`}>{run.name}</Link></td>
      <td>{run.started || ''}</td>
    </tr>
  )
}

export default function RunList({ runs }: { runs: Run[] }){
  return (
    <table className="table">
      <thead><tr><th>Status</th><th>Name</th><th>Started</th></tr></thead>
      <tbody>
        {runs.map(r => <RunListItem key={r.path} run={r} />)}
      </tbody>
    </table>
  )
}
