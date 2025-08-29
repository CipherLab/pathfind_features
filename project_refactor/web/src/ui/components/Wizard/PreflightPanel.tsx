import * as React from 'react'
import { useEffect, useState } from 'react'
import { jpost } from '../../lib/api'

type Props = {
  input_data: string
  features_json: string
  stage1_from?: string
  stage2_from?: string
  skip_walk_forward?: boolean
  max_new_features: number
  disable_pathfinding?: boolean
  smoke_mode?: boolean
  smoke_max_eras?: number
  smoke_row_limit?: number
  smoke_feature_limit?: number
  seed?: number
}

type Issue = { severity: 'info'|'warning'|'error'; field?: string; message: string }

type Result = {
  valid: boolean
  issues: Issue[]
  dataset: any
  features?: { exists:boolean; kind:string; valid_for_pipeline:boolean; medium_count?: number }
  features_diff?: { baseline_exists: boolean; added: string[]; removed: string[] }
  estimates: {
    effective_rows: number
    features_considered: number
    stage_minutes: { stage1:number; stage2:number; stage3:number }
    total_minutes: number
    memory_gb: number
    disk_gb: number
  }
}

export default function PreflightPanel(props: Props){
  const [result, setResult] = useState<Result | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string|undefined>()

  async function run(){
    setBusy(true); setErr(undefined)
    try{
      const res = await jpost('/preflight', props as any)
      setResult(res as Result)
    }catch(e:any){ setErr(e?.message || 'Preflight failed') }
    finally{ setBusy(false) }
  }

  useEffect(()=>{ run() }, [
    props.input_data, props.features_json, props.stage1_from, props.stage2_from,
    props.skip_walk_forward, props.max_new_features, props.disable_pathfinding,
    props.smoke_mode, props.smoke_max_eras, props.smoke_row_limit, props.smoke_feature_limit,
  ])

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900 p-4">
      <div className="mb-2 text-sm font-semibold text-slate-200">Pre-flight validation & estimates</div>
      <div className="mb-2 text-xs text-slate-400">Validates inputs and provides rough runtime and resource estimates based on your parameters.</div>
      <button className="btn btn-sm" onClick={run} disabled={busy}>{busy? 'Checking…':'Re-run check'}</button>
      {err && <div className="mt-2 text-xs text-red-500">{err}</div>}
      {result && (
        <div className="mt-3 space-y-3">
          <div className="text-xs">
            <div className="font-medium text-slate-200">Dataset</div>
            <div className="text-slate-300">rows≈{result.dataset?.rows ?? '—'}, columns={result.dataset?.columns ?? '—'}, features≈{result.dataset?.num_features ?? '—'}, targets≈{result.dataset?.num_targets ?? '—'}, era_col={String(result.dataset?.has_era)}</div>
          </div>
          <div className="text-xs">
            <div className="font-medium text-slate-200">Estimates</div>
            <div className="text-slate-300">total≈{result.estimates.total_minutes} min (stage1 {result.estimates.stage_minutes.stage1}m, stage2 {result.estimates.stage_minutes.stage2}m, stage3 {result.estimates.stage_minutes.stage3}m)</div>
            <div className="text-slate-300">memory≈{result.estimates.memory_gb} GB, disk≈{result.estimates.disk_gb} GB</div>
          </div>
          <div className="text-xs">
            <div className="font-medium text-slate-200">Issues</div>
            <ul className="mt-1 space-y-1">
              {result.issues.map((i,idx)=> (
                <li key={idx} className={`rounded border px-2 py-1 ${i.severity==='error'? 'border-red-700 bg-red-900/30 text-red-200': i.severity==='warning'? 'border-yellow-700 bg-yellow-900/30 text-yellow-200':'border-slate-700 bg-slate-800 text-slate-200'}`}>
                  <span className="uppercase text-[10px] font-bold mr-2">{i.severity}</span>
                  {i.message}
                </li>
              ))}
            </ul>
          </div>
          <div className="text-xs">
            <div className="font-medium text-slate-200">Features JSON</div>
            <div className="text-slate-300">exists={String(result.features?.exists)} kind={result.features?.kind} medium_count={result.features?.medium_count ?? '—'} valid={String(result.features?.valid_for_pipeline)}</div>
            {result.features_diff?.baseline_exists && (
              <div className="mt-1 grid grid-cols-1 gap-2 md:grid-cols-2">
                <div>
                  <div className="font-semibold text-slate-300">Added vs baseline</div>
                  <ul className="mt-1 max-h-28 overflow-auto rounded border border-slate-700 bg-slate-800 p-2">
                    {(result.features_diff.added||[]).slice(0,50).map((n)=> <li key={n}>{n}</li>)}
                    {(!result.features_diff.added || result.features_diff.added.length===0) && <li className="text-slate-500">(none)</li>}
                  </ul>
                </div>
                <div>
                  <div className="font-semibold text-slate-300">Removed vs baseline</div>
                  <ul className="mt-1 max-h-28 overflow-auto rounded border border-slate-700 bg-slate-800 p-2">
                    {(result.features_diff.removed||[]).slice(0,50).map((n)=> <li key={n}>{n}</li>)}
                    {(!result.features_diff.removed || result.features_diff.removed.length===0) && <li className="text-slate-500">(none)</li>}
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
