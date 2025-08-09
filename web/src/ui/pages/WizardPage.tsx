import * as React from 'react'
import { useMemo, useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import StageSelector from '../components/Wizard/StageSelector'
import ParameterForm from '../components/Wizard/ParameterForm'
import CommandPreview from '../components/Wizard/CommandPreview'
import { API_BASE, jpost, jget } from '../lib/api'
import ActiveRuns from '../components/ActiveRuns'
import ParameterHelp from '../components/Wizard/ParameterHelp'
import PreflightPanel from '../components/Wizard/PreflightPanel'

export default function WizardPage(){
  // Guard: keep run name within a reasonable length for folders and UI
  const MAX_RUN_NAME_LENGTH = 64
  const [inputData, setInputData] = useState('v5.0/train.parquet')
  const [featuresJson, setFeaturesJson] = useState('v5.0/features.json')
  const [runName, setRunName] = useState('wizard')
  const [maxNew, setMaxNew] = useState(8)
  const [disablePF, setDisablePF] = useState(false)
  const [pretty, setPretty] = useState(true)
  const [smoke, setSmoke] = useState(true)
  const [smokeEras, setSmokeEras] = useState(60)
  const [smokeRows, setSmokeRows] = useState(150000)
  const [smokeFeat, setSmokeFeat] = useState(300)
  const [seed, setSeed] = useState(42)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')
  const nav = useNavigate()
  const [previous, setPrevious] = useState<string[]>([])
  const [reuse, setReuse] = useState('')
  const [stage1FromRun, setStage1FromRun] = useState('')
  const [stage2FromRun, setStage2FromRun] = useState('')
  const [hasRunning, setHasRunning] = useState(false)
  const [step, setStep] = useState(1)
  // Phase mode: full pipeline vs focused phases
  type PhaseMode = 'full' | 'phase1' | 'phase2' | 'phase3'
  const [mode, setMode] = useState<PhaseMode>('full')
  useEffect(()=>{ (async()=>{ try{ const list = await jget<any[]>(`/runs/list-fs`); setPrevious(list.map(x=> x.name)) }catch{} })() }, [])

  // When a run is selected for reuse, try to infer artifacts and map to parameters
  useEffect(()=>{
    if (!reuse) return
    let ignore=false
    ;(async()=>{
      try{
        const arts = await jget<{name:string}[]>(`/runs/${encodeURIComponent(reuse)}/artifacts`)
        const names = new Set((arts||[]).map(a=> a.name))
        // Heuristics
        // Prefer a features.json in the run dir
        const fjson = [...names].find(n=> n.endsWith('features.json'))
        if (!ignore && fjson) setFeaturesJson(`pipeline_runs/${reuse}/${fjson}`)
        // Prefer train.parquet in v5.0/, else any *.parquet in run dir
        const train = [...names].find(n=> /train\.parquet$/i.test(n))
        const anypq = [...names].find(n=> n.endsWith('.parquet'))
        if (!ignore) setInputData(train? `pipeline_runs/${reuse}/${train}` : (anypq? `pipeline_runs/${reuse}/${anypq}`: inputData))
        // Suggest a new run name based on reuse
        if (!ignore) setRunName(`${reuse}_re${Math.floor(Math.random()*1000)}`.slice(0,MAX_RUN_NAME_LENGTH))
        if (!ignore) { setStage1FromRun(reuse); setStage2FromRun(reuse) }
      }catch{}
    })()
    return ()=>{ ignore=true }
  }, [reuse])

  // Base payload from current form values
  const basePayload = useMemo(()=>({
    input_data: inputData,
  features_json: featuresJson,
    run_name: runName,
    stage1_from: stage1FromRun? `pipeline_runs/${stage1FromRun}`: undefined,
    stage2_from: stage2FromRun? `pipeline_runs/${stage2FromRun}`: undefined,
    max_new_features: maxNew,
    disable_pathfinding: disablePF,
    pretty,
    smoke_mode: smoke,
    smoke_max_eras: smoke? smokeEras: undefined,
    smoke_row_limit: smoke? smokeRows: undefined,
    smoke_feature_limit: smoke? smokeFeat: undefined,
    seed,
  }), [inputData, featuresJson, runName, stage1FromRun, stage2FromRun, maxNew, disablePF, pretty, smoke, smokeEras, smokeRows, smokeFeat, seed])

  // Adjust payload based on phase-specific rules
  const payload = useMemo(()=>{
    const p: any = { ...basePayload }
    if (mode === 'phase1'){
      // Stage 1 only
      p.disable_pathfinding = true
    } else if (mode === 'phase2'){
      // Must reuse Stage 1, and skip feature engineering
      p.disable_pathfinding = false
      p.max_new_features = 0
      p.stage2_from = undefined
    } else if (mode === 'phase3'){
      // Must reuse Stage 1 and Stage 2
      p.disable_pathfinding = false
    }
    return p
  }, [basePayload, mode])

  // Command string preview reflecting phase mode
  const cmd = useMemo(()=>{
    const p: any = payload
    const parts: string[] = [
      `./.venv/bin/python run_pipeline.py run`,
      `--input-data ${p.input_data}`,
      `--features-json ${p.features_json}`,
      `--run-name ${p.run_name}`,
    ]
    if (p.stage1_from && mode !== 'phase1') parts.push(`--stage1-from ${p.stage1_from}`)
    if (p.stage2_from && mode === 'phase3') parts.push(`--stage2-from ${p.stage2_from}`)
    parts.push(`--max-new-features ${p.max_new_features}`)
    if (p.disable_pathfinding) parts.push(`--disable-pathfinding`)
    if (pretty) parts.push(`--pretty`)
    if (p.smoke_mode){
      parts.push(`--smoke-mode`)
      if (p.smoke_max_eras) parts.push(`--smoke-max-eras ${p.smoke_max_eras}`)
      if (p.smoke_row_limit) parts.push(`--smoke-row-limit ${p.smoke_row_limit}`)
      if (p.smoke_feature_limit) parts.push(`--smoke-feature-limit ${p.smoke_feature_limit}`)
    }
    parts.push(`--seed ${p.seed}`)
    return parts.join(' ')
  }, [payload, mode, pretty])

  async function submit(){
    setBusy(true)
    setMsg('')
    try{
      // Quick client-side validation: features_json must look like a features.json, not a new-features list
      if (!/\bfeatures\.json$/i.test(featuresJson)) {
        throw new Error('Features JSON must be a features.json file. Tip: pick from the dropdown suggestions.')
      }
      // Phase-specific lineage checks
      if (mode === 'phase2' && !stage1FromRun) {
        throw new Error('Phase 2 requires selecting a Stage 1 run to inherit from.')
      }
      if (mode === 'phase3' && (!stage1FromRun || !stage2FromRun)) {
        throw new Error('Phase 3 requires selecting both Stage 1 and Stage 2 runs to inherit from.')
      }
      await jpost(`/runs`, payload)
      setMsg('Run started')
      setTimeout(()=> nav('/'), 500)
    }catch(err:any){ setMsg('Failed: '+err.message) }
    finally{ setBusy(false) }
  }

  return (
    <div className="space-y-4">
      <Link to="/">← Back</Link>
      {/* Phase Mode Switch */}
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-slate-400">Mode:</span>
        {[
          {k:'full', label:'Full Pipeline'},
          {k:'phase1', label:'Phase 1: Target Discovery'},
          {k:'phase2', label:'Phase 2: Relationship Mining'},
          {k:'phase3', label:'Phase 3: Feature Engineering'},
        ].map(opt=> (
          <button key={opt.k}
            className={`px-3 py-1 rounded-full border ${mode===opt.k? 'bg-blue-600 text-white border-blue-500':'bg-slate-800 text-slate-200 border-slate-700'}`}
            onClick={()=> setMode(opt.k as PhaseMode)}>
            {opt.label}
          </button>
        ))}
      </div>
      <div className="mt-2"><ActiveRuns onHasRunning={setHasRunning} /></div>
      <div className="flex gap-2 text-sm">
        {['Select Run','Configure','Review'].map((t,i)=> (
          <div key={t} className={`px-3 py-1 rounded-full ${step===i+1? 'bg-blue-600 text-white':'bg-slate-700 text-slate-300'}`}>{i+1}. {t}</div>
        ))}
      </div>

      {step===1 && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap gap-4">
            <StageSelector previousRuns={previous} value={reuse} onChange={setReuse} label="Reuse from run" />
            {(mode==='phase2' || mode==='phase3' || mode==='full') && (
              <StageSelector previousRuns={previous} value={stage1FromRun} onChange={setStage1FromRun} label={`Stage 1 from run${mode==='phase2' || mode==='phase3' ? ' (required)': ''}`} />
            )}
            {(mode==='phase3' || mode==='full') && (
              <StageSelector previousRuns={previous} value={stage2FromRun} onChange={setStage2FromRun} label={`Stage 2 from run${mode==='phase3' ? ' (required)': ''}`} />
            )}
          </div>
        </div>
      )}

      {step===2 && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="space-y-4">
            <ParameterForm
              inputData={inputData} setInputData={setInputData}
              featuresJson={featuresJson} setFeaturesJson={setFeaturesJson}
              runName={runName} setRunName={setRunName}
              maxNew={mode==='phase2' ? 0 : maxNew} setMaxNew={mode==='phase2' ? ((()=>{}) as any) : setMaxNew}
              disablePF={mode==='phase1' ? true : disablePF} setDisablePF={mode==='phase1' ? ((()=>{}) as any) : setDisablePF}
              pretty={pretty} setPretty={setPretty}
              smoke={smoke} setSmoke={setSmoke}
              smokeEras={smokeEras} setSmokeEras={setSmokeEras}
              smokeRows={smokeRows} setSmokeRows={setSmokeRows}
              smokeFeat={smokeFeat} setSmokeFeat={setSmokeFeat}
              seed={seed} setSeed={setSeed}
            />
            <PreflightPanel
              input_data={payload.input_data}
              features_json={payload.features_json}
              stage1_from={payload.stage1_from}
              stage2_from={payload.stage2_from}
              skip_walk_forward={false}
              max_new_features={payload.max_new_features as number}
              disable_pathfinding={payload.disable_pathfinding as boolean}
              smoke_mode={payload.smoke_mode as boolean}
              smoke_max_eras={payload.smoke_max_eras as number | undefined}
              smoke_row_limit={payload.smoke_row_limit as number | undefined}
              smoke_feature_limit={payload.smoke_feature_limit as number | undefined}
              seed={payload.seed as number}
            />
          </div>
          <ParameterHelp setters={{
            input_data: setInputData,
            features_json: setFeaturesJson,
            run_name: setRunName,
            max_new_features: (mode==='phase2' ? undefined : (setMaxNew as any)) as any,
            disable_pathfinding: (mode==='phase1' ? undefined : (setDisablePF as any)) as any,
            pretty: setPretty as any,
            smoke_mode: setSmoke as any,
            smoke_max_eras: setSmokeEras as any,
            smoke_row_limit: setSmokeRows as any,
            smoke_feature_limit: setSmokeFeat as any,
            seed: setSeed as any,
            stage1_from: (v:string)=> setStage1FromRun(v||''),
            stage2_from: (v:string)=> setStage2FromRun(v||''),
          }} />
        </div>
      )}

      {step===3 && (
        <>
          <CommandPreview cmd={cmd} />
          <button disabled={busy || hasRunning} onClick={submit} className="btn btn-primary">{hasRunning? 'Busy (wait for active run)':'Start'}</button>
          {msg && <div className="mt-2">{msg}</div>}
        </>
      )}

      <div className="flex justify-between pt-4">
        <button className="btn" disabled={step===1} onClick={()=>setStep(s=>Math.max(1,s-1))}>Back</button>
        {step<3 && <button className="btn btn-primary" onClick={()=>setStep(s=>Math.min(3,s+1))}>Next</button>}
      </div>
    </div>
  )
}
