import * as React from 'react'
import { useMemo, useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import StageSelector from '../components/Wizard/StageSelector'
import ParameterForm from '../components/Wizard/ParameterForm'
import CommandPreview from '../components/Wizard/CommandPreview'
import { API_BASE, jpost, jget } from '../lib/api'
import ActiveRuns from '../components/ActiveRuns'

export default function WizardPage(){
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

  const payload = useMemo(()=>({
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

  async function submit(){
    setBusy(true)
    setMsg('')
    try{
      // Quick client-side validation: features_json must look like a features.json, not a new-features list
      if (!/\bfeatures\.json$/i.test(featuresJson)) {
        throw new Error('Features JSON must be a features.json file. Tip: pick from the dropdown suggestions.')
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
            <StageSelector previousRuns={previous} value={stage1FromRun} onChange={setStage1FromRun} label="Stage 1 from run" />
            <StageSelector previousRuns={previous} value={stage2FromRun} onChange={setStage2FromRun} label="Stage 2 from run" />
          </div>
        </div>
      )}

      {step===2 && (
        <ParameterForm
          inputData={inputData} setInputData={setInputData}
          featuresJson={featuresJson} setFeaturesJson={setFeaturesJson}
          runName={runName} setRunName={setRunName}
          maxNew={maxNew} setMaxNew={setMaxNew}
          disablePF={disablePF} setDisablePF={setDisablePF}
          pretty={pretty} setPretty={setPretty}
          smoke={smoke} setSmoke={setSmoke}
          smokeEras={smokeEras} setSmokeEras={setSmokeEras}
          smokeRows={smokeRows} setSmokeRows={setSmokeRows}
          smokeFeat={smokeFeat} setSmokeFeat={setSmokeFeat}
          seed={seed} setSeed={setSeed}
        />
      )}

      {step===3 && (
        <>
          <CommandPreview cmd={`./.venv/bin/python run_pipeline.py run --input-data ${inputData} --features-json ${featuresJson} --run-name ${runName}${stage1FromRun? ` --stage1-from pipeline_runs/${stage1FromRun}`:''}${stage2FromRun? ` --stage2-from pipeline_runs/${stage2FromRun}`:''} --max-new-features ${maxNew}${disablePF? ' --disable-pathfinding':''}${pretty? ' --pretty':''}${smoke? ` --smoke-mode --smoke-max-eras ${smokeEras} --smoke-row-limit ${smokeRows} --smoke-feature-limit ${smokeFeat}`:''} --seed ${seed}`} />
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
