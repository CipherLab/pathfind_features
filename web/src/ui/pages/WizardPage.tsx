import * as React from 'react'
import { useMemo, useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import StageSelector from '../components/Wizard/StageSelector'
import ParameterForm from '../components/Wizard/ParameterForm'
import CommandPreview from '../components/Wizard/CommandPreview'
import { API_BASE, jpost, jget } from '../lib/api'

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
  useEffect(()=>{ (async()=>{ try{ const list = await jget<any[]>(`/runs/fs`); setPrevious(list.map(x=> x.name)) }catch{} })() }, [])

  const payload = useMemo(()=>({
    input_data: inputData,
    features_json: featuresJson,
    run_name: runName,
    max_new_features: maxNew,
    disable_pathfinding: disablePF,
    pretty,
    smoke_mode: smoke,
    smoke_max_eras: smoke? smokeEras: undefined,
    smoke_row_limit: smoke? smokeRows: undefined,
    smoke_feature_limit: smoke? smokeFeat: undefined,
    seed,
  }), [inputData, featuresJson, runName, maxNew, disablePF, pretty, smoke, smokeEras, smokeRows, smokeFeat, seed])

  async function submit(){
    setBusy(true)
    setMsg('')
    try{
      await jpost(`/runs`, payload)
      setMsg('Run started')
      setTimeout(()=> nav('/'), 500)
    }catch(err:any){ setMsg('Failed: '+err.message) }
    finally{ setBusy(false) }
  }

  return (
    <div>
      <Link to="/">← Back</Link>
      <div className="mt8 row">
        <StageSelector previousRuns={previous} value={''} onChange={()=>{}} label="Reuse from run" />
      </div>
      <div className="mt8">
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
      </div>
      <CommandPreview cmd={`./.venv/bin/python run_pipeline.py run --input-data ${inputData} --features-json ${featuresJson} --run-name ${runName} --max-new-features ${maxNew}${disablePF? ' --disable-pathfinding':''}${pretty? ' --pretty':''}${smoke? ` --smoke-mode --smoke-max-eras ${smokeEras} --smoke-row-limit ${smokeRows} --smoke-feature-limit ${smokeFeat}`:''} --seed ${seed}`} />
      <button disabled={busy} onClick={submit} className="mt8">Start</button>
      {msg && <div className="mt8">{msg}</div>}
    </div>
  )
}
