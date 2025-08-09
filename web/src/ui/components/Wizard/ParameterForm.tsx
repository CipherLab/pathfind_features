import * as React from 'react'
import ArtifactPicker from './ArtifactPicker'

type Props = {
  inputData: string; setInputData: (v:string)=>void
  featuresJson: string; setFeaturesJson: (v:string)=>void
  runName: string; setRunName: (v:string)=>void
  maxNew: number; setMaxNew: (v:number)=>void
  disablePF: boolean; setDisablePF: (v:boolean)=>void
  pretty: boolean; setPretty: (v:boolean)=>void
  smoke: boolean; setSmoke: (v:boolean)=>void
  smokeEras: number; setSmokeEras: (v:number)=>void
  smokeRows: number; setSmokeRows: (v:number)=>void
  smokeFeat: number; setSmokeFeat: (v:number)=>void
  seed: number; setSeed: (v:number)=>void
}

export default function ParameterForm(p: Props){
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800 p-4">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <ArtifactPicker
            label="Input data"
            value={p.inputData}
            onChange={p.setInputData}
            includeExts={[".parquet"]}
            standardOptions={["v5.0/train.parquet", "v5.0/validation.parquet", "v5.0/live.parquet"]}
          />
          <div className="text-xs text-slate-400">Parquet file containing the training rows.</div>
        </div>
        <div>
          <ArtifactPicker
            label="Features JSON"
            value={p.featuresJson}
            onChange={p.setFeaturesJson}
            includeExts={[".json"]}
            standardOptions={["v5.0/features.json"]}
            filterName={(n)=> /(^|\/)features\.json$/i.test(n) || /features\.json$/i.test(n)}
          />
          <div className="text-xs text-slate-400">Feature definition file with feature_sets.medium.</div>
        </div>
        <label className="flex flex-col gap-2">
          Run name
          <input className="input" value={p.runName} onChange={e=>p.setRunName(e.target.value)} />
          <span className="text-xs text-slate-400">Directory to save run artifacts.</span>
        </label>
        <label className="flex flex-col gap-2">
          Max new
          <input className="input" type="number" value={p.maxNew} onChange={e=>p.setMaxNew(parseInt(e.target.value||'0',10))} />
          <span className="text-xs text-slate-400">Limit on new features to generate.</span>
        </label>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={p.disablePF} onChange={e=>p.setDisablePF(e.target.checked)} /> Disable pathfinding
        </label>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={p.pretty} onChange={e=>p.setPretty(e.target.checked)} /> Pretty
        </label>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={p.smoke} onChange={e=>p.setSmoke(e.target.checked)} /> Smoke mode
        </label>
        <label className="flex flex-col gap-2">
          Eras
          <input className="input" type="number" value={p.smokeEras} onChange={e=>p.setSmokeEras(parseInt(e.target.value||'0',10))} />
          <span className="text-xs text-slate-400">Max eras when smoke mode is enabled.</span>
        </label>
        <label className="flex flex-col gap-2">
          Rows
          <input className="input" type="number" value={p.smokeRows} onChange={e=>p.setSmokeRows(parseInt(e.target.value||'0',10))} />
          <span className="text-xs text-slate-400">Row limit for smoke mode runs.</span>
        </label>
        <label className="flex flex-col gap-2">
          Feat
          <input className="input" type="number" value={p.smokeFeat} onChange={e=>p.setSmokeFeat(parseInt(e.target.value||'0',10))} />
          <span className="text-xs text-slate-400">Feature limit for smoke mode runs.</span>
        </label>
        <label className="flex flex-col gap-2">
          Seed
          <input className="input" type="number" value={p.seed} onChange={e=>p.setSeed(parseInt(e.target.value||'0',10))} />
          <span className="text-xs text-slate-400">Random seed for reproducibility.</span>
        </label>
      </div>
    </div>
  )
}
