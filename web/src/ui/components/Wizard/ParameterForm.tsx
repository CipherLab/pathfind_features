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
        <ArtifactPicker
          label="Input data"
          value={p.inputData}
          onChange={p.setInputData}
          includeExts={[".parquet"]}
          standardOptions={["v5.0/train.parquet", "v5.0/validation.parquet", "v5.0/live.parquet"]}
        />
        <ArtifactPicker
          label="Features JSON"
          value={p.featuresJson}
          onChange={p.setFeaturesJson}
          includeExts={[".json"]}
          standardOptions={["v5.0/features.json"]}
          filterName={(n)=> /(^|\/)features\.json$/i.test(n) || /features\.json$/i.test(n)}
        />
        <label className="flex flex-col gap-2">Run name<input className="input" value={p.runName} onChange={e=>p.setRunName(e.target.value)} /></label>
        <label className="flex flex-col gap-2">Max new<input className="input" type="number" value={p.maxNew} onChange={e=>p.setMaxNew(parseInt(e.target.value||'0',10))} /></label>
        <label className="flex items-center gap-2"><input type="checkbox" checked={p.disablePF} onChange={e=>p.setDisablePF(e.target.checked)} /> Disable pathfinding</label>
        <label className="flex items-center gap-2"><input type="checkbox" checked={p.pretty} onChange={e=>p.setPretty(e.target.checked)} /> Pretty</label>
        <label className="flex items-center gap-2"><input type="checkbox" checked={p.smoke} onChange={e=>p.setSmoke(e.target.checked)} /> Smoke mode</label>
        <label className="flex flex-col gap-2">Eras<input className="input" type="number" value={p.smokeEras} onChange={e=>p.setSmokeEras(parseInt(e.target.value||'0',10))} /></label>
        <label className="flex flex-col gap-2">Rows<input className="input" type="number" value={p.smokeRows} onChange={e=>p.setSmokeRows(parseInt(e.target.value||'0',10))} /></label>
        <label className="flex flex-col gap-2">Feat<input className="input" type="number" value={p.smokeFeat} onChange={e=>p.setSmokeFeat(parseInt(e.target.value||'0',10))} /></label>
        <label className="flex flex-col gap-2">Seed<input className="input" type="number" value={p.seed} onChange={e=>p.setSeed(parseInt(e.target.value||'0',10))} /></label>
      </div>
    </div>
  )
}
