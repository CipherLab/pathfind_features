import * as React from 'react'

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
    <div className="card">
      <div className="row">
        <label>Input data<input value={p.inputData} onChange={e=>p.setInputData(e.target.value)} /></label>
        <label>Features JSON<input value={p.featuresJson} onChange={e=>p.setFeaturesJson(e.target.value)} /></label>
        <label>Run name<input value={p.runName} onChange={e=>p.setRunName(e.target.value)} /></label>
      </div>
      <div className="row mt8">
        <label>Max new<input type="number" value={p.maxNew} onChange={e=>p.setMaxNew(parseInt(e.target.value||'0',10))} /></label>
        <label><input type="checkbox" checked={p.disablePF} onChange={e=>p.setDisablePF(e.target.checked)} /> Disable pathfinding</label>
        <label><input type="checkbox" checked={p.pretty} onChange={e=>p.setPretty(e.target.checked)} /> Pretty</label>
      </div>
      <div className="row mt8">
        <label><input type="checkbox" checked={p.smoke} onChange={e=>p.setSmoke(e.target.checked)} /> Smoke mode</label>
        <label>Eras<input type="number" value={p.smokeEras} onChange={e=>p.setSmokeEras(parseInt(e.target.value||'0',10))} /></label>
        <label>Rows<input type="number" value={p.smokeRows} onChange={e=>p.setSmokeRows(parseInt(e.target.value||'0',10))} /></label>
        <label>Feat<input type="number" value={p.smokeFeat} onChange={e=>p.setSmokeFeat(parseInt(e.target.value||'0',10))} /></label>
      </div>
      <div className="row mt8">
        <label>Seed<input type="number" value={p.seed} onChange={e=>p.setSeed(parseInt(e.target.value||'0',10))} /></label>
      </div>
    </div>
  )
}
