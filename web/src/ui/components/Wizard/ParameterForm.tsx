import * as React from 'react'
import GlobalPickerModal from './GlobalPickerModal'
import { generateExperimentName } from '../../lib/nameGenerator'

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
  const [open, setOpen] = React.useState<null | 'features' | 'parquet'>(null)
  const seedRef = React.useRef<HTMLInputElement>(null)

  // Heuristics: estimate interactions and total engineered features for preview
  const interactions = Math.round(p.maxNew * 3)
  const totalEngineered = Math.round(p.maxNew * 9)
  const zone: 'green'|'yellow'|'red' = p.maxNew <= 10 ? 'green' : (p.maxNew <= 30 ? 'yellow' : 'red')

  function generateAndSetRunName() {
    p.setRunName(generateExperimentName())
  }

  React.useEffect(() => {
    if (!p.runName) {
        generateAndSetRunName();
    }
  }, []);

  function luckyDefaults(){
    generateAndSetRunName()
    p.setMaxNew(8)
    p.setSmoke(true)
    p.setSmokeEras(60)
    p.setSmokeRows(150000)
    p.setSmokeFeat(300)
    p.setDisablePF(false)
    p.setPretty(true)
    p.setSeed(Math.floor(Math.random()*1_000_000_000))
  }

  function applyPreset(kind:'fast'|'balanced'|'thorough'){
    p.setSmoke(true)
    if (kind==='fast'){ p.setSmokeEras(30); p.setSmokeRows(100_000); p.setSmokeFeat(200); p.setMaxNew(Math.min(p.maxNew, 8)) }
    if (kind==='balanced'){ p.setSmokeEras(60); p.setSmokeRows(150_000); p.setSmokeFeat(300); }
    if (kind==='thorough'){ p.setSmokeEras(120); p.setSmokeRows(250_000); p.setSmokeFeat(600); p.setMaxNew(Math.max(p.maxNew, 12)) }
  }

  function estimateMinutes(): number {
    // Rough local estimate; Preflight panel shows authoritative numbers
    const eras = p.smoke ? Math.max(1, p.smokeEras||60) : 300
    const rows = p.smoke ? Math.max(50_000, p.smokeRows||150_000) : 3_000_000
    const feats = p.smoke ? Math.max(100, p.smokeFeat||300) : 2000
    const base = 12 // baseline minutes for fast run
    const eraFactor = eras/60
    const rowFactor = rows/150_000
    const featFactor = Math.pow(feats/300, 0.7)
    const pfFactor = p.disablePF ? 0.5 : 1.0
    const newFactor = Math.pow(Math.max(1,p.maxNew)/8, 0.4)
    return Math.round(base * eraFactor * rowFactor * featFactor * pfFactor * newFactor)
  }

  const est = estimateMinutes()
  const modeLabel = p.smoke ? 'Quick Test' : 'Full Run'

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800 p-4">
      {/* Core Settings */}
      <div className="mb-4 rounded-md border border-slate-700 bg-slate-900 p-3">
        <div className="mb-3 flex items-center justify-between">
          <div className="text-sm font-semibold text-slate-100">Core Settings</div>
          <button className="btn btn-xs" onClick={luckyDefaults} title="Pick sensible defaults">I'm Feeling Lucky</button>
        </div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <label className="flex flex-col gap-2">
            <span className="text-sm font-medium">Input data</span>
            <button
              type="button"
              onClick={()=> setOpen('parquet')}
              title={p.inputData}
              className="w-full truncate rounded-md border border-slate-600 bg-slate-800 px-3 py-2 text-left hover:border-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-600"
            >
              {p.inputData || '(select data parquet)'}
            </button>
            <div className="text-xs text-slate-400">Parquet file containing the training rows.</div>
          </label>
          <label className="flex flex-col gap-2">
            <span className="text-sm font-medium">Features JSON</span>
            <button
              type="button"
              onClick={()=> setOpen('features')}
              title={p.featuresJson}
              className="w-full truncate rounded-md border border-slate-600 bg-slate-800 px-3 py-2 text-left hover:border-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-600"
            >
              {p.featuresJson || '(select features.json)'}
            </button>
            <div className="text-xs text-slate-400">Feature definition file with feature_sets.medium.</div>
          </label>
          <label className="flex flex-col gap-2">
            <span className="text-sm font-medium">Run name</span>
            <div className="relative">
              <input
                className="input truncate w-full pr-10"
                value={p.runName}
                onChange={e=>p.setRunName(e.target.value)}
                title={p.runName}
                placeholder="Experiment name"
                aria-label="Run name"
              />
              <button
                type="button"
                className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded-md px-2 py-1 text-slate-300 hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-600"
                onClick={generateAndSetRunName}
                title="Randomize name"
                aria-label="Randomize name"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                  <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V4a1 1 0 011-1zm12 1a1 1 0 011 1v5a1 1 0 01-1 1h-5a1 1 0 010-2h2.101a5.002 5.002 0 00-9.499-1.404 1 1 0 11-1.885-.666A7.002 7.002 0 0115.899 6H18a1 1 0 011-1z" clipRule="evenodd" />
                </svg>
              </button>
            </div>
            <span className="text-xs text-slate-400">Directory to save run artifacts.</span>
          </label>
          {/* Max new slider with live preview */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium" title="Higher = more creative, longer runtime, potential overfitting">Max features</span>
              <span className={`rounded px-2 py-0.5 text-xs ${zone==='green'? 'bg-green-900 text-green-200': zone==='yellow'? 'bg-yellow-900 text-yellow-200':'bg-red-900 text-red-200'}`}>{p.maxNew}</span>
            </div>
            <input
              type="range"
              min={0}
              max={60}
              step={1}
              value={p.maxNew}
              onChange={e=> p.setMaxNew(parseInt(e.target.value,10))}
              aria-label="Max new features slider"
              title="Max new features"
            />
            <div className="text-xs text-slate-400">
              {p.maxNew} features → ~{interactions} interactions → ~{totalEngineered} total new features
            </div>
          </div>
        </div>
      </div>

      {/* Performance Mode (Smoke presets) */}
      <div className="mb-4 rounded-md border border-slate-700 bg-slate-900 p-3">
        <div className="mb-2 text-sm font-semibold text-slate-100">Performance Mode</div>
        <div className="mb-2 flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-2">
              <input type="radio" name="mode" checked={!p.smoke} onChange={()=>p.setSmoke(false)} /> Full Run
            </label>
            <label className="flex items-center gap-2">
              <input type="radio" name="mode" checked={p.smoke} onChange={()=>p.setSmoke(true)} /> Quick Test
            </label>
          </div>
          {p.smoke && (
            <div className="flex items-center gap-2">
              <button className="btn btn-xs" onClick={()=>applyPreset('fast')}>Fast</button>
              <button className="btn btn-xs" onClick={()=>applyPreset('balanced')}>Balanced</button>
              <button className="btn btn-xs" onClick={()=>applyPreset('thorough')}>Thorough</button>
            </div>
          )}
          <div className="ml-auto text-xs text-slate-300">Estimated: ~{est} minutes</div>
        </div>
        {p.smoke && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <label className="flex flex-col gap-2">
              <span className="text-sm">Eras</span>
              <input className="input" type="number" value={p.smokeEras} onChange={e=>p.setSmokeEras(parseInt(e.target.value||'0',10))} />
              <span className="text-xs text-slate-400">Caps eras processed.</span>
            </label>
            <label className="flex flex-col gap-2">
              <span className="text-sm">Rows</span>
              <input className="input" type="number" value={p.smokeRows} onChange={e=>p.setSmokeRows(parseInt(e.target.value||'0',10))} />
              <span className="text-xs text-slate-400">Total row limit.</span>
            </label>
            <label className="flex flex-col gap-2">
              <span className="text-sm">Feature cap</span>
              <input className="input" type="number" value={p.smokeFeat} onChange={e=>p.setSmokeFeat(parseInt(e.target.value||'0',10))} />
              <span className="text-xs text-slate-400">Used by pathfinding.</span>
            </label>
          </div>
        )}
      </div>

      {/* Advanced Toggles */}
      <div className="mb-4 rounded-md border border-slate-700 bg-slate-900 p-3">
        <div className="mb-2 text-sm font-semibold text-slate-100">Advanced Toggles</div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">

          <label className="flex items-center gap-2">
            <input type="checkbox" checked={p.disablePF} onChange={e=>p.setDisablePF(e.target.checked)} /> Disable pathfinding
          </label>
          <div className="flex flex-col gap-2">
            <span className="text-sm">Reproducibility</span>
            <div className="relative w-full md:w-80">
              <input
                ref={seedRef}
                className="input w-full pr-40"
                type="number"
                value={p.seed}
                onChange={e=>p.setSeed(parseInt(e.target.value||'0',10))}
                aria-label="Seed"
                title="Seed"
                placeholder="Seed"
              />
              <div className="absolute inset-y-0 right-1 flex items-center gap-1">
                <button
                  className="btn btn-xs"
                  onClick={()=>p.setSeed(Math.floor(Math.random()*1_000_000_000))}
                  title="Random seed"
                  aria-label="Random seed"
                >
                  Random
                </button>
                <button
                  className="btn btn-xs"
                  onClick={()=>p.setSeed(42)}
                  title="Lucky 42"
                  aria-label="Lucky 42"
                >
                  Lucky 42
                </button>
              </div>
            </div>
            <div className="text-xs text-slate-400">Same seed = same sampling and randomized choices.</div>
          </div>
        </div>
      </div>

      {/* What am I doing? */}
      <div className="rounded-md border border-slate-700 bg-slate-900 p-3">
        <div className="text-sm font-semibold text-slate-100">What am I actually doing?</div>
        <div className="mt-1 text-xs text-slate-300">
          You're about to run a <span className="font-semibold">{modeLabel}</span> that will take approximately <span className="font-semibold">~{est} minutes</span>.
          {p.disablePF ? ' Pathfinding is disabled (no new features will be engineered).' : ' Pathfinding is enabled and will engineer new features.'}
          {' '}Max features set to <span className="font-semibold">{p.maxNew}</span> ({zone==='green'? 'safe': zone==='yellow'? 'ambitious':'risky'} zone).
        </div>
        {(zone==='red') && (
          <div className="mt-2 inline-block rounded bg-red-900 px-2 py-1 text-xs text-red-200">⚠️ This may be slow and increase overfitting risk.</div>
        )}
      </div>

      {open && (
        <GlobalPickerModal
          mode={open==='parquet'? 'parquet':'features'}
          onSelect={(v)=>{
            if (open==='parquet') p.setInputData(v); else p.setFeaturesJson(v)
          }}
          onClose={()=> setOpen(null)}
        />
      )}
    </div>
  )
}
