import * as React from 'react'

type HelpItem = {
  key: string
  label: string
  what: string
  tradeoffs?: string
  recommended?: string
  runtime?: string
  suggested?: any
}

const HELP: HelpItem[] = [
  {
    key: 'input_data',
    label: 'Input data',
    what: 'Parquet dataset used for target discovery and optional feature engineering.',
    tradeoffs: 'Using train.parquet finds targets on historical data. validation/live.parquet are for analysis only.',
    recommended: 'v5.0/train.parquet for new discovery runs.',
    runtime: 'Larger files and more eras increase runtime roughly linearly.',
    suggested: 'v5.0/train.parquet'
  },
  {
    key: 'features_json',
    label: 'Features JSON',
    what: 'Feature definition file providing the base feature set used in pathfinding (Stage 2/3).',
    tradeoffs: 'Richer feature sets can improve discovery but increase memory/time.',
    recommended: 'v5.0/features.json (ships with sensible defaults).',
    runtime: 'More features → slower pathfinding; use Smoke feature limit to cap.',
    suggested: 'v5.0/features.json'
  },
  {
    key: 'experiment_name',
    label: 'Run name',
    what: 'Suffix used for the pipeline run folder name and artifact grouping.',
    tradeoffs: 'Descriptive names help with tracking and comparisons.',
    recommended: 'Short, descriptive (e.g., wizard, api_cli, exp_full).',
    suggested: 'wizard'
  },
  {
    key: 'max_new_features',
    label: 'Max new features',
    what: 'How many new relationship features to generate in Stage 3.',
    tradeoffs: 'Higher may improve performance but risks overfitting and longer runtime.',
    recommended: 'Smoke: 4–12. Full: 20 (or 40+ with yolo-mode).',
    runtime: 'Increases Stage 3 time roughly linearly.',
    suggested: 8
  },
  {
    key: 'disable_pathfinding',
    label: 'Disable pathfinding',
    what: 'Skips Stage 2 and Stage 3. Only performs target discovery (Stage 1).',
    tradeoffs: 'Much faster; no new features engineered.',
    recommended: 'Enable for quick target-only experiments.',
    runtime: 'Significant speed-up when enabled.',
    suggested: false
  },
  {
    key: 'pretty',
    label: 'Pretty output',
    what: 'Prints a formatted run summary table at the end.',
    tradeoffs: 'Cosmetic only.',
    recommended: 'Safe to leave on.',
    suggested: true
  },
  {
    key: 'smoke_mode',
    label: 'Smoke mode',
    what: 'Applies sampling limits for eras/rows/features to speed up iteration.',
    tradeoffs: 'Faster feedback with lower statistical confidence.',
    recommended: 'Enable during prototyping; disable for final results.',
    runtime: 'Large reduction in runtime and resource use when enabled.',
    suggested: true
  },
  {
    key: 'smoke_max_eras',
    label: 'Smoke → Max eras',
    what: 'Caps number of eras processed in smoke mode.',
    tradeoffs: 'Too small may underfit walk-forward weighting.',
    recommended: '30–120 for quick checks.',
    runtime: 'Linear with eras.',
    suggested: 60
  },
  {
    key: 'smoke_row_limit',
    label: 'Smoke → Row limit',
    what: 'Total row limit across batches per stage in smoke mode.',
    tradeoffs: 'Lower values reduce fidelity of metrics.',
    recommended: '100k–250k depending on machine.',
    runtime: 'Linear with rows.',
    suggested: 150000
  },
  {
    key: 'smoke_feature_limit',
    label: 'Smoke → Feature limit',
    what: 'Caps number of feature columns used by pathfinding.',
    tradeoffs: 'Too low may miss useful relationships.',
    recommended: '200–600 for quick runs.',
    runtime: 'Higher limits increase memory/time for Stage 2/3.',
    suggested: 300
  },
  {
    key: 'seed',
    label: 'Seed',
    what: 'Random seed for reproducibility of sampling and randomized choices.',
    tradeoffs: 'Changing seed can reveal sensitivity.',
    recommended: 'Keep fixed (e.g., 42) for comparability.',
    suggested: 42
  },
  {
    key: 'stage1_from',
    label: 'Reuse Stage 1 from run',
    what: 'Reuses previously discovered adaptive targets from a past run.',
    tradeoffs: 'Saves time; may carry forward biases from that run.',
    recommended: 'Use when iterating on Stage 2/3 with stable targets.',
    runtime: 'Skips Stage 1 work when provided.',
    suggested: ''
  },
  {
    key: 'stage2_from',
    label: 'Reuse Stage 2 from run',
    what: 'Reuses previously discovered relationships for feature engineering.',
    tradeoffs: 'Saves time; may constrain exploration.',
    recommended: 'Use to quickly test Stage 3 or model training with fixed relationships.',
    runtime: 'Skips Stage 2 work when provided.',
    suggested: ''
  },
]

type SetterMap = {
  [key: string]: (v:any)=>void
}

export default function ParameterHelp({ setters }: { setters?: SetterMap }){
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900 p-4 max-h-[70vh] overflow-auto">
      <div className="mb-2 text-sm font-semibold text-slate-200">Parameter guide</div>
      <div className="mb-3 text-xs text-slate-400">
        Tips for common scenarios:
        <ul className="list-disc pl-4">
          <li><span className="font-medium text-slate-300">Quick smoke test:</span> Smoke mode ON, eras 60, rows 150k, features 300, max new 8, disable PF optional.</li>
          <li><span className="font-medium text-slate-300">Full discovery:</span> Smoke OFF, max new 20 (or more with YOLO), pathfinding ON. Expect hours not minutes.</li>
        </ul>
      </div>
      <div className="grid grid-cols-1 gap-3">
        {HELP.map(h => (
          <div key={h.key} className="rounded-md border border-slate-700 bg-slate-800 p-3">
            <div className="text-sm font-medium text-slate-100 flex items-center justify-between">
              <span>{h.label}</span>
              {setters && h.hasOwnProperty('suggested') && (
                <button
                  className="btn btn-xs"
                  onClick={()=> setters[h.key]?.(h.suggested)}
                  disabled={!setters[h.key]}
                  title="Apply suggested value"
                >Use suggested</button>
              )}
            </div>
            <div className="mt-1 text-xs text-slate-300">{h.what}</div>
            {h.tradeoffs && <div className="mt-1 text-xs text-slate-400"><span className="font-semibold">Trade-offs:</span> {h.tradeoffs}</div>}
            {h.recommended && <div className="mt-1 text-xs text-slate-400"><span className="font-semibold">Recommended:</span> {h.recommended}</div>}
            {h.runtime && <div className="mt-1 text-xs text-slate-400"><span className="font-semibold">Runtime impact:</span> {h.runtime}</div>}
          </div>
        ))}
      </div>
    </div>
  )
}
