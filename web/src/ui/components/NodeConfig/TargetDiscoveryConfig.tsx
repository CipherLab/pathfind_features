import * as React from 'react'
import GlobalPickerModal from '../Wizard/GlobalPickerModal'
import { API_BASE } from '../../lib/api'

interface TargetDiscoveryConfig {
  inputData?: string
  smoke?: boolean
  smokeEras?: number
  smokeRows?: number
  smokeTargets?: number
  walkForward?: boolean
  seed?: number
  targetsName?: string
}

type Props = {
  cfg: TargetDiscoveryConfig
  onChange: (patch: any) => void
}

export default function TargetDiscoveryConfig({ cfg, onChange }: Props) {
  const [open, setOpen] = React.useState<null | 'parquet'>(null)
  const [maxHint, setMaxHint] = React.useState<{eras?: number; rows?: number} | null>(null)

  const ensureTargetsPrefix = (name: string) => {
    const base = (name || '').trim().replace(/\.json$/i, '')
    const withPrefix = base.startsWith('targets_') ? base : `targets_${base}`
    return `${withPrefix}.json`
  }

  const fetchDatasetStats = async () => {
    if (!cfg.inputData) return null
    try {
      const r = await fetch(`${API_BASE}/datasets/stats`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: cfg.inputData, era_col: 'era' }),
      })
      if (!r.ok) return null
      const data = await r.json()
      const eras = typeof data?.distinct_eras === 'number' ? data.distinct_eras : undefined
      const rows = typeof data?.rows === 'number' ? data.rows : undefined
      setMaxHint({ eras, rows })
      return { eras, rows }
    } catch {
      return null
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-md border border-slate-700 bg-slate-900 p-3">
        <label className="flex flex-col gap-2">
          <span className="text-sm font-medium">Input data</span>
          <button
            type="button"
            onClick={() => setOpen('parquet')}
            title={cfg.inputData}
            className="w-full truncate rounded-md border border-slate-600 bg-slate-800 px-3 py-2 text-left hover:border-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-600"
          >
            {cfg.inputData || '(select data parquet)'}
          </button>
          <span className="text-xs text-slate-400">Parquet file containing the training rows.</span>
        </label>
      </div>

      <div className="rounded-md border border-slate-700 bg-slate-900 p-3">
        <label className="flex flex-col gap-2">
          <span className="text-sm font-medium">Targets JSON output</span>
          <input
            className="input"
            type="text"
            placeholder="targets_myexperiment.json"
            value={cfg.targetsName || ''}
            onChange={(e) => onChange({ targetsName: ensureTargetsPrefix(e.target.value) })}
          />
          <span className="text-xs text-slate-400">Will be saved with a 'targets_' prefix, e.g. targets_&lt;name&gt;.json.</span>
        </label>
      </div>

      <div className="rounded-md border border-slate-700 bg-slate-900 p-3">
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={cfg.smoke || false}
              onChange={(e) => onChange({ smoke: e.target.checked })}
            />
            <span className="text-sm font-medium">Smoke mode</span>
          </label>

          {cfg.smoke && (
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="btn-secondary"
                onClick={() =>
                  onChange({
                    smokeEras: 20,
                    smokeRows: 500,
                    smokeTargets: 20,
                  })
                }
              >
                Quick Test
              </button>
              <button
                type="button"
                className="btn-secondary"
                onClick={() =>
                  onChange({
                    smokeEras: 100,
                    smokeRows: 10000,
                    smokeTargets: 50,
                  })
                }
              >
                Full Run
              </button>
            </div>
          )}
        </div>

        {cfg.smoke && (
          <div className="mt-2 grid grid-cols-1 gap-4 md:grid-cols-2">
            <label className="flex flex-col gap-2">
              <span className="flex items-center justify-between text-sm">
                <span>Era limit</span>
                <button
                  type="button"
                  className="btn-secondary px-2 py-1 text-xs"
                  onClick={async () => {
                    const s = (await fetchDatasetStats()) || maxHint
                    const v = s?.eras ?? 0
                    onChange({ smokeEras: v })
                  }}
                  title={maxHint?.eras ? `Max eras: ${maxHint.eras}` : 'Set to max eras in dataset'}
                >
                  Max
                </button>
              </span>
              <input
                className="input"
                type="number"
                max={maxHint?.eras}
                value={cfg.smokeEras || 0}
                onChange={(e) => onChange({ smokeEras: parseInt(e.target.value || '0', 10) })}
              />
            </label>
            <label className="flex flex-col gap-2">
              <span className="flex items-center justify-between text-sm">
                <span>Rows per era</span>
                <button
                  type="button"
                  className="btn-secondary px-2 py-1 text-xs"
                  onClick={async () => {
                    const s = (await fetchDatasetStats()) || maxHint
                    const v = s?.rows ?? 0
                    onChange({ smokeRows: v })
                  }}
                  title={maxHint?.rows ? `Dataset rows (per-era max not available): ${maxHint.rows}` : 'No per-era max available'}
                >
                  Max
                </button>
              </span>
              <input
                className="input"
                type="number"
                max={maxHint?.rows}
                value={cfg.smokeRows || 0}
                onChange={(e) => onChange({ smokeRows: parseInt(e.target.value || '0', 10) })}
              />
            </label>
            <label className="flex flex-col gap-2">
              <span className="text-sm">Targets cap</span>
              <input
                className="input"
                type="number"
                value={cfg.smokeTargets || 0}
                onChange={(e) => onChange({ smokeTargets: parseInt(e.target.value || '0', 10) })}
              />
            </label>
          </div>
        )}
      </div>

      <div className="rounded-md border border-slate-700 bg-slate-900 p-3">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={cfg.walkForward || false}
            onChange={(e) => onChange({ walkForward: e.target.checked })}
          />
          <span className="text-sm font-medium">Walk-forward validation</span>
        </label>
      </div>

      <div className="rounded-md border border-slate-700 bg-slate-900 p-3">
        <label className="flex flex-col gap-2">
          <span className="text-sm font-medium">Seed</span>
          <input
            className="input"
            type="number"
            value={cfg.seed || 0}
            onChange={(e) => onChange({ seed: parseInt(e.target.value || '0', 10) })}
          />
        </label>
      </div>

      {open && (
        <GlobalPickerModal
          mode={open}
          onSelect={(v) => onChange({ inputData: v })}
          onClose={() => setOpen(null)}
        />
      )}
    </div>
  )
}

