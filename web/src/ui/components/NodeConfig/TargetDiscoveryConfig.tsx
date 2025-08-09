import * as React from 'react'
import GlobalPickerModal from '../Wizard/GlobalPickerModal'

interface TargetDiscoveryConfigCfg {
  inputData?: string
  smoke?: boolean
  smokeEras?: number
  smokeRows?: number
  walkForward?: boolean
  seed?: number
}

type Props = {
  cfg: TargetDiscoveryConfigCfg
  onChange: (patch: any) => void
}

export default function TargetDiscoveryConfig({ cfg, onChange }: Props) {
  const [open, setOpen] = React.useState<null | 'parquet'>(null)

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
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={cfg.smoke || false}
            onChange={(e) => onChange({ smoke: e.target.checked })}
          />
          <span className="text-sm font-medium">Smoke mode</span>
        </label>
        {cfg.smoke && (
          <div className="mt-2 grid grid-cols-1 gap-4 md:grid-cols-2">
            <label className="flex flex-col gap-2">
              <span className="text-sm">Era limit</span>
              <input
                className="input"
                type="number"
                value={cfg.smokeEras || 0}
                onChange={(e) => onChange({ smokeEras: parseInt(e.target.value || '0', 10) })}
              />
            </label>
            <label className="flex flex-col gap-2">
              <span className="text-sm">Row limit</span>
              <input
                className="input"
                type="number"
                value={cfg.smokeRows || 0}
                onChange={(e) => onChange({ smokeRows: parseInt(e.target.value || '0', 10) })}
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

