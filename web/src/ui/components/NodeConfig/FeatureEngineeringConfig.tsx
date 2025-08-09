import * as React from 'react'


interface FeatureEngineeringNodeConfig {
  maxNew?: number
  strategy?: string
  naming?: string
}

type Props = {
  cfg: FeatureEngineeringNodeConfig
  onChange: (patch: Partial<FeatureEngineeringNodeConfig>) => void
}

export default function FeatureEngineeringConfig({ cfg, onChange }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <div className="text-xs text-slate-400">Inherits relationships.json from upstream.</div>

      <div className="rounded-md border border-slate-700 bg-slate-900 p-3">
        <label className="flex flex-col gap-2">
          <span className="text-sm font-medium">Max new features</span>
          <input
            className="input"
            type="number"
            value={cfg.maxNew || 0}
            onChange={(e) => onChange({ maxNew: parseInt(e.target.value || '0', 10) })}
          />
        </label>
      </div>

      <div className="rounded-md border border-slate-700 bg-slate-900 p-3">
        <label className="flex flex-col gap-2">
          <span className="text-sm font-medium">Generation strategy</span>
          <input
            className="input"
            type="text"
            value={cfg.strategy || ''}
            onChange={(e) => onChange({ strategy: e.target.value })}
          />
        </label>
      </div>

      <div className="rounded-md border border-slate-700 bg-slate-900 p-3">
        <label className="flex flex-col gap-2">
          <span className="text-sm font-medium">Naming pattern</span>
          <input
            className="input"
            type="text"
            value={cfg.naming || ''}
            onChange={(e) => onChange({ naming: e.target.value })}
          />
        </label>
      </div>
    </div>
  )
}

