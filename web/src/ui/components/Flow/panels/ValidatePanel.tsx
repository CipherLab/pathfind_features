import React from 'react'

interface ValidationConfig {
  split?: number
  metrics?: string
}

type Props = {
  cfg: ValidationConfig
  updateData: (patch: any) => void
}

export default function ValidatePanel({ cfg, updateData }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <label className="flex flex-col gap-1">
        <span className="text-sm text-slate-300">Validation Split</span>
        <input
          className="input text-sm"
          type="number"
          step="0.1"
          min="0.1"
          max="1"
          value={cfg.split || 0.2}
          onChange={e => updateData({ split: parseFloat(e.target.value || '0.2') })}
        />
      </label>
      <label className="flex flex-col gap-1">
        <span className="text-sm text-slate-300">Metrics</span>
        <input
          className="input text-sm"
          value={cfg.metrics || ''}
          onChange={e => updateData({ metrics: e.target.value })}
          placeholder="accuracy, auc"
        />
      </label>
    </div>
  )
}
