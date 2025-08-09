import React from 'react'

type Props = {
  cfg: any
  updateData: (patch: any) => void
}

export default function FeaturesPanel({ cfg, updateData }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <label className="flex flex-col gap-1">
        <span className="text-sm">Max new engineered features</span>
        <input
          className="input"
          type="number"
          value={cfg.maxNew}
          onChange={e => updateData({ maxNew: parseInt(e.target.value || '0', 10) })}
        />
      </label>
      <label className="row-center">
        <input
          type="checkbox"
          checked={cfg.pretty}
          onChange={e => updateData({ pretty: e.target.checked })}
        />{' '}
        Pretty output
      </label>
    </div>
  )
}
