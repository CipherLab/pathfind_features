import React from 'react'

type Props = {
  cfg: any
  updateData: (patch: any) => void
}

export default function OutputPanel({ cfg, updateData }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <label className="row-center">
        <input
          type="checkbox"
          checked={cfg.pretty}
          onChange={e => updateData({ pretty: e.target.checked })}
        />{' '}
        Pretty output
      </label>
      <label className="flex flex-col gap-1">
        <span className="text-sm">Seed</span>
        <input
          className="input"
          type="number"
          value={cfg.seed}
          onChange={e => updateData({ seed: parseInt(e.target.value || '0', 10) })}
        />
      </label>
    </div>
  )
}
