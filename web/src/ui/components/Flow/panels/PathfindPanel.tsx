import React from 'react'

type Props = {
  cfg: any
  updateData: (patch: any) => void
}

export default function PathfindPanel({ cfg, updateData }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className="row-between">
          <span className="text-sm font-medium">Max new features</span>
          <span className="text-xs rounded bg-indigo-900/50 px-2 py-0.5">{cfg.maxNew}</span>
        </div>
        <input
          aria-label="Max new features"
          title="Max new features"
          type="range"
          min={0}
          max={60}
          step={1}
          value={cfg.maxNew}
          onChange={e => updateData({ maxNew: parseInt(e.target.value, 10) })}
        />
      </div>
      <label className="flex flex-col gap-1">
        <span className="text-sm">Feature cap (for PF)</span>
        <input
          className="input"
          type="number"
          value={cfg.smokeFeat}
          onChange={e => updateData({ smokeFeat: parseInt(e.target.value || '0', 10) })}
        />
      </label>
      <label className="row-center">
        <input
          type="checkbox"
          checked={cfg.disablePF}
          onChange={e => updateData({ disablePF: e.target.checked })}
        />{' '}
        Disable pathfinding
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
