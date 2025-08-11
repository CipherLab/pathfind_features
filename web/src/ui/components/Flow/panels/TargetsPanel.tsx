import React, { useEffect, useState } from 'react'

type Props = {
  cfg: any
  updateData: (patch: any) => void
  upstreamCap?: number // for future extensibility
}

export default function TargetsPanel({ cfg, updateData }: Props) {
  const ensureTargetsPrefix = (name: string) => {
    const base = (name || '').trim().replace(/\.json$/i, '')
    const withPrefix = base.startsWith('targets_') ? base : `targets_${base}`
    return `${withPrefix}.json`
  }
  // Clamp targets cap to a reasonable max (e.g. 1000)
  const MAX_TARGETS = 1000
  const [showWarning, setShowWarning] = useState(false)
  useEffect(() => {
    if (cfg.smokeTargets > MAX_TARGETS) {
      updateData({ smokeTargets: MAX_TARGETS, smokeFeat: MAX_TARGETS })
      setShowWarning(true)
    } else {
      setShowWarning(false)
    }
  }, [cfg.smokeTargets])
  return (
    <div className="flex flex-col gap-4">
      <div className="text-xs text-slate-400">
        {cfg.inheritFeaturesFrom
          ? `Inherits features.json from ${cfg.inheritFeaturesFrom === true ? 'an upstream features node' : cfg.inheritFeaturesFrom}.`
          : 'Does not inherit features.json from any upstream node.'}
      </div>
      <div>
        <div className="text-sm font-medium mb-2">Performance Mode</div>
        <div className="row items-center">
          <label className="row-center">
            <input
              type="radio"
              checked={!cfg.smoke}
              onChange={() => updateData({ smoke: false })}
            />{' '}
            Full Run
          </label>
          <label className="row-center">
            <input
              type="radio"
              checked={cfg.smoke}
              onChange={() => updateData({ smoke: true })}
            />{' '}
            Quick Test
          </label>
        </div>
        {cfg.smoke && (
          <div className="grid grid-cols-3 gap-2 mt-2">
            <label className="flex flex-col gap-1">
              <span className="text-xs">Eras</span>
              <input
                className="input"
                type="number"
                value={cfg.smokeEras}
                min={1}
                max={1000}
                onChange={e =>
                  updateData({ smokeEras: parseInt(e.target.value || '0', 10) })
                }
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs">Rows</span>
              <input
                className="input"
                type="number"
                value={cfg.smokeRows}
                min={1}
                max={1000000}
                onChange={e =>
                  updateData({ smokeRows: parseInt(e.target.value || '0', 10) })
                }
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs">Targets cap</span>
              <input
                className="input"
                type="number"
                value={(cfg as any).smokeTargets ?? (cfg as any).smokeFeat ?? 0}
                min={1}
                max={MAX_TARGETS}
                onChange={e => {
                  const v = parseInt(e.target.value || '0', 10)
                  updateData({ smokeTargets: v, smokeFeat: v })
                }}
              />
              <div className="text-xs text-slate-400">(Max: {MAX_TARGETS})</div>
              {showWarning && (
                <div className="text-xs text-red-400">Value exceeds max allowed; clamped to {MAX_TARGETS}.</div>
              )}
            </label>
          </div>
        )}
      </div>
      <div>
        <div className="text-sm font-medium mb-2">Targets JSON output</div>
        <input
          className="input"
          type="text"
          placeholder="targets_myexperiment.json"
          value={cfg.targetsName || ''}
          onChange={e => updateData({ targetsName: ensureTargetsPrefix(e.target.value) })}
        />
        <div className="text-xs text-slate-400 mt-1">Will save using a 'targets_' prefix.</div>
      </div>
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
