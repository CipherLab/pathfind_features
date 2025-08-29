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

      <div className="rounded-md border border-slate-700 bg-slate-900/50 p-3">
        <details>
          <summary className="cursor-pointer text-sm font-semibold text-slate-300">Advanced Discovery Settings</summary>
          <div className="mt-4 space-y-4">
            <label className="flex flex-col gap-1">
              <span className="text-sm text-slate-300">Evaluation Mode</span>
              <select
                className="input text-sm"
                value={cfg.td_eval_mode || 'hybrid'}
                onChange={e => updateData({ td_eval_mode: e.target.value })}
              >
                <option value="hybrid">Hybrid (Linear screen + GBM refine)</option>
                <option value="linear_fast">Linear Fast (Ridge only)</option>
                <option value="gbm_full">GBM Full (High fidelity)</option>
              </select>
            </label>

            <div className="grid grid-cols-2 gap-3">
              <label className="flex flex-col gap-1">
                <span className="text-sm text-slate-300">Top K Models (Hybrid)</span>
                <input
                  className="input text-sm"
                  type="number"
                  value={cfg.td_top_full_models || 3}
                  onChange={e => updateData({ td_top_full_models: parseInt(e.target.value || '3', 10) })}
                />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-sm text-slate-300">Ridge Lambda</span>
                <input
                  className="input text-sm"
                  type="number"
                  step="0.1"
                  value={cfg.td_ridge_lambda || 1.0}
                  onChange={e => updateData({ td_ridge_lambda: parseFloat(e.target.value || '1.0') })}
                />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-sm text-slate-300">Sample per Era</span>
                <input
                  className="input text-sm"
                  type="number"
                  value={cfg.td_sample_per_era || 1500}
                  onChange={e => updateData({ td_sample_per_era: parseInt(e.target.value || '1500', 10) })}
                />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-sm text-slate-300">Max Combinations</span>
                <input
                  className="input text-sm"
                  type="number"
                  value={cfg.td_max_combinations || 12}
                  onChange={e => updateData({ td_max_combinations: parseInt(e.target.value || '12', 10) })}
                />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-sm text-slate-300">Feature Fraction</span>
                <input
                  className="input text-sm"
                  type="number"
                  step="0.05"
                  min="0.1"
                  max="1.0"
                  value={cfg.td_feature_fraction || 0.35}
                  onChange={e => updateData({ td_feature_fraction: parseFloat(e.target.value || '0.35') })}
                />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-sm text-slate-300">Boosting Rounds</span>
                <input
                  className="input text-sm"
                  type="number"
                  value={cfg.td_num_boost_round || 12}
                  onChange={e => updateData({ td_num_boost_round: parseInt(e.target.value || '12', 10) })}
                />
              </label>
            </div>

            <div className="text-sm font-medium text-slate-200 pt-2 border-t border-slate-700">Caching</div>
            <div className="grid grid-cols-2 gap-3">
                <label className="flex flex-col gap-1">
                    <span className="text-sm text-slate-300">Max Era Cache (mem)</span>
                    <input
                        className="input text-sm"
                        type="number"
                        value={cfg.td_max_era_cache || 0}
                        onChange={e => updateData({ td_max_era_cache: parseInt(e.target.value || '0', 10) })}
                    />
                </label>
                <label className="flex flex-col gap-1">
                    <span className="text-sm text-slate-300">Clear Cache Every (eras)</span>
                    <input
                        className="input text-sm"
                        type="number"
                        value={cfg.td_clear_cache_every || 0}
                        onChange={e => updateData({ td_clear_cache_every: parseInt(e.target.value || '0', 10) })}
                    />
                </label>
            </div>
            <label className="flex flex-col gap-1">
                <span className="text-sm text-slate-300">Persistent Cache Dir</span>
                <input
                    className="input text-sm"
                    type="text"
                    value={cfg.td_pre_cache_dir || ''}
                    onChange={e => updateData({ td_pre_cache_dir: e.target.value })}
                    placeholder="cache/td_pre_cache"
                />
            </label>
            <label className="flex items-center gap-2">
                <input
                    type="checkbox"
                    checked={cfg.td_persist_pre_cache || false}
                    onChange={e => updateData({ td_persist_pre_cache: e.target.checked })}
                />
                <span className="text-sm text-slate-300">Enable Persistent Cache</span>
            </label>
          </div>
        </details>
      </div>
    </div>
  )
}
