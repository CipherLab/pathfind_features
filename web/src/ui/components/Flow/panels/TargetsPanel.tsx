import React from 'react'

type Props = {
  cfg: any
  updateData: (patch: any) => void
}

export default function TargetsPanel({ cfg, updateData }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <div className="text-xs text-slate-400">
        Inherits features.json from an upstream features node.
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
                onChange={e =>
                  updateData({ smokeRows: parseInt(e.target.value || '0', 10) })
                }
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs">Feature cap</span>
              <input
                className="input"
                type="number"
                value={cfg.smokeFeat}
                onChange={e =>
                  updateData({ smokeFeat: parseInt(e.target.value || '0', 10) })
                }
              />
            </label>
          </div>
        )}
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
