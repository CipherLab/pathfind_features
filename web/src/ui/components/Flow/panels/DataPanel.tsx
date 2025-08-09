import React, { useState } from 'react'
import GlobalPickerModal from '../../Wizard/GlobalPickerModal'

type Props = {
  cfg: any
  updateData: (patch: any) => void
}

export default function DataPanel({ cfg, updateData }: Props) {
  const [open, setOpen] = useState<null | 'parquet'>(null)
  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className="text-sm font-medium mb-2">Input data</div>
        <button
          className="btn w-full justify-start"
          onClick={() => setOpen('parquet')}
          title={cfg.inputData}
        >
          {cfg.inputData}
        </button>
        <div className="text-xs text-slate-400 mt-1">
          Parquet file containing the training rows.
        </div>
      </div>
      <label className="flex flex-col gap-2">
        <span className="text-sm">Run name</span>
        <input
          className="input"
          value={cfg.runName}
          onChange={e => updateData({ runName: e.target.value })}
        />
      </label>
      <label className="flex flex-col gap-2">
        <span className="text-sm">Seed</span>
        <input
          className="input"
          type="number"
          value={cfg.seed}
          onChange={e => updateData({ seed: parseInt(e.target.value || '0', 10) })}
        />
      </label>
      {open && (
        <GlobalPickerModal
          mode="parquet"
          onSelect={v => {
            updateData({ inputData: v })
            setOpen(null)
          }}
          onClose={() => setOpen(null)}
        />
      )}
    </div>
  )
}
