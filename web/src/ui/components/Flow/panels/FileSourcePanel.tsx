import React, { useState } from 'react'
import GlobalPickerModal from '../../Wizard/GlobalPickerModal'
import { PayloadType } from '../../Flow/types'

// Map of friendly options to PayloadType and picker mode
const TYPE_OPTIONS: { key: string; label: string; payload: PayloadType; picker: 'parquet' | 'features' | 'none' }[] = [
  { key: 'PARQUET', label: 'Numerai parquet', payload: 'PARQUET', picker: 'parquet' },
  { key: 'JSON_ARTIFACT', label: 'Features JSON (model/features)', payload: 'JSON_ARTIFACT', picker: 'features' },
  { key: 'RELATIONSHIPS', label: 'Pathfinding relationships (json)', payload: 'RELATIONSHIPS', picker: 'none' },
  { key: 'TD_CANDIDATE_TARGETS', label: 'Adaptive targets parquet', payload: 'TD_CANDIDATE_TARGETS', picker: 'parquet' },
  { key: 'TD_DISCOVERY_META', label: 'Target discovery meta (json)', payload: 'TD_DISCOVERY_META', picker: 'none' },
]

type Props = {
  cfg: any
  updateData: (patch: any) => void
}

export default function FileSourcePanel({ cfg, updateData }: Props) {
  const [open, setOpen] = useState<null | 'parquet' | 'features'>(null)
  const selected = TYPE_OPTIONS.find(o => o.payload === cfg.payloadType) || TYPE_OPTIONS[0]

  const chooseType = (optKey: string) => {
    const opt = TYPE_OPTIONS.find(o => o.key === optKey) || TYPE_OPTIONS[0]
    updateData({ payloadType: opt.payload })
  }

  const openPickerForType = () => {
    const picker = selected.picker
    if (picker === 'none') return
    setOpen(picker)
  }

  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className="text-sm font-medium mb-2">File type</div>
        <div className="grid grid-cols-1 gap-2">
          {TYPE_OPTIONS.map(opt => (
            <label key={opt.key} className="flex items-center gap-2 text-sm">
              <input
                type="radio"
                name="fileType"
                className="accent-indigo-400"
                checked={(cfg.payloadType || 'PARQUET') === opt.payload}
                onChange={() => chooseType(opt.key)}
              />
              <span>{opt.label}</span>
            </label>
          ))}
        </div>
        <div className="text-xs text-slate-400 mt-1">Select what kind of artifact this file represents. Connections will be validated against this.</div>
      </div>

      <div>
        <div className="text-sm font-medium mb-2">Select file</div>
        <button
          className="btn w-full justify-start"
          onClick={openPickerForType}
          title={cfg.inputPath}
        >
          {cfg.inputPath || 'Choose a file…'}
        </button>
        <div className="text-xs text-slate-400 mt-1">
          Picker will filter to relevant choices for parquet and features.json. For JSON relationships or discovery meta, paste a path.
        </div>
        {(selected.picker === 'none') && (
          <input
            className="input mt-2 text-sm"
            placeholder="e.g. pipeline_runs/exp/02_relationships_ABC.json"
            value={cfg.inputPath || ''}
            onChange={e => updateData({ inputPath: e.target.value })}
          />
        )}
      </div>

      {open && (
        <GlobalPickerModal
          mode={open}
          onSelect={v => {
            updateData({ inputPath: v })
            setOpen(null)
          }}
          onClose={() => setOpen(null)}
        />
      )}
    </div>
  )
}
