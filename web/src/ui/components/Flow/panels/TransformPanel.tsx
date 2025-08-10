import React from 'react'

interface TransformConfig {
  script?: string
}

type Props = {
  cfg: TransformConfig
  updateData: (patch: any) => void
}

export default function TransformPanel({ cfg, updateData }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <label className="flex flex-col gap-1">
        <span className="text-sm text-slate-300">Transform Script</span>
        <textarea
          className="input text-sm"
          rows={3}
          value={cfg.script || ''}
          onChange={e => updateData({ script: e.target.value })}
          placeholder="e.g., normalize features"
        />
      </label>
    </div>
  )
}
