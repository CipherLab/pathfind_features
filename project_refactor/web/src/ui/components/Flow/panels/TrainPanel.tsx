import React from 'react'

interface TrainConfig {
  learningRate?: number
  epochs?: number
}

type Props = {
  cfg: TrainConfig
  updateData: (patch: any) => void
}

export default function TrainPanel({ cfg, updateData }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <label className="flex flex-col gap-1">
        <span className="text-sm text-slate-300">Learning Rate</span>
        <input
          className="input text-sm"
          type="number"
          step="0.01"
          value={cfg.learningRate || 0.1}
          onChange={e => updateData({ learningRate: parseFloat(e.target.value || '0.1') })}
        />
      </label>
      <label className="flex flex-col gap-1">
        <span className="text-sm text-slate-300">Epochs</span>
        <input
          className="input text-sm"
          type="number"
          value={cfg.epochs || 10}
          onChange={e => updateData({ epochs: parseInt(e.target.value || '10', 10) })}
        />
      </label>
    </div>
  )
}
