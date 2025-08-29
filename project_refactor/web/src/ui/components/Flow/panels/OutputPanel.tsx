import React from 'react'

interface OutputConfig {
  outputPath?: string;
}

type Props = {
  cfg: OutputConfig
  updateData: (patch: any) => void
}

export default function OutputPanel({ cfg, updateData }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <label className="flex flex-col gap-1">
        <span className="text-sm text-slate-300">Output File Path</span>
        <input
          className="input text-sm"
          type="text"
          value={cfg.outputPath || ''}
          onChange={e => updateData({ outputPath: e.target.value })}
          placeholder="e.g., pipeline_runs/my_output.parquet"
        />
      </label>
      <div className="text-xs text-slate-500">
        This path is auto-generated from the upstream node. You can override it here.
      </div>
    </div>
  )
}