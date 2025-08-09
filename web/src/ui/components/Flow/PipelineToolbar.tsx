import React from 'react'

type Props = {
  onRunPipeline: () => void
  onClear: () => void
  onAutoArrange: () => void
  progress: { total: number; completed: number }
}

export default function PipelineToolbar({ onRunPipeline, onClear, onAutoArrange, progress }: Props) {
  const pct = progress.total ? Math.round((progress.completed / progress.total) * 100) : 0
  return (
    <div className="flex items-center gap-2 border-b border-slate-700 bg-slate-900/60 p-2">
      <button className="btn btn-primary" onClick={onRunPipeline}>
        Run Pipeline
      </button>
      <button className="btn" onClick={onClear}>
        Clear
      </button>
      <button className="btn" onClick={onAutoArrange}>
        Auto-arrange to lanes
      </button>
      {progress.total > 0 && (
        <div className="flex items-center gap-2 flex-1">
          <progress className="flex-1" value={progress.completed} max={progress.total}></progress>
          <span className="text-xs text-slate-300 w-10 text-right">{pct}%</span>
        </div>
      )}
    </div>
  )
}
