import React from 'react'

type Props = {
  experimentName: string;
  onExperimentNameChange: (name: string) => void;
  seed: number;
  onSeedChange: (seed: number) => void;
  onRunPipeline: () => void
  onClear: () => void
  onSave: () => void
  onLoad: (event: React.ChangeEvent<HTMLInputElement>) => void
  progress: { total: number; completed: number }
}

export default function PipelineToolbar({ 
  experimentName,
  onExperimentNameChange,
  seed,
  onSeedChange,
  onRunPipeline, 
  onClear, 
  onSave, 
  onLoad, 
  progress 
}: Props) {
  const pct = progress.total ? Math.round((progress.completed / progress.total) * 100) : 0
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleLoadClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="flex items-center gap-4 border-b border-slate-700 bg-slate-900/60 p-2">
      <div className="flex items-center gap-2">
        <span className="text-sm text-slate-300">Experiment:</span>
        <input
          type="text"
          className="input input-sm w-40"
          value={experimentName}
          onChange={e => onExperimentNameChange(e.target.value)}
        />
      </div>
      <div className="flex items-center gap-2">
        <span className="text-sm text-slate-300">Seed:</span>
        <input
          type="number"
          className="input input-sm w-24"
          value={seed}
          onChange={e => onSeedChange(parseInt(e.target.value || '0', 10))}
        />
      </div>
      <div className="flex-1" />
      <button className="btn btn-primary" onClick={onRunPipeline}>
        Run Pipeline
      </button>
      <button className="btn" onClick={onClear}>
        Clear
      </button>
      <button className="btn" onClick={onSave}>
        Save
      </button>
      <button className="btn" onClick={handleLoadClick}>
        Load
      </button>
      <input
        type="file"
        ref={fileInputRef}
        onChange={onLoad}
        style={{ display: 'none' }}
        accept=".json"
      />
      {progress.total > 0 && (
        <div className="flex items-center gap-2">
          <progress value={progress.completed} max={progress.total}></progress>
          <span className="text-xs text-slate-300 w-10 text-right">{pct}%</span>
        </div>
      )}
    </div>
  )
}