import React, { useState, useEffect } from 'react'
import { jget } from '../../../lib/api'

interface TransformConfig {
  script?: string;
  outputPath?: string;
  logs?: string;
}

type Props = {
  cfg: TransformConfig
  updateData: (patch: any) => void
}

interface TransformDef {
  name: string;
  script: string;
}

export default function TransformPanel({ cfg, updateData }: Props) {
  const [transforms, setTransforms] = useState<TransformDef[]>([]);
  const [showConsole, setShowConsole] = useState(false);

  useEffect(() => {
    jget<TransformDef[]>('/transforms')
      .then(setTransforms)
      .catch(console.error);
  }, []);


  return (
    <div className="flex flex-col gap-4">
      {cfg.outputPath && (
        <div className="flex flex-col gap-1">
          <span className="text-sm text-slate-300">Output Path</span>
          <div className="input text-sm text-slate-400 bg-slate-800">{cfg.outputPath}</div>
        </div>
      )}
      <label className="flex flex-col gap-1">
        <span className="text-sm text-slate-300">Transform Script (Python)</span>
        <textarea
          className="input text-sm font-mono"
          rows={8}
          value={cfg.script || ''}
          onChange={e => updateData({ script: e.target.value })}
          placeholder="def transform(df):\n  # Your pandas transform logic here\n  return df"
        />
      </label>
      <div className="grid grid-cols-2 gap-2">
        {transforms.map((transform) => (
          <button
            key={transform.name}
            className="btn-sm bg-slate-700 hover:bg-slate-600 text-xs"
            onClick={() => updateData({ script: transform.script })}
          >
            {transform.name}
          </button>
        ))}
      </div>
      {cfg.logs && (
        <div>
          <button
            className="text-sm text-slate-300 hover:text-slate-100"
            onClick={() => setShowConsole(!showConsole)}
          >
            {showConsole ? 'Hide' : 'Show'} Console Output
          </button>
          {showConsole && (
            <pre className="text-xs text-slate-400 bg-slate-800 p-2 rounded-md mt-2">
              {cfg.logs}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}