import React, { useState, useEffect } from 'react'
import { jget } from '../../../lib/api'

interface TransformConfig {
  script?: string;
  outputPath?: string;
  logs?: string;
  // Optional: raw arguments string and parsed array for the transform script
  scriptArgsStr?: string;
  scriptArgs?: string[];
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

  // Minimal shell-like argv splitter (supports quotes)
  const parseArgs = (s: string): string[] => {
    const out: string[] = [];
    let cur = '';
    let quote: '"' | "'" | null = null;
    for (let i = 0; i < s.length; i++) {
      const ch = s[i];
      if (quote) {
        if (ch === quote) {
          quote = null;
        } else if (ch === '\\' && i + 1 < s.length && s[i + 1] === quote) {
          cur += quote; i++;
        } else {
          cur += ch;
        }
      } else {
        if (ch === '"' || ch === "'") {
          quote = ch as any;
        } else if (ch === ' ' || ch === '\n' || ch === '\t') {
          if (cur) { out.push(cur); cur = ''; }
        } else {
          cur += ch;
        }
      }
    }
    if (cur) out.push(cur);
    return out;
  };


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
      <label className="flex flex-col gap-1">
        <span className="text-sm text-slate-300">Arguments (optional)</span>
        <input
          className="input text-sm font-mono"
          value={cfg.scriptArgsStr || ''}
          onChange={e => {
            const raw = e.target.value;
            updateData({ scriptArgsStr: raw, scriptArgs: parseArgs(raw) });
          }}
          placeholder="e.g. --last-n 200 --era-col era"
        />
        <span className="text-xs text-slate-400">Passed to your script as sys.argv tokens. Quotes are supported.</span>
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