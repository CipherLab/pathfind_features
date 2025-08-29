import * as React from 'react'
import { Node } from '@xyflow/react'
import { NodeData } from '../Flow/types'

interface PathfindingNodeConfig {
  yolo?: boolean
  featureLimit?: number
  debug?: boolean
  relationships?: number
  // Extended PF params
  smokeFeat?: number // internal PF feature cap
  lastNEras?: number
  cacheDir?: string
  sanityCheck?: boolean
  nPaths?: number
  maxPathLen?: number
  minStrength?: number
  topK?: number
  batchSize?: number
}

type Props = {
  nodes: Node<NodeData>[]
  cfg: PathfindingNodeConfig
  onChange: (patch: Partial<PathfindingNodeConfig>) => void
}

export default function PathfindingConfig({ nodes, cfg, onChange }: Props) {
  // Compute a read-only preview of the cache folder based on params
  const paramKeyInputs = {
    yolo: Boolean(cfg.yolo || false),
    feature_limit: cfg.featureLimit || undefined,
    row_limit: undefined as number | undefined, // not exposed here; left undefined
    pf_feature_cap: cfg.smokeFeat || undefined,
    n_paths: cfg.nPaths || undefined,
    max_path_length: cfg.maxPathLen || undefined,
    min_strength: cfg.minStrength ?? undefined,
    top_k: cfg.topK || undefined,
    batch_size: cfg.batchSize || undefined,
    last_n_eras: cfg.lastNEras || undefined,
    era_col: 'era',
  } as const;
  const paramKey = React.useMemo(() => {
    try {
      const json = JSON.stringify(paramKeyInputs);
      // lightweight hash
      let h = 0;
      for (let i = 0; i < json.length; i++) {
        h = (h * 31 + json.charCodeAt(i)) >>> 0;
      }
      return h.toString(16).slice(0, 8);
    } catch {
      return 'na';
    }
  }, [paramKeyInputs]);
  const baseCache = cfg.cacheDir && cfg.cacheDir.length > 0 ? cfg.cacheDir : 'cache/pathfinding_cache';
  const cachePreview = `${baseCache}/pf_${paramKey}`;
  return (
    <div className="flex flex-col gap-4">
      <div className="text-xs text-slate-400">Uses target_discovery.json from upstream.</div>

      <div className="rounded-md border border-slate-700 bg-slate-900 p-3">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={cfg.yolo || false}
            onChange={(e) => onChange({ yolo: e.target.checked })}
          />
          <span className="text-sm font-medium">YOLO mode</span>
          <div className="rounded-md border border-slate-700 bg-slate-900 p-3 grid grid-cols-2 gap-3">
            <label className="flex flex-col gap-1">
              <span className="text-sm font-medium">Last N eras</span>
              <input
                className="input"
                type="number"
                min={0}
                value={cfg.lastNEras || 0}
                onChange={(e) => onChange({ lastNEras: parseInt(e.target.value || '0', 10) })}
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-sm font-medium">PF feature cap</span>
              <input
                className="input"
                type="number"
                min={0}
                value={cfg.smokeFeat || 0}
                onChange={(e) => onChange({ smokeFeat: parseInt(e.target.value || '0', 10) })}
              />
            </label>
            <label className="flex flex-col gap-1 col-span-2">
              <span className="text-sm font-medium">Cache directory</span>
              <input
                className="input"
                type="text"
                value={cfg.cacheDir || ''}
                onChange={(e) => onChange({ cacheDir: e.target.value })}
                placeholder="cache/pathfinding_cache"
              />
              <span className="text-xs text-slate-400">Resolved cache folder (read-only): {cachePreview}</span>
            </label>
            <label className="flex items-center gap-2 col-span-2">
              <input
                type="checkbox"
                checked={cfg.sanityCheck || false}
                onChange={(e) => onChange({ sanityCheck: e.target.checked })}
              />
              <span className="text-sm font-medium">Run sanity check</span>
            </label>
          </div>

          <details className="rounded-md border border-slate-700 bg-slate-900 p-3">
            <summary className="cursor-pointer text-sm font-medium">Advanced</summary>
            <div className="grid grid-cols-2 gap-3 mt-3">
              <label className="flex flex-col gap-1">
                <span className="text-sm">nPaths</span>
                <input
                  className="input"
                  type="number"
                  min={0}
                  value={cfg.nPaths || 0}
                  onChange={(e) => onChange({ nPaths: parseInt(e.target.value || '0', 10) })}
                />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-sm">Max path length</span>
                <input
                  className="input"
                  type="number"
                  min={0}
                  value={cfg.maxPathLen || 0}
                  onChange={(e) => onChange({ maxPathLen: parseInt(e.target.value || '0', 10) })}
                />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-sm">Min strength</span>
                <input
                  className="input"
                  type="number"
                  step="0.01"
                  value={cfg.minStrength ?? ''}
                  onChange={(e) => onChange({ minStrength: e.target.value === '' ? undefined : parseFloat(e.target.value) })}
                />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-sm">Top-K</span>
                <input
                  className="input"
                  type="number"
                  min={0}
                  value={cfg.topK || 0}
                  onChange={(e) => onChange({ topK: parseInt(e.target.value || '0', 10) })}
                />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-sm">Batch size</span>
                <input
                  className="input"
                  type="number"
                  min={0}
                  value={cfg.batchSize || 0}
                  onChange={(e) => onChange({ batchSize: parseInt(e.target.value || '0', 10) })}
                />
              </label>
            </div>
          </details>
        </label>
      </div>

      <div className="rounded-md border border-slate-700 bg-slate-900 p-3">
        <label className="flex flex-col gap-2">
          <span className="text-sm font-medium">Feature limit</span>
          <input
            className="input"
            type="number"
            value={cfg.featureLimit || 0}
            onChange={(e) => onChange({ featureLimit: parseInt(e.target.value || '0', 10) })}
          />
        </label>
      </div>

      <div className="rounded-md border border-slate-700 bg-slate-900 p-3">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={cfg.debug || false}
            onChange={(e) => onChange({ debug: e.target.checked })}
          />
          <span className="text-sm font-medium">Debug</span>
        </label>
      </div>

      <div className="rounded-md border border-slate-700 bg-slate-900 p-3">
        <label className="flex flex-col gap-2">
          <span className="text-sm font-medium">Relationship count</span>
          <input
            className="input"
            type="number"
            value={cfg.relationships || 0}
            onChange={(e) => onChange({ relationships: parseInt(e.target.value || '0', 10) })}
          />
        </label>
      </div>
    </div>
  )
}

