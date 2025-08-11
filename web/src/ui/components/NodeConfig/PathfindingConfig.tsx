import * as React from 'react'
import { Node } from '@xyflow/react'
import { NodeData } from '../Flow/types'

interface PathfindingNodeConfig {
  yolo?: boolean
  featureLimit?: number
  debug?: boolean
  relationships?: number
}

type Props = {
  nodes: Node<NodeData>[]
  cfg: PathfindingNodeConfig
  onChange: (patch: Partial<PathfindingNodeConfig>) => void
}

export default function PathfindingConfig({ nodes, cfg, onChange }: Props) {
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

