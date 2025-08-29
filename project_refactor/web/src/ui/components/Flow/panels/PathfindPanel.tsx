import React, { useEffect, useState } from 'react'
import { Edge, Node } from '@xyflow/react'
import { NodeData } from '../types'

type Props = {
  cfg: any
  updateData: (patch: any) => void
  nodes: Node<NodeData>[]
  edges: Edge[]
  selection: Node<NodeData>
  getUpstreamCap?: (node: Node<NodeData>, capType: 'features' | 'targets' | 'relationships') => number
}

export default function PathfindPanel({ cfg, updateData, nodes, edges, selection, getUpstreamCap }: Props) {
  // Use getUpstreamCap if provided, else fallback to old logic
  const upstreamTargetCap = getUpstreamCap
    ? getUpstreamCap(selection, 'features')
    : (() => {
        const edge = edges.find(e => e.target === selection.id && e.targetHandle === 'in-adaptive-targets');
        if (!edge) return 0;
        const upstream = nodes.find(n => n.id === edge.source);
        return upstream?.data?.config?.smokeTargets || 0;
      })();
  const [showWarning, setShowWarning] = useState(false)
  useEffect(() => {
    if (upstreamTargetCap > 0 && cfg.smokeFeat > upstreamTargetCap) {
      updateData({ smokeFeat: upstreamTargetCap })
      setShowWarning(true)
    } else {
      setShowWarning(false)
    }
  }, [upstreamTargetCap, cfg.smokeFeat])

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <button
          type="button"
          className="btn-secondary"
          onClick={() =>
            updateData({
              maxNew: 50,
              smokeFeat: Math.min(200, upstreamTargetCap || 200),
            })
          }
        >
          Quick Test
        </button>
        <button
          type="button"
          className="btn-secondary"
          onClick={() =>
            updateData({
              maxNew: 100,
              smokeFeat: Math.min(500, upstreamTargetCap || 500),
            })
          }
        >
          Full Run
        </button>
      </div>
      <div>
        <div className="row-between">
          <span className="text-sm font-medium">Max new features</span>
          <span className="text-xs rounded bg-indigo-900/50 px-2 py-0.5">{cfg.maxNew}</span>
        </div>
        <input
          aria-label="Max new features"
          title="Max new features"
          type="range"
          min={0}
          max={60}
          step={1}
          value={cfg.maxNew}
          onChange={e => updateData({ maxNew: parseInt(e.target.value, 10) })}
        />
      </div>
      <label className="flex flex-col gap-1">
        <span className="text-sm">Feature cap (for PF)</span>
        <input
          className="input"
          type="number"
          value={cfg.smokeFeat}
          min={1}
          max={upstreamTargetCap > 0 ? upstreamTargetCap : undefined}
          onChange={e => {
            const v = parseInt(e.target.value || '0', 10)
            updateData({ smokeFeat: v })
          }}
        />
        {upstreamTargetCap > 0 && (
          <div className="text-xs text-slate-400">
            (Max: {upstreamTargetCap} from upstream Target Discovery)
          </div>
        )}
        {showWarning && (
          <div className="text-xs text-red-400">Value exceeds upstream cap; clamped.</div>
        )}
      </label>
      <label className="row-center">
        <input
          type="checkbox"
          checked={cfg.disablePF}
          onChange={e => updateData({ disablePF: e.target.checked })}
        />{' '}
        Disable pathfinding
      </label>

    </div>
  )
}
