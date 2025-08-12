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

export default function FeaturesPanel({ cfg, updateData, nodes, edges, selection, getUpstreamCap }: Props) {
  // Use getUpstreamCap if provided, else fallback to old logic
  const upstreamMaxNew = getUpstreamCap
    ? getUpstreamCap(selection, 'relationships')
    : (() => {
        const edge = edges.find(e => e.target === selection.id && e.targetHandle === 'relationships');
        if (!edge) return 0;
        const upstream = nodes.find(n => n.id === edge.source);
        return upstream?.data?.config?.maxNew || 0;
      })();
  const [showWarning, setShowWarning] = useState(false)
  useEffect(() => {
    if (upstreamMaxNew > 0 && cfg.maxNew > upstreamMaxNew) {
      updateData({ maxNew: upstreamMaxNew })
      setShowWarning(true)
    } else {
      setShowWarning(false)
    }
  }, [upstreamMaxNew, cfg.maxNew])

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <button
          type="button"
          className="btn-secondary"
          onClick={() =>
            updateData({
              maxNew: Math.min(50, upstreamMaxNew || 50),
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
              maxNew: Math.min(100, upstreamMaxNew || 100),
            })
          }
        >
          Full Run
        </button>
      </div>
      <label className="flex flex-col gap-1">
        <span className="text-sm">Max new engineered features</span>
        <input
          className="input"
          type="number"
          value={cfg.maxNew || 0}
          min={1}
          max={upstreamMaxNew > 0 ? upstreamMaxNew : undefined}
          onChange={e => {
            const v = parseInt(e.target.value || '0', 10)
            updateData({ maxNew: v })
          }}
        />
        {upstreamMaxNew > 0 && (
          <div className="text-xs text-slate-400">
            (Max: {upstreamMaxNew} from upstream Pathfinding)
          </div>
        )}
        {showWarning && (
          <div className="text-xs text-red-400">Value exceeds upstream cap; clamped.</div>
        )}
      </label>
      
    </div>
  )
}
