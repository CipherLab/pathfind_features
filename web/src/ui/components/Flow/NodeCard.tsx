import * as React from 'react'
import { Handle, Position, useReactFlow } from '@xyflow/react'
import type { Connection, Edge as FlowEdge, Node as FlowNode, NodeProps } from '@xyflow/react'
import { NodeData, NodeStatus, PayloadType } from './types'
import { HandleTypes, NodeConstraints } from './node-spec'
import { isValidConnection } from './validation'

function StatusDot({ s }: { s: NodeStatus }) {
  const cls: Record<NodeStatus, string> = {
    idle: 'bg-slate-400',
    configured: 'bg-green-400',
  pending: 'bg-amber-400',
    running: 'bg-cyan-400',
    complete: 'bg-violet-400',
    failed: 'bg-red-400',
    blocked: 'bg-amber-400',
  }
  return <span className={`inline-block h-2.5 w-2.5 rounded-full ${cls[s]}`} />
}

export default function NodeCard(props: NodeProps<FlowNode<NodeData>>) {
  const { id } = props
  const data = props.data as unknown as NodeData
  const { getNodes, getEdges } = useReactFlow()
  const icon =
    data.kind === 'data-source'
      ? '📁'
      : data.kind === 'feature-selection'
      ? '🧬'
      : data.kind === 'target-discovery'
      ? '🎯'
      : data.kind === 'pathfinding'
      ? '🔍'
      : data.kind === 'feature-engineering'
      ? '⚗️'
      : data.kind === 'transform'
      ? '🔄'
      : data.kind === 'train'
      ? '🏋️'
      : data.kind === 'validate'
      ? '✅'
      : '📊'

  // helper: style per handle type
  const styleFor = (t: PayloadType, ghost = false): React.CSSProperties => {
    const base: React.CSSProperties = {
      background: HandleTypes[t].color,
      border: `2px solid ${HandleTypes[t].color}`,
      width: 12,
      height: 12,
      opacity: ghost ? 0.5 : 1,
      boxShadow: ghost ? `0 0 0 6px ${HandleTypes[t].color}22` : undefined,
    }
    switch (HandleTypes[t].shape) {
      case 'circle':
        base.borderRadius = 999
        break
      case 'square':
        base.borderRadius = 2
        break
      case 'diamond':
        base.transform = 'rotate(45deg)'
        base.borderRadius = 2
        break
      case 'star':
        // approximate star via clip-path
        base.clipPath = 'polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%)'
        break
      case 'triangle':
        base.clipPath = 'polygon(50% 0%, 0% 100%, 100% 100%)'
        break
    }
    return base
  }

  const cons = NodeConstraints[data.kind]
  const edges = getEdges()
  const incoming = (handleId: string) => edges.filter(e => e.target === id && e.targetHandle === handleId)

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 shadow-md text-slate-100 min-w-[160px] select-none hover:ring-1 hover:ring-indigo-400/40">
      {/* Render typed input handles (left) */}
    {cons.inputs.length === 0 ? (
        <></>
      ) : (
        <div>
      {cons.inputs.map((inp: { id: string; type: PayloadType; label: string }, idx: number) => (
            <Handle
              key={inp.id}
              id={inp.id}
              type="target"
              position={Position.Left}
              style={{ ...styleFor(inp.type, incoming(inp.id).length === 0), top: 16 + idx * 16 }}
        isValidConnection={(conn: any) =>
          isValidConnection(
            conn as Connection,
            getNodes() as unknown as FlowNode<NodeData>[],
            getEdges() as unknown as FlowEdge[]
          )}
            />
          ))}
        </div>
      )}
      <div className="flex items-center justify-between gap-2">
        <div className="font-semibold truncate" title={data.title}>
          {icon} {data.title}
        </div>
        <StatusDot s={data.status} />
      </div>
      {data.statusText && (
        <div className="mt-1 text-xs text-slate-300" title={data.statusText}>
          {data.statusText}
        </div>
      )}
      {data?.config?.summary && (
        <div
          className="mt-2 max-h-20 overflow-hidden text-ellipsis text-xs text-slate-300"
          title={data.config.summary}
        >
          {data.config.summary}
        </div>
      )}
      {/* Render typed output handles (right). Prefer plural outputs when available */}
      {Array.isArray(cons.outputs) && cons.outputs.length > 0 ? (
        <>
          {cons.outputs.map((out, idx) => (
            <div key={out.id} className="flex items-center justify-end" style={{ position: 'absolute', right: 0, top: 16 + idx * 16 }}>
              <span className="text-xs text-slate-400 mr-2">{out.label}</span>
              <Handle
                id={out.id}
                type="source"
                position={Position.Right}
                style={{ ...styleFor(out.type) }}
                isValidConnection={(conn: any) =>
                  isValidConnection(
                    conn as Connection,
                    getNodes() as unknown as FlowNode<NodeData>[],
                    getEdges() as unknown as FlowEdge[]
                  )}
              />
            </div>
          ))}
        </>
      ) : (
        cons.output && (
          <Handle
            id={cons.output.id}
            type="source"
            position={Position.Right}
            style={styleFor(cons.output.type)}
            isValidConnection={(conn: any) =>
              isValidConnection(
                conn as Connection,
                getNodes() as unknown as FlowNode<NodeData>[],
                getEdges() as unknown as FlowEdge[]
              )}
          />
        )
      )}
    </div>
  )
}
