import * as React from 'react'
import { useCallback, useRef, useState } from 'react'
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  addEdge,
  useEdgesState,
  useNodesState,
  Connection,
  Edge,
  Node,
  Position,
  NodeProps,
  useReactFlow,
} from '@xyflow/react'
import type { NodeTypes, ReactFlowInstance } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { jpost } from '../lib/api'
import { NodeData, NodeKind, NodeStatus } from '../components/Flow/types'
import { HandleTypes, NodeConstraints } from '../components/Flow/node-spec'
import { isValidConnection, validatePipeline } from '../components/Flow/validation'
import NodeCard from '../components/Flow/NodeCard'
import Sidebar from '../components/Flow/Sidebar'
import NodePalette from '../components/Flow/NodePalette'
import PipelineToolbar from '../components/Flow/PipelineToolbar'

const nodeTypes: NodeTypes = { appNode: NodeCard as any }

export default function BuilderPage() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<NodeData>>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])
  const [selection, setSelection] = useState<Node<NodeData> | null>(null)
  const [progress, setProgress] = useState({ total: 0, completed: 0 })
  const idRef = useRef(1)
  const wrapperRef = useRef<HTMLDivElement | null>(null)
  const [rf, setRf] = useState<ReactFlowInstance<Node<NodeData>, Edge> | null>(null)
  

  // Helper to pretty edge coloring and labels
  const edgeVisualsFor = (srcKind: NodeKind): { stroke: string; label: string; labelColor: string } => {
    const out = NodeConstraints[srcKind].output
    const color = out ? HandleTypes[out.type].color : '#64748b'
    const label = out?.label || ''
    return { stroke: color, label, labelColor: color }
  }

  const onConnect = useCallback(
    (conn: Connection) => {
      const source = nodes.find((n: Node<NodeData>) => n.id === conn.source)
      const target = nodes.find((n: Node<NodeData>) => n.id === conn.target)
      if (!source || !target) return
      // validate with current graph
      if (!isValidConnection(conn, nodes, edges)) return
      const vis = edgeVisualsFor(source.data.kind)
      const payloadType = NodeConstraints[source.data.kind].output?.type
      const label = vis.label
      setEdges(eds =>
        addEdge(
          {
            ...conn,
            type: 'smoothstep',
            data: { payloadType },
            label,
            labelStyle: { fill: vis.labelColor, fontSize: 10 },
            style: { stroke: vis.stroke, strokeWidth: 2 },
          } as any,
          eds as any
        ) as any
      )
    },
  [nodes, edges, setEdges]
  )

  const addNode = useCallback(
    (kind: NodeKind, position?: { x: number; y: number }) => {
      const id = `n${idRef.current++}`
      const title =
        kind === 'data-source'
          ? 'Data Source'
          : kind === 'target-discovery'
          ? 'Target Discovery'
          : kind === 'pathfinding'
          ? 'Pathfinding'
          : kind === 'feature-engineering'
          ? 'Feature Engineering'
          : kind === 'transform'
          ? 'Transform'
          : kind === 'train'
          ? 'Train'
          : kind === 'validate'
          ? 'Validate'
          : 'Output'
  const n: Node<NodeData> = {
        id,
        type: 'appNode',
        position: position ?? { x: 140 + nodes.length * 50, y: 100 + nodes.length * 20 },
        data: { kind, title, status: 'idle', statusText: '', config: {} },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      }
      setNodes((ns: Node<NodeData>[]) => [...ns, n])
      setSelection(n)
    },
    [nodes.length, setNodes]
  )

  // Recompute blocked states when graph changes
  

  React.useEffect(() => {
    setNodes((ns: Node<NodeData>[]) => {
      const nodeMap = new Map(ns.map(n => [n.id, n]))
      const incomingByTarget = new Map<string, Edge[]>(
        ns.map(n => [n.id, edges.filter(e => e.target === n.id)])
      )
      const next = ns.map(n => {
        const cons = NodeConstraints[n.data.kind]
        // compute missing inputs
        const missing: string[] = []
        for (const inp of cons.inputs) {
          const inc = (incomingByTarget.get(n.id) || []).find(e => e.targetHandle === inp.id)
          if (!inc) {
            missing.push(inp.label)
            continue
          }
          const up = inc ? nodeMap.get(inc.source) : undefined
          if (!up || up.data.status === 'failed' || up.data.status === 'blocked' || up.data.status === 'idle') {
            missing.push(inp.label)
          }
        }
        // preserve running/complete/failed; otherwise flip between blocked/configured/idle
        if (n.data.status === 'running' || n.data.status === 'complete' || n.data.status === 'failed') return n
        if (missing.length > 0) {
          return {
            ...n,
            data: { ...n.data, status: 'blocked' as NodeStatus, statusText: `Missing: ${missing.join(', ')}` },
          }
        }
        // if previously blocked and now satisfied, become configured if had config
        if (n.data.status === 'blocked') {
          const configured = n.data.config && Object.keys(n.data.config).length > 0
          return { ...n, data: { ...n.data, status: (configured ? 'configured' : 'idle') as NodeStatus, statusText: '' } }
        }
        return n
      })
      return next
    })
  }, [edges, setNodes])

  // Inherit artifacts downstream after certain nodes complete
  const propagateArtifacts = useCallback(
    (src: Node<NodeData>) => {
      setNodes((ns: Node<NodeData>[]) => {
        let updated = ns
        if (src.data.kind === 'target-discovery') {
          const downstream = edges.filter(e => e.source === src.id).map(e => e.target)
          updated = ns.map(n => {
            if (downstream.includes(n.id) && n.data.kind === 'pathfinding') {
              const cfg = {
                ...n.data.config,
                inheritTargetsFrom: src.id,
                targetsJson: 'target_discovery.json',
              }
              const status = (n.data.status === 'idle' ? 'configured' : n.data.status) as NodeStatus
              return { ...n, data: { ...n.data, config: cfg, status } }
            }
            return n
          })
        } else if (src.data.kind === 'pathfinding') {
          const downstream = edges.filter(e => e.source === src.id).map(e => e.target)
          updated = ns.map(n => {
            if (downstream.includes(n.id) && n.data.kind === 'feature-engineering') {
              const cfg = {
                ...n.data.config,
                inheritRelationshipsFrom: src.id,
                relationshipsJson: 'relationships.json',
              }
              const status = (n.data.status === 'idle' ? 'configured' : n.data.status) as NodeStatus
              return { ...n, data: { ...n.data, config: cfg, status } }
            }
            return n
          })
        }
        return updated
      })
    },
    [edges, setNodes]
  )

  const runNode = useCallback(
    async (node: Node<NodeData>) => {
      const { id, data } = node
      const cfg = data.config || {}

      // mark running
      setNodes(ns =>
        ns.map(n =>
          n.id === id
            ? {
                ...n,
                data: {
                  ...n.data,
                  status: 'running' as NodeStatus,
                  statusText:
                    data.kind === 'target-discovery'
                      ? 'Running target discovery...'
                      : data.kind === 'pathfinding'
                      ? 'Exploring relationships...'
                      : data.kind === 'feature-engineering'
                      ? 'Brewing features...'
                      : 'Working...',
                },
              }
            : n
        )
      )

      try {
        if (data.kind === 'target-discovery') {
          const payload = {
            input_data: cfg.inputData || 'v5.0/train.parquet',
            features_json: cfg.featuresJson || 'v5.0/features.json',
            run_name: cfg.runName || 'wizard',
            max_new_features: cfg.maxNew ?? 8,
            disable_pathfinding: true,
            pretty: cfg.pretty ?? true,
            smoke_mode: cfg.smoke ?? true,
            smoke_max_eras: cfg.smokeEras,
            smoke_row_limit: cfg.smokeRows,
            smoke_feature_limit: cfg.smokeFeat,
            seed: cfg.seed ?? 42,
          }
          await jpost('/runs', payload)
        } else {
          // Placeholder for other node kinds until real endpoints exist
          await new Promise(res => setTimeout(res, 300))
        }

        // mark complete and write a friendly status line
        setNodes(ns =>
          ns.map(n =>
            n.id === id
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    status: 'complete' as NodeStatus,
                    statusText:
                      data.kind === 'target-discovery'
                        ? '✅ Targets discovered'
                        : data.kind === 'pathfinding'
                        ? '✅ Relationships mapped'
                        : data.kind === 'feature-engineering'
                        ? '✅ Features generated'
                        : '✅ Done',
                  },
                }
              : n
          )
        )

        // propagate outputs to downstream nodes when relevant
        if (data.kind === 'target-discovery' || data.kind === 'pathfinding') {
          propagateArtifacts(node)
        }
      } catch (e: any) {
        const msg =
          e && typeof e === 'object' && 'message' in e ? String(e.message) : 'Unknown error'
        setNodes(ns =>
          ns.map(n =>
            n.id === id
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    status: 'failed' as NodeStatus,
                    statusText: `❌ ${msg}`,
                  },
                }
              : n
          )
        )
        // propagate blocked state to downstream
        setNodes(ns => {
          const downstreamIds = edges.filter(e => e.source === id).map(e => e.target)
          if (downstreamIds.length === 0) return ns
          return ns.map(n => {
            if (!downstreamIds.includes(n.id)) return n
            if (n.data.status === 'running' || n.data.status === 'complete' || n.data.status === 'failed') return n
            return {
              ...n,
              data: { ...n.data, status: 'blocked' as NodeStatus, statusText: `Upstream failed: ${node.data.title}` },
            }
          })
        })
        throw e
      }
    },
    [setNodes, propagateArtifacts, edges]
  )

  const onRunNode = useCallback(async () => {
    if (!selection) return
    await runNode(selection)
  }, [selection, runNode])

  const onUpdateSelection = useCallback(
    (updater: (prev: NodeData) => NodeData) => {
      if (!selection) return
      setNodes((ns: Node<NodeData>[]) =>
        ns.map((n: Node<NodeData>) =>
          n.id === selection.id ? { ...n, data: updater(n.data) } : n
        )
      )
    },
    [selection, setNodes]
  )

  const onRunPipeline = useCallback(async () => {
    const order = validatePipeline(nodes, edges)
    if (!order) return
    setProgress({ total: order.length, completed: 0 })
    const map = new Map(nodes.map(n => [n.id, n]))
    for (const id of order) {
      const node = map.get(id)
      if (!node) continue
      try {
        await runNode(node)
        setProgress(p => ({ total: p.total, completed: p.completed + 1 }))
      } catch {
        alert(`Node "${node.data.title}" failed. Pipeline stopped.`)
        break
      }
    }
  }, [nodes, edges, runNode])

  const onClear = useCallback(() => {
    setNodes([])
    setEdges([])
    setSelection(null)
  }, [setNodes, setEdges])

  const deleteSelection = useCallback(() => {
    if (!selection) return
    setNodes(ns => ns.filter(n => n.id !== selection.id))
    setEdges(es => es.filter(e => e.source !== selection.id && e.target !== selection.id))
    setSelection(null)
  }, [selection, setNodes, setEdges])

  const onEdgeDoubleClick = useCallback(
    (_: React.MouseEvent, edge: Edge) => {
      setEdges(es => es.filter(e => e.id !== edge.id))
    },
    [setEdges]
  )

  

  // DnD handlers
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault()
      const kind = event.dataTransfer.getData('application/reactflow') as NodeKind
      if (!kind || !rf) return
      const pos = rf.screenToFlowPosition({ x: event.clientX, y: event.clientY })
      addNode(kind, pos)
    },
    [rf, addNode]
  )

  return (
    <div className="flex h-[calc(100vh-120px)] gap-2">
      <div className="w-56 shrink-0">
        <NodePalette onAdd={addNode} />
      </div>
      <div className="flex min-w-0 flex-1 flex-col rounded-lg border border-slate-700 bg-slate-900/40">
  <PipelineToolbar onRunPipeline={onRunPipeline} onClear={onClear} progress={progress} />
        <div className="relative min-h-0 flex-1" ref={wrapperRef}>
          <ReactFlow<Node<NodeData>, Edge>
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onEdgeDoubleClick={onEdgeDoubleClick}
            deleteKeyCode={[ 'Backspace', 'Delete' ]}
            onNodesDelete={nd => {
              if (selection && nd.some(n => n.id === selection.id)) setSelection(null)
            }}
            nodeTypes={nodeTypes}
            onNodeClick={(_evt: React.MouseEvent, n: Node<NodeData>) => setSelection(n)}
            onInit={setRf}
            onDrop={onDrop}
            onDragOver={onDragOver}
            
            fitView
          >
            <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
            <MiniMap pannable zoomable />
            <Controls />
          </ReactFlow>
          
        </div>
      </div>
        <div className="w-[420px] shrink-0 rounded-lg border border-slate-700 bg-slate-900/60">
          <Sidebar
            selection={selection}
            edges={edges}
            onUpdate={onUpdateSelection}
            onRun={onRunNode}
            onDelete={deleteSelection}
          />
        </div>
    </div>
  )
}