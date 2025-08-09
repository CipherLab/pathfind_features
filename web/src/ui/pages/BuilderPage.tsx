import * as React from 'react'
import { useCallback, useRef, useState } from 'react'
import { ReactFlow,
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
  Handle,
  Position,
  NodeProps,
  NodeTypes,
  useReactFlow,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import ParameterForm from '../components/Wizard/ParameterForm'
import GlobalPickerModal from '../components/Wizard/GlobalPickerModal'
import { jpost } from '../lib/api'

// Types
export type NodeKind = 'data-source' | 'target-discovery' | 'pathfinding' | 'feature-engineering' | 'output'
export type NodeStatus = 'idle' | 'configured' | 'running' | 'complete' | 'failed'

export type NodeData = {
  kind: NodeKind
  title: string
  status: NodeStatus
  // Lightweight config shared across types; extended per-kind via any for now
  config?: any
}

const connectionOrder: NodeKind[] = ['data-source', 'target-discovery', 'pathfinding', 'feature-engineering', 'output']
const allowsConnection = (a: NodeKind, b: NodeKind) => {
  const idx = connectionOrder.indexOf(a)
  return idx !== -1 && connectionOrder[idx + 1] === b
}

// Basic node renderer used for all nodes initially
function StatusDot({ s }: { s: NodeStatus }){
  const cls: Record<NodeStatus, string> = {
    idle: 'bg-slate-400',
    configured: 'bg-green-400',
    running: 'bg-cyan-400',
    complete: 'bg-violet-400',
    failed: 'bg-red-400',
  }
  return <span className={`inline-block h-2.5 w-2.5 rounded-full ${cls[s]}`} />
}

function NodeCard({ data }: NodeProps<Node<NodeData>>){
  const { getNode } = useReactFlow<Node<NodeData>>()
  const isValidConnection = useCallback((conn: Connection) => {
    const src = getNode(conn.source as string)
    const tgt = getNode(conn.target as string)
    return !!(src && tgt && allowsConnection(src.data.kind, tgt.data.kind))
  }, [getNode])

  const icon = data.kind === 'data-source' ? '📁'
    : data.kind === 'target-discovery' ? '🎯'
    : data.kind === 'pathfinding' ? '🔍'
    : data.kind === 'feature-engineering' ? '⚗️'
    : '📊'
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 shadow-md text-slate-100 min-w-[160px]">
  <Handle type="target" position={Position.Left} isValidConnection={isValidConnection} />
      <div className="flex items-center justify-between gap-2">
        <div className="font-semibold truncate" title={data.title}>{icon} {data.title}</div>
        <StatusDot s={data.status} />
      </div>
      {data?.config?.summary && (
        <div className="mt-2 max-h-20 overflow-hidden text-ellipsis text-xs text-slate-300" title={data.config.summary}>{data.config.summary}</div>
      )}
  <Handle type="source" position={Position.Right} isValidConnection={isValidConnection} />
    </div>
  )
}

const nodeTypes: NodeTypes = { default: NodeCard }

// Sidebar for configuration (reuses ParameterForm for now)
function Sidebar({
  selection,
  onUpdate,
  onRun,
}: {
  selection: Node<NodeData> | null
  onUpdate: (updater: (prev: NodeData) => NodeData) => void
  onRun: () => void
}){
  // Shared config shim; we reuse ParameterForm to edit common pipeline params
  const [cfg, setCfg] = useState<any>(() => ({
    inputData: 'v5.0/train.parquet',
    featuresJson: 'v5.0/features.json',
    runName: 'wizard',
    maxNew: 8,
    disablePF: false,
    pretty: true,
    smoke: true,
    smokeEras: 60,
    smokeRows: 150000,
    smokeFeat: 300,
    seed: 42,
  }))

  React.useEffect(() => {
    if (!selection) return
    // Load selection config into form if it exists
    if (selection.data?.config) {
      setCfg((p: any) => ({ ...p, ...selection.data.config }))
    }
  }, [selection?.id])

  const updateData = useCallback((patch: any) => {
    setCfg((prev: any) => {
      const next = { ...prev, ...patch }
      onUpdate((p) => ({ ...p, config: next, status: 'configured' }))
      return next
    })
  }, [onUpdate])

  if (!selection) {
    return (
      <div className="h-full p-3 text-sm text-slate-300">Select a node to configure it.</div>
    )
  }

  const d = selection.data
  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-slate-700 p-3">
        <div className="text-sm font-semibold text-slate-100">{d.title}</div>
        <div className="mt-1 text-xs text-slate-400">Status: {d.status}</div>
      </div>
      <div className="flex-1 overflow-auto p-3">
        {/* For Phase 1 we reuse ParameterForm directly. Later this will branch by kind. */}
        <ParameterForm
          inputData={cfg.inputData} setInputData={(v)=>updateData({ inputData: v })}
          featuresJson={cfg.featuresJson} setFeaturesJson={(v)=>updateData({ featuresJson: v })}
          runName={cfg.runName} setRunName={(v)=>updateData({ runName: v })}
          maxNew={cfg.maxNew} setMaxNew={(v)=>updateData({ maxNew: v })}
          disablePF={cfg.disablePF} setDisablePF={(v)=>updateData({ disablePF: v })}
          pretty={cfg.pretty} setPretty={(v)=>updateData({ pretty: v })}
          smoke={cfg.smoke} setSmoke={(v)=>updateData({ smoke: v })}
          smokeEras={cfg.smokeEras} setSmokeEras={(v)=>updateData({ smokeEras: v })}
          smokeRows={cfg.smokeRows} setSmokeRows={(v)=>updateData({ smokeRows: v })}
          smokeFeat={cfg.smokeFeat} setSmokeFeat={(v)=>updateData({ smokeFeat: v })}
          seed={cfg.seed} setSeed={(v)=>updateData({ seed: v })}
        />
      </div>
      <div className="border-t border-slate-700 p-3">
        <div className="row-between">
          <button className="btn btn-primary" onClick={onRun}>Run Node</button>
        </div>
      </div>
    </div>
  )
}

function NodePalette({ onAdd }: { onAdd: (kind: NodeKind) => void }){
  const items: { kind: NodeKind; label: string; icon: string }[] = [
    { kind: 'data-source', label: 'Data', icon: '📁' },
    { kind: 'target-discovery', label: 'Targets', icon: '🎯' },
    { kind: 'pathfinding', label: 'Pathfind', icon: '🔍' },
    { kind: 'feature-engineering', label: 'Features', icon: '⚗️' },
    { kind: 'output', label: 'Output', icon: '📊' },
  ]
  return (
    <div className="flex flex-col gap-2 p-2">
      <div className="text-xs font-semibold text-slate-300">Palette</div>
      {items.map((it) => (
        <button key={it.kind} className="btn" onClick={() => onAdd(it.kind)}>
          <span className="w-6 text-center">{it.icon}</span> {it.label}
        </button>
      ))}
    </div>
  )
}

function PipelineToolbar({ onRunPipeline, onClear }: { onRunPipeline: ()=>void; onClear: ()=>void }){
  return (
    <div className="flex items-center gap-2 border-b border-slate-700 bg-slate-900/60 p-2">
      <button className="btn btn-primary" onClick={onRunPipeline}>Run Pipeline</button>
      <button className="btn" onClick={onClear}>Clear</button>
    </div>
  )
}

export default function BuilderPage(){
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<NodeData>>([] as Node<NodeData>[])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([] as Edge[])
  const [selection, setSelection] = useState<Node<NodeData> | null>(null)
  const idRef = useRef(1)

  const onConnect = useCallback((conn: Edge | Connection) => {
    const source = nodes.find((n: Node<NodeData>) => n.id === conn.source)
    const target = nodes.find((n: Node<NodeData>) => n.id === conn.target)
    if (source && target && allowsConnection(source.data.kind, target.data.kind)) {
      setEdges((eds) => addEdge({ ...conn, type: 'smoothstep', style: { stroke: '#16a34a' } }, eds as any) as any)
    }
  }, [nodes])

  const addNode = useCallback((kind: NodeKind) => {
    const id = `n${idRef.current++}`
    const title = kind === 'data-source' ? 'Data Source'
      : kind === 'target-discovery' ? 'Target Discovery'
      : kind === 'pathfinding' ? 'Pathfinding'
      : kind === 'feature-engineering' ? 'Feature Engineering'
      : 'Output'
    const n: Node<NodeData> = {
      id,
      type: 'default',
      position: { x: 140 + nodes.length * 50, y: 100 + nodes.length * 20 },
      data: { kind, title, status: 'idle', config: {} },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    }
  setNodes((ns: Node<NodeData>[]) => [...ns, n])
    setSelection(n)
  }, [nodes.length])

  const onRunNode = useCallback(async()=>{
    if (!selection) return
    // For now, just show a command preview and call the same /runs endpoint when the node is the first stage
    const d = selection.data
    if (d.kind === 'target-discovery'){
      try{
        const cfg = d.config || {}
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
  setNodes((ns: Node<NodeData>[])=> ns.map((n: Node<NodeData>)=> n.id===selection.id? { ...n, data: { ...n.data, status: 'running' as NodeStatus } }: n))
        await jpost('/runs', payload)
  setNodes((ns: Node<NodeData>[])=> ns.map((n: Node<NodeData>)=> n.id===selection.id? { ...n, data: { ...n.data, status: 'complete' as NodeStatus } }: n))
      }catch{
  setNodes((ns: Node<NodeData>[])=> ns.map((n: Node<NodeData>)=> n.id===selection.id? { ...n, data: { ...n.data, status: 'failed' as NodeStatus } }: n))
      }
    }
  }, [selection])

  const onUpdateSelection = useCallback((updater: (prev: NodeData) => NodeData)=>{
    if (!selection) return
  setNodes((ns: Node<NodeData>[]) => ns.map((n: Node<NodeData>) => (n.id === selection.id ? { ...n, data: updater(n.data) } : n)))
  }, [selection])

  const onRunPipeline = useCallback(()=>{
    // Phase 1 foundation: if there is a target-discovery node configured, run it.
  const td = nodes.find((n: Node<NodeData>)=> n.data.kind==='target-discovery')
    setSelection(td || null)
    if (td) {
      // trigger node run for now
      setTimeout(()=>{
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        onRunNode()
      }, 0)
    }
  }, [nodes, onRunNode])

  const onClear = useCallback(()=>{ setNodes([]); setEdges([]); setSelection(null) }, [])

  return (
    <div className="flex h-[calc(100vh-140px)] gap-2">
      <div className="w-48 shrink-0">
        <NodePalette onAdd={addNode} />
      </div>
      <div className="flex min-w-0 flex-1 flex-col rounded-lg border border-slate-700 bg-slate-900/40">
        <PipelineToolbar onRunPipeline={onRunPipeline} onClear={onClear} />
        <div className="relative min-h-0 flex-1">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            onNodeClick={(_evt: React.MouseEvent, n: Node) => setSelection(n as Node<NodeData>)}
            fitView
          >
            <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
            <MiniMap pannable zoomable />
            <Controls />
          </ReactFlow>
        </div>
      </div>
      <div className="w-[420px] shrink-0 rounded-lg border border-slate-700 bg-slate-900/60">
        <Sidebar selection={selection} onUpdate={onUpdateSelection} onRun={onRunNode} />
      </div>
    </div>
  )
}
