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
  ReactFlowInstance,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
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
  const icon = data.kind === 'data-source' ? '📁'
    : data.kind === 'target-discovery' ? '🎯'
    : data.kind === 'pathfinding' ? '🔍'
    : data.kind === 'feature-engineering' ? '⚗️'
    : '📊'
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 shadow-md text-slate-100 min-w-[160px] select-none hover:ring-1 hover:ring-indigo-400/40">
  <Handle type="target" position={Position.Left} />
      <div className="flex items-center justify-between gap-2">
        <div className="font-semibold truncate" title={data.title}>{icon} {data.title}</div>
        <StatusDot s={data.status} />
      </div>
      {data?.config?.summary && (
        <div className="mt-2 max-h-20 overflow-hidden text-ellipsis text-xs text-slate-300" title={data.config.summary}>{data.config.summary}</div>
      )}
  <Handle type="source" position={Position.Right} />
    </div>
  )
}

const nodeTypes: NodeTypes = { appNode: NodeCard }

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

  // Small node-specific panels
  function DataPanel(){
    const [open, setOpen] = useState<null|'parquet'>(null)
    return (
      <div className="flex flex-col gap-4">
        <div>
          <div className="text-sm font-medium mb-2">Input data</div>
          <button className="btn w-full justify-start" onClick={()=>setOpen('parquet')} title={cfg.inputData}>{cfg.inputData}</button>
          <div className="text-xs text-slate-400 mt-1">Parquet file containing the training rows.</div>
        </div>
        <label className="flex flex-col gap-2">
          <span className="text-sm">Run name</span>
          <input className="input" value={cfg.runName} onChange={e=>updateData({ runName: e.target.value })} />
        </label>
        <label className="flex flex-col gap-2">
          <span className="text-sm">Seed</span>
          <input className="input" type="number" value={cfg.seed} onChange={e=>updateData({ seed: parseInt(e.target.value||'0',10) })} />
        </label>
        {open && (
          <GlobalPickerModal mode="parquet" onSelect={(v)=>{ updateData({ inputData: v }); setOpen(null) }} onClose={()=> setOpen(null)} />
        )}
      </div>
    )
  }

  function TargetsPanel(){
    const [open, setOpen] = useState<null|'features'>(null)
    return (
      <div className="flex flex-col gap-4">
        <div>
          <div className="text-sm font-medium mb-2">Features JSON</div>
          <button className="btn w-full justify-start" onClick={()=>setOpen('features')} title={cfg.featuresJson}>{cfg.featuresJson}</button>
          <div className="text-xs text-slate-400 mt-1">Feature definition file with feature_sets.medium.</div>
        </div>
        <div>
          <div className="text-sm font-medium mb-2">Performance Mode</div>
          <div className="row items-center">
            <label className="row-center"><input type="radio" checked={!cfg.smoke} onChange={()=>updateData({ smoke: false })} /> Full Run</label>
            <label className="row-center"><input type="radio" checked={cfg.smoke} onChange={()=>updateData({ smoke: true })} /> Quick Test</label>
          </div>
          {cfg.smoke && (
            <div className="grid grid-cols-3 gap-2 mt-2">
              <label className="flex flex-col gap-1"><span className="text-xs">Eras</span><input className="input" type="number" value={cfg.smokeEras} onChange={e=>updateData({ smokeEras: parseInt(e.target.value||'0',10) })} /></label>
              <label className="flex flex-col gap-1"><span className="text-xs">Rows</span><input className="input" type="number" value={cfg.smokeRows} onChange={e=>updateData({ smokeRows: parseInt(e.target.value||'0',10) })} /></label>
              <label className="flex flex-col gap-1"><span className="text-xs">Feature cap</span><input className="input" type="number" value={cfg.smokeFeat} onChange={e=>updateData({ smokeFeat: parseInt(e.target.value||'0',10) })} /></label>
            </div>
          )}
        </div>
        <label className="row-center"><input type="checkbox" checked={cfg.pretty} onChange={e=>updateData({ pretty: e.target.checked })} /> Pretty output</label>
        {open && (
          <GlobalPickerModal mode="features" onSelect={(v)=>{ updateData({ featuresJson: v }); setOpen(null) }} onClose={()=> setOpen(null)} />
        )}
      </div>
    )
  }

  function PathfindPanel(){
    return (
      <div className="flex flex-col gap-4">
        <div>
          <div className="row-between">
            <span className="text-sm font-medium">Max new features</span>
            <span className="text-xs rounded bg-indigo-900/50 px-2 py-0.5">{cfg.maxNew}</span>
          </div>
          <input aria-label="Max new features" title="Max new features" type="range" min={0} max={60} step={1} value={cfg.maxNew} onChange={e=>updateData({ maxNew: parseInt(e.target.value,10) })} />
        </div>
        <label className="flex flex-col gap-1">
          <span className="text-sm">Feature cap (for PF)</span>
          <input className="input" type="number" value={cfg.smokeFeat} onChange={e=>updateData({ smokeFeat: parseInt(e.target.value||'0',10) })} />
        </label>
        <label className="row-center"><input type="checkbox" checked={cfg.disablePF} onChange={e=>updateData({ disablePF: e.target.checked })} /> Disable pathfinding</label>
        <label className="row-center"><input type="checkbox" checked={cfg.pretty} onChange={e=>updateData({ pretty: e.target.checked })} /> Pretty output</label>
      </div>
    )
  }

  function FeaturesPanel(){
    return (
      <div className="flex flex-col gap-4">
        <label className="flex flex-col gap-1">
          <span className="text-sm">Max new engineered features</span>
          <input className="input" type="number" value={cfg.maxNew} onChange={e=>updateData({ maxNew: parseInt(e.target.value||'0',10) })} />
        </label>
        <label className="row-center"><input type="checkbox" checked={cfg.pretty} onChange={e=>updateData({ pretty: e.target.checked })} /> Pretty output</label>
      </div>
    )
  }

  function OutputPanel(){
    return (
      <div className="flex flex-col gap-4">
        <label className="row-center"><input type="checkbox" checked={cfg.pretty} onChange={e=>updateData({ pretty: e.target.checked })} /> Pretty output</label>
        <label className="flex flex-col gap-1"><span className="text-sm">Seed</span><input className="input" type="number" value={cfg.seed} onChange={e=>updateData({ seed: parseInt(e.target.value||'0',10) })} /></label>
      </div>
    )
  }

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
        {selection.data.kind==='data-source' && <DataPanel />}
        {selection.data.kind==='target-discovery' && <TargetsPanel />}
        {selection.data.kind==='pathfinding' && <PathfindPanel />}
        {selection.data.kind==='feature-engineering' && <FeaturesPanel />}
        {selection.data.kind==='output' && <OutputPanel />}
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
        <div key={it.kind} className="btn cursor-grab active:cursor-grabbing"
          draggable
          onDragStart={(e)=>{
            e.dataTransfer.setData('application/reactflow', it.kind)
            e.dataTransfer.effectAllowed = 'move'
          }}
          onDoubleClick={()=> onAdd(it.kind)}
          role="button"
          tabIndex={0}
          onKeyDown={(e)=>{ if (e.key==='Enter') onAdd(it.kind) }}
        >
          <span className="w-6 text-center">{it.icon}</span> {it.label}
        </div>
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
  const wrapperRef = useRef<HTMLDivElement | null>(null)
  const [rf, setRf] = useState<ReactFlowInstance<Node<NodeData>, Edge> | null>(null)

  const onConnect = useCallback((conn: Edge | Connection) => setEdges((eds) => addEdge({ ...conn, type: 'smoothstep' }, eds as any) as any), [])

  const addNode = useCallback((kind: NodeKind, position?: { x: number; y: number }) => {
    const id = `n${idRef.current++}`
    const title = kind === 'data-source' ? 'Data Source'
      : kind === 'target-discovery' ? 'Target Discovery'
      : kind === 'pathfinding' ? 'Pathfinding'
      : kind === 'feature-engineering' ? 'Feature Engineering'
      : 'Output'
    const n: Node<NodeData> = {
      id,
  type: 'appNode',
      position: position ?? { x: 140 + nodes.length * 50, y: 100 + nodes.length * 20 },
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

  // DnD handlers
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    const kind = event.dataTransfer.getData('application/reactflow') as NodeKind
    if (!kind || !rf) return
    const pos = rf.screenToFlowPosition({ x: event.clientX, y: event.clientY })
    addNode(kind, pos)
  }, [rf, addNode])

  return (
    <div className="flex h-[calc(100vh-120px)] gap-2">
      <div className="w-56 shrink-0">
        <NodePalette onAdd={addNode} />
      </div>
      <div className="flex min-w-0 flex-1 flex-col rounded-lg border border-slate-700 bg-slate-900/40">
        <PipelineToolbar onRunPipeline={onRunPipeline} onClear={onClear} />
        <div className="relative min-h-0 flex-1" ref={wrapperRef}>
          <ReactFlow<Node<NodeData>, Edge>
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
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
        <Sidebar selection={selection} onUpdate={onUpdateSelection} onRun={onRunNode} />
      </div>
    </div>
  )
}
