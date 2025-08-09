import React, { useCallback, useState } from 'react'
import { Node } from '@xyflow/react'
import { NodeData } from './types'
import DataPanel from './panels/DataPanel'
import TargetsPanel from './panels/TargetsPanel'
import PathfindPanel from './panels/PathfindPanel'
import FeaturesPanel from './panels/FeaturesPanel'
import OutputPanel from './panels/OutputPanel'

type Props = {
  selection: Node<NodeData> | null
  onUpdate: (updater: (prev: NodeData) => NodeData) => void
  onRun: () => void
}

export default function Sidebar({ selection, onUpdate, onRun }: Props) {
  // Shared config shim
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
    if (selection.data?.config) {
      setCfg((p: any) => ({ ...p, ...selection.data.config }))
    }
  }, [selection?.id])

  const updateData = useCallback(
    (patch: any) => {
      setCfg((prev: any) => {
        const next = { ...prev, ...patch }
        // configuring a node clears any stale status text
        onUpdate(p => ({ ...p, config: next, status: 'configured', statusText: '' }))
        return next
      })
    },
    [onUpdate]
  )

  if (!selection) {
    return <div className="h-full p-3 text-sm text-slate-300">Select a node to configure it.</div>
  }

  const d = selection.data
  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-slate-700 p-3">
        <div className="text-sm font-semibold text-slate-100">{d.title}</div>
        <div className="mt-1 text-xs text-slate-400">
          Status: {d.status}
          {d.statusText ? ` - ${d.statusText}` : ''}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-3">
        {selection.data.kind === 'data-source' && <DataPanel cfg={cfg} updateData={updateData} />}
        {selection.data.kind === 'target-discovery' && <TargetsPanel cfg={cfg} updateData={updateData} />}
        {selection.data.kind === 'pathfinding' && <PathfindPanel cfg={cfg} updateData={updateData} />}
        {selection.data.kind === 'feature-engineering' && <FeaturesPanel cfg={cfg} updateData={updateData} />}
        {selection.data.kind === 'output' && <OutputPanel cfg={cfg} updateData={updateData} />}
      </div>
      <div className="border-t border-slate-700 p-3">
        <div className="row-between">
          <button className="btn btn-primary" onClick={onRun}>
            Run Node
          </button>
        </div>
      </div>
    </div>
  )
}
