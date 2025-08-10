import React, { useCallback, useState } from 'react'
import { Node, Edge } from '@xyflow/react'
import { NodeData, PayloadType } from './types'
import { HandleTypes, NodeConstraints, NodePanelConfigs } from './node-spec'
import DataPanel from './panels/DataPanel'
import TargetsPanel from './panels/TargetsPanel'
import PathfindPanel from './panels/PathfindPanel'
import FeaturesPanel from './panels/FeaturesPanel'
import TransformPanel from './panels/TransformPanel'
import TrainPanel from './panels/TrainPanel'
import ValidatePanel from './panels/ValidatePanel'
import OutputPanel from './panels/OutputPanel'

type Props = {
  selection: Node<NodeData> | null
  edges: Edge[]
  onUpdate: (updater: (prev: NodeData) => NodeData) => void
  onRun: () => void
  onDelete: () => void
}

function InputOutputSection({ selection, edges }: { selection: Node<NodeData>; edges: Edge[] }) {
  if (!selection) return null
  const constraints = NodeConstraints[selection.data.kind]
  const panelConfig = NodePanelConfigs[selection.data.kind]
  return (
    <div className="space-y-4">
      {constraints.inputs.length > 0 && (
        <div className="rounded-md border border-slate-600 bg-slate-800/50 p-3">
          <h4 className="text-sm font-semibold text-blue-300 mb-2">📥 Inputs Required</h4>
          <div className="space-y-2">
            {constraints.inputs.map((input) => {
              const connected = edges.some(e => e.target === selection.id && e.targetHandle === input.id)
              return (
                <div key={input.id} className="flex items-center gap-2 text-xs">
                  <span style={styleFor(input.type, !connected)} />
                  <span className="text-slate-300">{input.label}</span>
                  {!connected && input.required && <span className="text-red-400">required</span>}
                  {!input.required && <span className="text-slate-500">(optional)</span>}
                </div>
              )
            })}
          </div>
          {panelConfig?.inputs && (
            <div className="mt-2 text-xs text-slate-400 border-t border-slate-600 pt-2">
              {panelConfig.inputs.map((desc, idx) => (
                <div key={idx}>• {desc}</div>
              ))}
            </div>
          )}
        </div>
      )}
      {constraints.outputs.length > 0 && (
        <div className="rounded-md border border-slate-600 bg-slate-800/50 p-3">
          <h4 className="text-sm font-semibold text-green-300 mb-2">📤 Outputs Generated</h4>
          <div className="space-y-2">
            {constraints.outputs.map((output) => (
              <div key={output.id} className="flex items-center gap-2 text-xs">
                <span style={styleFor(output.type)} />
                <span className="text-slate-300">{output.label}</span>
              </div>
            ))}
          </div>
          {panelConfig?.outputs && (
            <div className="mt-2 text-xs text-slate-400 border-t border-slate-600 pt-2">
              {panelConfig.outputs.map((desc, idx) => (
                <div key={idx}>• {desc}</div>
              ))}
            </div>
          )}
        </div>
      )}
      <div className="rounded-md border border-slate-600 bg-slate-800/30 p-3">
        <h4 className="text-sm font-semibold text-purple-300 mb-1">ℹ️ What This Node Does</h4>
        <p className="text-xs text-slate-400">{constraints.description}</p>
      </div>
    </div>
  )
}

function LightGBMConfigPanel({ cfg, updateData }: { cfg: any, updateData: (patch: any) => void }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-sm text-slate-300">Boosting Rounds</span>
          <input
            className="input text-sm"
            type="number"
            value={cfg.numBoostRound || 100}
            onChange={e => updateData({ numBoostRound: parseInt(e.target.value || '100', 10) })}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-sm text-slate-300">Learning Rate</span>
          <input
            className="input text-sm"
            type="number"
            step="0.01"
            value={cfg.learningRate || 0.1}
            onChange={e => updateData({ learningRate: parseFloat(e.target.value || '0.1') })}
          />
        </label>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-sm text-slate-300">Max Depth</span>
          <input
            className="input text-sm"
            type="number"
            value={cfg.maxDepth || 6}
            onChange={e => updateData({ maxDepth: parseInt(e.target.value || '6', 10) })}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-sm text-slate-300">Subsample</span>
          <input
            className="input text-sm"
            type="number"
            step="0.1"
            min="0.1"
            max="1.0"
            value={cfg.subsample || 0.8}
            onChange={e => updateData({ subsample: parseFloat(e.target.value || '0.8') })}
          />
        </label>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-sm text-slate-300">Feature Fraction</span>
          <input
            className="input text-sm"
            type="number"
            step="0.1"
            min="0.1"
            max="1.0"
            value={cfg.featureFraction || 0.8}
            onChange={e => updateData({ featureFraction: parseFloat(e.target.value || '0.8') })}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-sm text-slate-300">Early Stopping</span>
          <input
            className="input text-sm"
            type="number"
            value={cfg.earlyStoppingRounds || 10}
            onChange={e => updateData({ earlyStoppingRounds: parseInt(e.target.value || '10', 10) })}
          />
        </label>
      </div>
      <label className="flex flex-col gap-1">
        <span className="text-sm text-slate-300">Random Seed</span>
        <input
          className="input text-sm"
          type="number"
          value={cfg.randomSeed || 42}
          onChange={e => updateData({ randomSeed: parseInt(e.target.value || '42', 10) })}
        />
      </label>
      <div className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3">
        <h5 className="text-sm font-medium text-amber-300 mb-1">💡 Quick Presets</h5>
        <div className="flex gap-2">
          <button 
            className="btn-sm bg-slate-700 hover:bg-slate-600 text-xs"
            onClick={() => updateData({ 
              numBoostRound: 50, learningRate: 0.2, maxDepth: 4, subsample: 0.9, featureFraction: 0.9
            })}
          >
            Fast & Light
          </button>
          <button 
            className="btn-sm bg-slate-700 hover:bg-slate-600 text-xs"
            onClick={() => updateData({ 
              numBoostRound: 500, learningRate: 0.05, maxDepth: 8, subsample: 0.7, featureFraction: 0.7
            })}
          >
            Deep & Thorough
          </button>
        </div>
      </div>
    </div>
  )
}

function ParquetFilterConfigPanel({ cfg, updateData }: { cfg: any, updateData: (patch: any) => void }) {
  return (
    <div className="space-y-4">
      <label className="flex flex-col gap-1">
        <span className="text-sm text-slate-300">Feature Columns (comma-separated)</span>
        <textarea
          className="input text-sm"
          rows={3}
          value={cfg.featureColumns || ''}
          onChange={e => updateData({ featureColumns: e.target.value })}
          placeholder="feature_001, feature_002, target"
        />
      </label>
      <label className="flex flex-col gap-1">
        <span className="text-sm text-slate-300">Row Filter (SQL WHERE syntax)</span>
        <textarea
          className="input text-sm font-mono"
          rows={2}
          value={cfg.rowFilter || ''}
          onChange={e => updateData({ rowFilter: e.target.value })}
          placeholder="era > 100 AND target IS NOT NULL"
        />
      </label>
      <label className="flex flex-col gap-1">
        <span className="text-sm text-slate-300">Sample Fraction</span>
        <div className="flex items-center gap-2">
          <input
            className="input text-sm flex-1"
            type="range"
            min="0.1"
            max="1.0"
            step="0.1"
            value={cfg.sampleFraction || 1.0}
            onChange={e => updateData({ sampleFraction: parseFloat(e.target.value) })}
          />
          <span className="text-xs text-slate-400 w-12">{(cfg.sampleFraction || 1.0) * 100}%</span>
        </div>
      </label>
      <div className="rounded-md border border-blue-500/30 bg-blue-500/10 p-3">
        <h5 className="text-sm font-medium text-blue-300 mb-1">🎯 Common Filters</h5>
        <div className="space-y-1">
          <button 
            className="btn-sm bg-slate-700 hover:bg-slate-600 text-xs w-full text-left"
            onClick={() => updateData({ rowFilter: 'era >= 200' })}
          >
            Recent eras only (era ≥ 200)
          </button>
          <button 
            className="btn-sm bg-slate-700 hover:bg-slate-600 text-xs w-full text-left"
            onClick={() => updateData({ rowFilter: 'target IS NOT NULL' })}
          >
            Remove rows with missing targets
          </button>
        </div>
      </div>
    </div>
  )
}

export default function Sidebar({ selection, edges, onUpdate, onRun, onDelete }: Props) {
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
        onUpdate(p => ({ ...p, config: next, status: 'configured', statusText: '' }))
        return next
      })
    },
    [onUpdate]
  )
  if (!selection) {
    return (
      <div className="h-full p-4 text-center">
        <div className="text-slate-400 text-sm">Select a node to configure it</div>
        <div className="text-xs text-slate-500 mt-2">
          Pro tip: The sidebar actually tells you what each node does now!
        </div>
      </div>
    )
  }
  const d = selection.data
  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-slate-700 p-4">
        <div className="text-lg font-semibold text-slate-100">{d.title}</div>
        <div className="mt-1 flex items-center gap-2">
          <span className={`inline-block h-2 w-2 rounded-full ${
            d.status === 'idle' ? 'bg-slate-400' :
            d.status === 'configured' ? 'bg-green-400' :
            d.status === 'running' ? 'bg-cyan-400' :
            d.status === 'complete' ? 'bg-violet-400' :
            d.status === 'failed' ? 'bg-red-400' : 'bg-amber-400'
          }`} />
          <span className="text-xs text-slate-400">
            {d.status} {d.statusText ? ` - ${d.statusText}` : ''}
          </span>
        </div>
      </div>
      <div className="border-b border-slate-700 p-4">
        <InputOutputSection selection={selection} edges={edges} />
      </div>
      <div className="flex-1 overflow-auto p-4">
        <h3 className="text-sm font-semibold text-slate-200 mb-3">⚙️ Configuration</h3>
        {selection.data.kind === 'data-source' && (
          <DataPanel cfg={cfg} updateData={updateData} />
        )}
        {selection.data.kind === 'target-discovery' && (
          <TargetsPanel cfg={cfg} updateData={updateData} />
        )}
        {selection.data.kind === 'feature-selection' && (
          <FeatureSourcePanel cfg={cfg} updateData={updateData} />
        )}
        {selection.data.kind === 'pathfinding' && (
          <PathfindPanel cfg={cfg} updateData={updateData} />
        )}
        {selection.data.kind === 'feature-engineering' && (
          <FeaturesPanel cfg={cfg} updateData={updateData} />
        )}
        {selection.data.kind === 'transform' && (
          <TransformPanel cfg={cfg} updateData={updateData} />
        )}
        {selection.data.kind === 'train' && (
          <TrainPanel cfg={cfg} updateData={updateData} />
        )}
        {selection.data.kind === 'validate' && (
          <ValidatePanel cfg={cfg} updateData={updateData} />
        )}
        {selection.data.kind === 'output' && (
          <OutputPanel cfg={cfg} updateData={updateData} />
        )}
      </div>
      <div className="border-t border-slate-700 p-4">
        <div className="flex gap-2">
          <button
            className="btn btn-primary flex-1"
            onClick={onRun}
            disabled={d.status === 'running'}
          >
            {d.status === 'running' ? 'Running...' : 'Run Node'}
          </button>
          <button className="btn bg-red-700 hover:bg-red-800" onClick={onDelete}>
            Delete
          </button>
        </div>
        {d.status === 'blocked' && (
          <div className="mt-2 text-xs text-amber-400">
            ⚠️ This node is blocked. Check that all required inputs are connected
            and upstream nodes are complete.
          </div>
        )}
      </div>
    </div>
  )
}
