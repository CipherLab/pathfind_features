import { NodeKind, PayloadType } from './types'

// Visual and semantic handle types
export const HandleTypes: Record<PayloadType, { color: string; shape: 'square' | 'circle' | 'diamond' | 'star' }> = {
  PARQUET: { color: '#22d3ee', shape: 'square' },
  JSON_ARTIFACT: { color: '#a855f7', shape: 'circle' },
  ENHANCED_DATA: { color: '#10b981', shape: 'diamond' },
  FINAL_OUTPUT: { color: '#f59e0b', shape: 'star' },
}

// Node IO constraints and expected input handles
export const NodeConstraints: Record<
  NodeKind,
  {
    maxInputs: number
    maxOutputs: number
    inputs: Array<{ id: string; type: PayloadType; label: string }>
    output?: { id: string; type: PayloadType; label: string }
    // allowed downstream kinds for convenience validation
    canConnectTo: NodeKind[]
  }
> = {
  'data-source': {
    maxInputs: 0,
    maxOutputs: 999,
    inputs: [],
    output: { id: 'out-parquet', type: 'PARQUET', label: 'train.parquet' },
    canConnectTo: ['target-discovery', 'pathfinding', 'feature-engineering'],
  },
  'target-discovery': {
    maxInputs: 1,
    maxOutputs: 999,
    inputs: [{ id: 'in-parquet', type: 'PARQUET', label: 'input.parquet' }],
    output: { id: 'out-targets', type: 'JSON_ARTIFACT', label: 'targets.json' },
    canConnectTo: ['pathfinding'],
  },
  pathfinding: {
    maxInputs: 2,
    maxOutputs: 1,
    inputs: [
      { id: 'in-parquet', type: 'PARQUET', label: 'input.parquet' },
      { id: 'in-artifact', type: 'JSON_ARTIFACT', label: 'targets.json' },
    ],
    output: { id: 'out-relationships', type: 'JSON_ARTIFACT', label: 'relationships.json' },
    canConnectTo: ['feature-engineering'],
  },
  'feature-engineering': {
    maxInputs: 2,
    maxOutputs: 1,
    inputs: [
      { id: 'in-parquet', type: 'PARQUET', label: 'input.parquet' },
      { id: 'in-artifact', type: 'JSON_ARTIFACT', label: 'relationships.json' },
    ],
    output: { id: 'out-enhanced', type: 'ENHANCED_DATA', label: 'enhanced.parquet' },
    canConnectTo: ['output'],
  },
  output: {
    maxInputs: 1,
    maxOutputs: 0,
    inputs: [{ id: 'in-enhanced', type: 'ENHANCED_DATA', label: 'enhanced.parquet' }],
    // no output
    canConnectTo: [],
  },
}
