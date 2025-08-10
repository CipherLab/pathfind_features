import { NodeKind, PayloadType } from "./types";

// Visual and semantic handle types
export const HandleTypes: Record<
  PayloadType,
  {
    color: string;
    shape: "square" | "circle" | "diamond" | "star" | "triangle";
  }
> = {
  PARQUET: { color: "#22d3ee", shape: "square" },
  JSON_ARTIFACT: { color: "#a855f7", shape: "circle" },
  RELATIONSHIPS: { color: "#ef4444", shape: "triangle" },
  ENHANCED_DATA: { color: "#10b981", shape: "diamond" },
  FINAL_OUTPUT: { color: "#f59e0b", shape: "star" },
};

// Node IO constraints and expected input handles
export const NodeConstraints: Record<
  NodeKind,
  {
    maxInputs: number;
    maxOutputs: number;
    inputs: Array<{ id: string; type: PayloadType; label: string; required?: boolean }>;
    // plural outputs for sidebar display; keep single `output` for existing components
    outputs: Array<{ id: string; type: PayloadType; label: string }>;
    output?: { id: string; type: PayloadType; label: string };
    description: string;
    // allowed downstream kinds for convenience validation
    canConnectTo: NodeKind[];
  }
> = {
  "data-source": {
    maxInputs: 0,
    maxOutputs: 999,
    inputs: [],
    outputs: [{ id: "out-parquet", type: "PARQUET", label: "train.parquet" }],
    output: { id: "out-parquet", type: "PARQUET", label: "train.parquet" },
    description: "Provides the raw training parquet to downstream nodes.",
    canConnectTo: ["target-discovery", "pathfinding", "feature-engineering"],
  },
  "feature-selection": {
    maxInputs: 0,
    maxOutputs: 999,
    inputs: [],
    outputs: [{ id: "out-features", type: "JSON_ARTIFACT", label: "features.json" }],
    output: { id: "out-features", type: "JSON_ARTIFACT", label: "features.json" },
    description: "Provides a features.json file to downstream nodes.",
    canConnectTo: ["target-discovery"],
  },
  "target-discovery": {
    maxInputs: 2,
    maxOutputs: 999,
    inputs: [
      { id: "in-parquet", type: "PARQUET", label: "input.parquet", required: true },
      { id: "in-features", type: "JSON_ARTIFACT", label: "features.json", required: true },
    ],
    outputs: [
      { id: "out-targets", type: "JSON_ARTIFACT", label: "targets.json" },
    ],
    output: { id: "out-targets", type: "JSON_ARTIFACT", label: "targets.json" },
    description: "Discovers candidate targets from the parquet and emits targets.json.",
    canConnectTo: ["pathfinding"],
  },
  pathfinding: {
    maxInputs: 2,
    maxOutputs: 1,
    inputs: [
      { id: "in-parquet", type: "PARQUET", label: "input.parquet", required: true },
      { id: "in-artifact", type: "JSON_ARTIFACT", label: "targets.json", required: true },
    ],
    outputs: [
      { id: "out-relationships", type: "RELATIONSHIPS", label: "relationships.json" },
    ],
    output: { id: "out-relationships", type: "RELATIONSHIPS", label: "relationships.json" },
    description: "Explores relationships between features relative to targets; emits relationships.json.",
    canConnectTo: ["feature-engineering"],
  },
  "feature-engineering": {
    maxInputs: 2,
    maxOutputs: 1,
    inputs: [
      { id: "in-parquet", type: "PARQUET", label: "input.parquet", required: true },
      { id: "in-artifact", type: "RELATIONSHIPS", label: "relationships.json", required: true },
    ],
    outputs: [
      { id: "out-enhanced", type: "ENHANCED_DATA", label: "enhanced.parquet" },
    ],
    output: { id: "out-enhanced", type: "ENHANCED_DATA", label: "enhanced.parquet" },
    description: "Generates engineered features using relationships; outputs enhanced.parquet.",
    canConnectTo: ["output"],
  },
  output: {
    maxInputs: 1,
    maxOutputs: 0,
    inputs: [
      { id: "in-enhanced", type: "ENHANCED_DATA", label: "enhanced.parquet", required: true },
    ],
    outputs: [],
    // no output
    description: "Sinks or exports the final enhanced dataset.",
    canConnectTo: [],
  },
};

// Descriptive tips shown in the sidebar under Inputs/Outputs
export const NodePanelConfigs: Record<
  NodeKind,
  { inputs?: string[]; outputs?: string[] }
> = {
  'data-source': {
    outputs: ['Emits the configured parquet file path to be shared downstream.'],
  },
  'feature-selection': {
    outputs: ['Emits a features.json file chosen from storage.'],
  },
  'target-discovery': {
    inputs: [
      'Requires the raw parquet to scan for candidate targets.',
      'Needs a features.json listing candidate features.',
    ],
    outputs: ['Produces a targets.json artifact listing target definitions.'],
  },
  pathfinding: {
    inputs: [
      'Needs parquet rows to explore feature interactions.',
      'Consumes targets.json to focus the search.',
    ],
    outputs: ['Produces relationships.json describing discovered relationships.'],
  },
  'feature-engineering': {
    inputs: [
      'Consumes base parquet to engineer on top of.',
      'Requires relationships.json describing feature relationships.',
    ],
    outputs: ['Emits enhanced.parquet containing engineered features.'],
  },
  output: {
    inputs: ['Consumes enhanced.parquet to export or visualize results.'],
  },
};
