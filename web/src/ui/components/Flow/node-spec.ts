import { NodeKind, PayloadType } from "./types";
import React from "react";

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
  FINAL_OUTPUT: { color: "#f59e0b", shape: "star" },
};

// Node IO constraints and expected input handles
export const NodeConstraints: Record<
  NodeKind,
  {
    maxInputs: number;
    maxOutputs: number;
    inputs: Array<{
      id: string;
      type: PayloadType;
      label: string;
      required?: boolean;
    }>;
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
    outputs: [{ id: "out-parquet", type: "PARQUET", label: "" }],
    output: { id: "out-parquet", type: "PARQUET", label: "" },
    description: "Provides the raw training parquet to downstream nodes.",
    canConnectTo: [
      "target-discovery",
      "pathfinding",
      "feature-engineering",
      "transform",
    ],
  },
  "feature-selection": {
    maxInputs: 0,
    maxOutputs: 999,
    inputs: [],
    outputs: [{ id: "out-features", type: "JSON_ARTIFACT", label: "" }],
    output: {
      id: "out-features",
      type: "JSON_ARTIFACT",
      label: "features.json",
    },
    description: "Provides a features.json file to downstream nodes.",
    canConnectTo: ["target-discovery"],
  },
  "target-discovery": {
    maxInputs: 2,
    maxOutputs: 999,
    inputs: [
      {
        id: "in-parquet",
        type: "PARQUET",
        label: "",
        required: true,
      },
      {
        id: "in-features",
        type: "JSON_ARTIFACT",
        label: "features.json",
        required: true,
      },
    ],
    outputs: [
      { id: "out-parquet", type: "PARQUET", label: "" },
      {
        id: "out-targets",
        type: "JSON_ARTIFACT",
        label: "",
      },
    ],
    output: {
      id: "out-parquet",
      type: "PARQUET",
      label: "adaptive_targets.parquet",
    },
    description:
      "Discovers candidate targets from the parquet and emits targets.json.",
    canConnectTo: ["pathfinding"],
  },
  pathfinding: {
    maxInputs: 2,
    maxOutputs: 1,
    inputs: [
      {
        id: "in-parquet",
        type: "PARQUET",
        label: "",
        required: true,
      },
      {
        id: "in-artifact",
        type: "JSON_ARTIFACT",
        label: "",
        required: true,
      },
    ],
    outputs: [
      {
        id: "out-relationships",
        type: "RELATIONSHIPS",
        label: "",
      },
    ],
    output: {
      id: "out-relationships",
      type: "RELATIONSHIPS",
      label: "",
    },
    description:
      "Explores relationships between features relative to targets; emits relationships.json.",
    canConnectTo: ["feature-engineering", "pathfinding"],
  },
  "feature-engineering": {
    maxInputs: 2,
    maxOutputs: 1,
    inputs: [
      {
        id: "in-parquet",
        type: "PARQUET",
        label: "",
        required: true,
      },
      {
        id: "in-artifact",
        type: "RELATIONSHIPS",
        label: "",
        required: true,
      },
    ],
    outputs: [
      { id: "out-parquet", type: "PARQUET", label: "enhanced.parquet" },
    ],
    output: {
      id: "out-parquet",
      type: "PARQUET",
      label: "enhanced.parquet",
    },
    description:
      "Generates engineered features using relationships; outputs enhanced.parquet.",
    canConnectTo: ["train", "validate", "output", "transform"],
  },
  transform: {
    maxInputs: 1,
    maxOutputs: 999,
    inputs: [
      {
        id: "in-parquet",
        type: "PARQUET",
        label: "",
        required: true,
      },
    ],
    outputs: [
      { id: "out-parquet", type: "PARQUET", label: "" },
      { id: "out-features", type: "JSON_ARTIFACT", label: "" },
    ],
    output: {
      id: "out-parquet",
      type: "PARQUET",
      label: "",
    },
    description:
      "Applies transformations and emits transformed.parquet; can also derive a features.json artifact.",
    canConnectTo: [
      "train",
      "validate",
      "output",
      "transform",
      "target-discovery",
      "pathfinding",
      "feature-engineering",
    ],
  },
  train: {
    maxInputs: 1,
    maxOutputs: 1,
    inputs: [
      {
        id: "in-parquet",
        type: "PARQUET",
        label: "",
        required: true,
      },
    ],
    outputs: [{ id: "out-model", type: "JSON_ARTIFACT", label: "" }],
    output: { id: "out-model", type: "JSON_ARTIFACT", label: "" },
    description: "Trains a model from a parquet file and outputs model.json.",
    canConnectTo: ["validate"],
  },
  validate: {
    maxInputs: 2,
    maxOutputs: 1,
    inputs: [
      {
        id: "in-parquet",
        type: "PARQUET",
        label: "",
        required: true,
      },
      {
        id: "in-model",
        type: "JSON_ARTIFACT",
        label: "",
        required: true,
      },
    ],
    outputs: [{ id: "out-report", type: "FINAL_OUTPUT", label: "" }],
    output: { id: "out-report", type: "FINAL_OUTPUT", label: "" },
    description: "Validates a trained model and produces a report.json.",
    canConnectTo: [],
  },
  output: {
    maxInputs: 1,
    maxOutputs: 1,
    inputs: [
      {
        id: "in-parquet",
        type: "PARQUET",
        label: "",
        required: true,
      },
    ],
    outputs: [
      {
        id: "out-parquet",
        type: "PARQUET",
        label: "output.parquet",
      },
    ],
    output: {
      id: "out-parquet",
      type: "PARQUET",
      label: "output.parquet",
    },
    description: "Sinks or exports the final dataset.",
    canConnectTo: ["pathfinding", "train", "validate", "output", "transform"],
  },
};

// Descriptive tips shown in the sidebar under Inputs/Outputs
export const NodePanelConfigs: Record<
  NodeKind,
  { inputs?: string[]; outputs?: string[] }
> = {
  "data-source": {
    outputs: [
      "Emits the configured parquet file path to be shared downstream.",
    ],
  },
  "feature-selection": {
    outputs: ["Emits a features.json file chosen from storage."],
  },
  "target-discovery": {
    inputs: [
      "Requires the raw parquet to scan for candidate targets.",
      "Needs a features.json listing candidate features.",
    ],
    outputs: ["Produces a targets.json artifact listing target definitions."],
  },
  pathfinding: {
    inputs: [
      "Needs parquet rows to explore feature interactions.",
      "Consumes targets.json to focus the search.",
    ],
    outputs: [
      "Produces relationships.json describing discovered relationships.",
    ],
  },
  "feature-engineering": {
    inputs: [
      "Consumes base parquet to engineer on top of.",
      "Requires relationships.json describing feature relationships.",
    ],
    outputs: ["Emits enhanced.parquet containing engineered features."],
  },
  transform: {
    inputs: ["Consumes input.parquet to apply transformations."],
    outputs: [
      "Emits transformed.parquet with applied transforms.",
      "Optionally derives a features.json listing resulting features.",
    ],
  },
  train: {
    inputs: ["Needs a parquet file containing features."],
    outputs: ["Produces model.json capturing the trained model."],
  },
  validate: {
    inputs: [
      "Requires a parquet file for evaluation.",
      "Consumes model.json from training.",
    ],
    outputs: ["Emits report.json with validation metrics."],
  },
  output: {
    inputs: ["Consumes a parquet file to export or visualize results."],
  },
};

export const styleFor = (
  payloadType: PayloadType,
  disconnected = false
): React.CSSProperties => {
  const spec = HandleTypes[payloadType];
  if (!spec) {
    return {};
  }
  const style: React.CSSProperties = {
    width: "0.75rem",
    height: "0.75rem",
    backgroundColor: spec.color,
    border: `1px solid ${disconnected ? "#f43f5e" : "#fff"}`,
    display: "inline-block",
  };
  switch (spec.shape) {
    case "circle":
      style.borderRadius = "99px";
      break;
    case "diamond":
      style.transform = "rotate(45deg)";
      break;
    case "triangle":
      style.width = "0";
      style.height = "0";
      style.borderLeft = "6px solid transparent";
      style.borderRight = "6px solid transparent";
      style.borderBottom = `12px solid ${spec.color}`;
      style.backgroundColor = "transparent";
      break;
    case "star":
      style.clipPath =
        "polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%)";
      break;
  }
  return style;
};
