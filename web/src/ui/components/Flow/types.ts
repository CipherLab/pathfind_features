// Types
export type NodeKind =
  | "data-source"
  | "feature-selection"
  | "target-discovery"
  | "pathfinding"
  | "feature-engineering"
  | "transform"
  | "train"
  | "validate"
  | "output";

export type NodeStatus =
  | "idle"
  | "configured"
  | "running"
  | "complete"
  | "failed"
  | "blocked";

export type NodeData = {
  kind: NodeKind;
  title: string;
  status: NodeStatus;
  statusText?: string;
  // Lightweight config shared across types; extended per-kind via any for now
  config?: any;
};

// Data payload flowing through an edge
export type PayloadType =
  | "PARQUET"
  | "JSON_ARTIFACT"
  | "RELATIONSHIPS"
  | "ENHANCED_DATA"
  | "FINAL_OUTPUT";
