// Types
export type NodeKind =
  | "data-source"
  | "file-source"
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
  | "pending"
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
  | "ADAPTIVE_TARGETS_PARQUET"
  | "TARGET_DISCOVERY_JSON"
  // New, distinct types for Target Discovery → Pathfinding handoff
  | "TD_CANDIDATE_TARGETS"
  | "TD_DISCOVERY_META"
  | "ENHANCED_DATA"
  | "FINAL_OUTPUT";
