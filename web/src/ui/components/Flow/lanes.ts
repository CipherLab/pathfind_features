import { Edge, Node } from "@xyflow/react";
import { jpost } from "../../lib/api";
import { NodeData } from "./types";

export type ExecutionLane = { index: number; nodes: string[] };

export async function planLanes(
  nodes: Node<NodeData>[],
  edges: Edge[]
): Promise<{
  lanes: ExecutionLane[];
  order: string[];
  hasCycle: boolean;
}> {
  const payload = {
    nodes: nodes.map((n) => ({ id: n.id, kind: n.data.kind })),
    edges: edges.map((e) => ({ source: e.source, target: e.target })),
  };
  const res = await jpost("/pipeline/plan", payload);
  return { lanes: res.lanes, order: res.order, hasCycle: res.has_cycle };
}

export function snapXForLane(
  index: number,
  baseX = 80,
  laneWidth = 260
): number {
  return baseX + index * laneWidth;
}
