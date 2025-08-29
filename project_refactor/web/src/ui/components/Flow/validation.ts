import { Connection, Edge, Node } from "@xyflow/react";
import { NodeData, NodeKind, PayloadType } from "./types";
import { NodeConstraints } from "./node-spec";

// Validate node-kind pairing (broader than strict adjacency; supports multi-input nodes)
export const allowsConnection = (a: NodeKind, b: NodeKind) =>
  NodeConstraints[a].canConnectTo.includes(b);

// Validate a potential connection between two nodes. Used both during drag previews
// and when establishing a final connection. Only allow edges that follow the
// canonical Data Source → Target Discovery → Pathfinding → Feature Engineering → Output order.
export function isValidConnection(
  conn: Connection,
  ns: Node<NodeData>[],
  es: Edge[]
) {
  if (!conn.source || !conn.target) return false;
  const src = ns.find((n) => n.id === conn.source);
  const tgt = ns.find((n) => n.id === conn.target);
  if (!src || !tgt) return false;
  if (!allowsConnection(src.data.kind, tgt.data.kind)) return false;

  const sCons = NodeConstraints[src.data.kind];
  const tCons = NodeConstraints[tgt.data.kind];

  // Cardinality: outputs
  const outCount = es.filter((e) => e.source === src.id).length;
  if (sCons.maxOutputs !== 999 && outCount >= sCons.maxOutputs) return false;

  // Cardinality: inputs
  const inCount = es.filter((e) => e.target === tgt.id).length;
  if (inCount >= tCons.maxInputs) return false;

  // Type compatibility via handle ids
  // Determine payload type for the specific source handle when multiple outputs exist
  const sourceHandleId = conn.sourceHandle || "";
  let srcPayload: PayloadType | undefined = undefined;
  if (src.data.kind === "file-source") {
    srcPayload = (src.data.config?.payloadType as PayloadType) || undefined;
  } else {
    if (sourceHandleId && sCons.outputs && sCons.outputs.length > 0) {
      const out = sCons.outputs.find((o) => o.id === sourceHandleId);
      srcPayload = out?.type;
    }
    if (!srcPayload) {
      srcPayload =
        sCons.output?.type || (sCons.outputs && sCons.outputs[0]?.type);
    }
  }
  const targetHandleId = conn.targetHandle || "";
  const expected = tCons.inputs.find((i) => i.id === targetHandleId);
  if (expected && srcPayload && expected.type !== srcPayload) return false;

  // Per-handle uniqueness (one connection per input handle)
  if (targetHandleId) {
    const existingOnHandle = es.some(
      (e) => e.target === tgt.id && e.targetHandle === targetHandleId
    );
    if (existingOnHandle) return false;
  }
  return true;
}

export function topoSort(ns: Node<NodeData>[], es: Edge[]): string[] | null {
  const inDeg = new Map<string, number>();
  const adj = new Map<string, string[]>();
  ns.forEach((n) => {
    inDeg.set(n.id, 0);
    adj.set(n.id, []);
  });
  es.forEach((e) => {
    if (inDeg.has(e.target) && adj.has(e.source)) {
      inDeg.set(e.target, (inDeg.get(e.target) || 0) + 1);
      adj.get(e.source)!.push(e.target);
    }
  });
  const q: string[] = [];
  inDeg.forEach((deg, id) => {
    if (deg === 0) q.push(id);
  });
  const order: string[] = [];
  while (q.length) {
    const id = q.shift()!;
    order.push(id);
    adj.get(id)!.forEach((t) => {
      const nd = (inDeg.get(t) || 0) - 1;
      inDeg.set(t, nd);
      if (nd === 0) q.push(t);
    });
  }
  if (order.length !== ns.length) return null;
  return order;
}

export function validatePipeline(
  ns: Node<NodeData>[],
  es: Edge[]
): string[] | null {
  if (ns.length === 0) {
    alert("Add some nodes first");
    return null;
  }
  for (const n of ns) {
    if (n.data.status === "idle") {
      alert(`Node "${n.data.title}" is not configured`);
      return null;
    }
  }
  const order = topoSort(ns, es);
  if (!order) {
    alert("Pipeline has cycles or disconnected nodes");
    return null;
  }
  return order;
}
