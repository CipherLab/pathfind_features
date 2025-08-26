import { useState, useCallback, useRef, useEffect } from "react";
import {
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  Node,
  Position,
} from "@xyflow/react";
import { NodeData, NodeKind } from "../components/Flow/types";
import { isValidConnection } from "../components/Flow/validation";
import { HandleTypes, NodeConstraints } from "../components/Flow/node-spec";

const LOCAL_STORAGE_KEY = "pipelineState";

export function usePipelineState() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<NodeData>>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selection, setSelection] = useState<Node<NodeData> | null>(null);
  const idRef = useRef(1);

  // Helper to compute next id base from existing node ids like "n123".
  const syncIdCounterWithNodes = useCallback((list: Node<NodeData>[]) => {
    let maxNum = 0;
    for (const n of list) {
      const m = /^n(?:[a-z0-9]+_)?(\d+)$/.exec(n.id);
      if (m) {
        const v = parseInt(m[1], 10);
        if (!Number.isNaN(v)) maxNum = Math.max(maxNum, v);
      }
    }
    // Always move forward, never backward
    idRef.current = Math.max(idRef.current, maxNum + 1);
  }, []);

  // Robust id generator: time-based prefix + monotonic counter avoids collisions after reloads
  const nextNodeId = useCallback(() => {
    const ts = Date.now().toString(36);
    const id = `n${ts}_${idRef.current++}`;
    return id;
  }, []);

  useEffect(() => {
    const storedState = localStorage.getItem(LOCAL_STORAGE_KEY);
    if (storedState) {
      const { nodes: storedNodes, edges: storedEdges } =
        JSON.parse(storedState);
      setNodes(storedNodes || []);
      setEdges(storedEdges || []);
      if (storedNodes && Array.isArray(storedNodes)) {
        syncIdCounterWithNodes(storedNodes as Node<NodeData>[]);
      }
    }
  }, [setNodes, setEdges, syncIdCounterWithNodes]);

  useEffect(() => {
    if (nodes.length > 0 || edges.length > 0) {
      const state = { nodes, edges };
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(state));
    }
  }, [nodes, edges]);

  // Whenever nodes change externally (e.g., load/import), keep id counter ahead
  useEffect(() => {
    if (nodes && nodes.length) {
      syncIdCounterWithNodes(nodes);
    }
  }, [nodes, syncIdCounterWithNodes]);

  const onConnect = useCallback(
    (conn: Connection) => {
      const source = nodes.find((n: Node<NodeData>) => n.id === conn.source);
      const target = nodes.find((n: Node<NodeData>) => n.id === conn.target);
      if (!source || !target) return;
      if (!isValidConnection(conn, nodes, edges)) return;

      // Determine which source handle is being used and pick its label/type.
      // For file-source, override by config.payloadType.
      const spec = NodeConstraints[source.data.kind];
      const outDefs =
        spec.outputs && spec.outputs.length
          ? spec.outputs
          : spec.output
          ? [spec.output]
          : [];
      let chosen =
        outDefs.find((o) => o.id === conn.sourceHandle) || outDefs[0];
      if (source.data.kind === "file-source" && chosen) {
        const pt = (source.data.config?.payloadType as any) || chosen.type;
        chosen = { ...chosen, type: pt } as any;
      }
      const color = chosen ? HandleTypes[chosen.type].color : "#64748b";
      const label = chosen?.label || "";

      setEdges(
        (eds) =>
          addEdge(
            {
              ...conn,
              type: "smoothstep",
              data: {
                payloadType: chosen?.type,
                sourceHandle: conn.sourceHandle,
                targetHandle: conn.targetHandle,
              },
              label,
              labelStyle: { fill: color, fontSize: 10 },
              style: { stroke: color, strokeWidth: 2 },
            } as any,
            eds as any
          ) as any
      );
    },
    [nodes, edges, setEdges]
  );

  const addNode = useCallback(
    (kind: NodeKind, position?: { x: number; y: number }) => {
      const id = nextNodeId();
      const title =
        kind === "data-source"
          ? "Data Source"
          : kind === "file-source"
          ? "File Source"
          : kind === "feature-selection"
          ? "Features"
          : kind === "target-discovery"
          ? "Target Discovery"
          : kind === "pathfinding"
          ? "Pathfinding"
          : kind === "feature-engineering"
          ? "Feature Engineering"
          : kind === "transform"
          ? "Transform"
          : kind === "train"
          ? "Train"
          : kind === "validate"
          ? "Validate"
          : "Output";

      const defaultConfig: Record<string, any> =
        kind === "pathfinding"
          ? {
              // Sensible defaults
              lastNEras: 200,
              smokeFeat: 8,
              cacheDir: "cache/pathfinding_cache",
              sanityCheck: true,
            }
          : kind === "file-source"
          ? {
              payloadType: "PARQUET",
              inputPath: "v5.0/train.parquet",
            }
          : {};

      const n: Node<NodeData> = {
        id,
        type: "appNode",
        position: position ?? {
          x: 140 + nodes.length * 50,
          y: 100 + nodes.length * 20,
        },
        data: {
          kind,
          title,
          status: "idle",
          statusText: "",
          config: defaultConfig,
        },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      };
      setNodes((ns: Node<NodeData>[]) => [...ns, n]);
      setSelection(n);
    },
    [nodes.length, setNodes, nextNodeId]
  );

  const deleteSelection = useCallback(() => {
    if (!selection) return;
    setNodes((ns) => ns.filter((n) => n.id !== selection.id));
    setEdges((es) =>
      es.filter((e) => e.source !== selection.id && e.target !== selection.id)
    );
    setSelection(null);
  }, [selection, setNodes, setEdges]);

  const onEdgeDoubleClick = useCallback(
    (_: React.MouseEvent, edge: Edge) => {
      setEdges((es) => es.filter((e) => e.id !== edge.id));
    },
    [setEdges]
  );

  const onUpdateSelection = useCallback(
    (updater: (prev: NodeData) => NodeData) => {
      if (!selection) return;
      setNodes((ns: Node<NodeData>[]) =>
        ns.map((n: Node<NodeData>) =>
          n.id === selection.id ? { ...n, data: updater(n.data) } : n
        )
      );
    },
    [selection, setNodes]
  );

  const getUpstreamNode = useCallback(
    (nodeId: string, handleId?: string) => {
      const edge = edges.find(
        (e) => e.target === nodeId && e.targetHandle === handleId
      );
      if (!edge) return null;
      return nodes.find((n) => n.id === edge.source) || null;
    },
    [edges, nodes]
  );

  // Utility: get relevant upstream cap for a node (by kind)
  const getUpstreamCap = useCallback(
    (
      node: Node<NodeData>,
      capType: "features" | "targets" | "relationships"
    ) => {
      // For pathfinding, capType 'features' comes from target-discovery's smokeTargets
      // For feature-engineering, capType 'relationships' comes from pathfinding's maxNew
      if (!node) return 0;
      if (node.data.kind === "pathfinding" && capType === "features") {
        // Find upstream target-discovery node
        const edge = edges.find(
          (e) => e.target === node.id && e.targetHandle === "in-parquet"
        );
        if (!edge) return 0;
        const upstream = nodes.find((n) => n.id === edge.source);
        return upstream?.data?.config?.smokeTargets || 0;
      }
      if (
        node.data.kind === "feature-engineering" &&
        capType === "relationships"
      ) {
        // Find upstream pathfinding node
        const edge = edges.find(
          (e) => e.target === node.id && e.targetHandle === "in-artifact"
        );
        if (!edge) return 0;
        const upstream = nodes.find((n) => n.id === edge.source);
        return upstream?.data?.config?.maxNew || 0;
      }
      return 0;
    },
    [edges, nodes]
  );

  // Auto-clamp downstream config if upstream cap decreases
  useEffect(() => {
    setNodes((ns) =>
      ns.map((n) => {
        if (n.data.kind === "pathfinding") {
          const cap = getUpstreamCap(n, "features");
          if (cap > 0 && n.data.config?.smokeFeat > cap) {
            return {
              ...n,
              data: { ...n.data, config: { ...n.data.config, smokeFeat: cap } },
            };
          }
        }
        if (n.data.kind === "feature-engineering") {
          const cap = getUpstreamCap(n, "relationships");
          if (cap > 0 && n.data.config?.maxNew > cap) {
            return {
              ...n,
              data: { ...n.data, config: { ...n.data.config, maxNew: cap } },
            };
          }
        }
        return n;
      })
    );
  }, [edges, nodes, getUpstreamCap, setNodes]);

  return {
    nodes,
    edges,
    selection,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode,
    deleteSelection,
    onEdgeDoubleClick,
    onUpdateSelection,
    setNodes,
    setEdges,
    setSelection,
    getUpstreamCap,
  };
}
