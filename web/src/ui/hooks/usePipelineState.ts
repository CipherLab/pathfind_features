import { useState, useCallback, useRef, useEffect } from 'react';
import { useNodesState, useEdgesState, addEdge, Connection, Edge, Node } from '@xyflow/react';
import { NodeData, NodeKind } from '../components/Flow/types';
import { isValidConnection } from '../components/Flow/validation';
import { HandleTypes, NodeConstraints } from '../components/Flow/node-spec';

const LOCAL_STORAGE_KEY = 'pipelineState';

export function usePipelineState() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<NodeData>>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selection, setSelection] = useState<Node<NodeData> | null>(null);
  const idRef = useRef(1);

  useEffect(() => {
    const storedState = localStorage.getItem(LOCAL_STORAGE_KEY);
    if (storedState) {
      const { nodes: storedNodes, edges: storedEdges } = JSON.parse(storedState);
      setNodes(storedNodes || []);
      setEdges(storedEdges || []);
    }
  }, [setNodes, setEdges]);

  useEffect(() => {
    if (nodes.length > 0 || edges.length > 0) {
      const state = { nodes, edges };
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(state));
    }
  }, [nodes, edges]);

  const onConnect = useCallback(
    (conn: Connection) => {
      const source = nodes.find((n: Node<NodeData>) => n.id === conn.source);
      const target = nodes.find((n: Node<NodeData>) => n.id === conn.target);
      if (!source || !target) return;
      if (!isValidConnection(conn, nodes, edges)) return;

      const vis = NodeConstraints[source.data.kind].output;
      const color = vis ? HandleTypes[vis.type].color : '#64748b';
      const label = vis?.label || '';

      setEdges(eds =>
        addEdge(
          {
            ...conn,
            type: 'smoothstep',
            data: { payloadType: vis?.type },
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
      const id = `n${idRef.current++}`;
      const title =
        kind === 'data-source'
          ? 'Data Source'
          : kind === 'feature-selection'
          ? 'Features'
          : kind === 'target-discovery'
          ? 'Target Discovery'
          : kind === 'pathfinding'
          ? 'Pathfinding'
          : kind === 'feature-engineering'
          ? 'Feature Engineering'
          : kind === 'transform'
          ? 'Transform'
          : kind === 'train'
          ? 'Train'
          : kind === 'validate'
          ? 'Validate'
          : 'Output';

      const n: Node<NodeData> = {
        id,
        type: 'appNode',
        position: position ?? { x: 140 + nodes.length * 50, y: 100 + nodes.length * 20 },
        data: { kind, title, status: 'idle', statusText: '', config: {} },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      };
      setNodes((ns: Node<NodeData>[]) => [...ns, n]);
      setSelection(n);
    },
    [nodes.length, setNodes]
  );

  const deleteSelection = useCallback(() => {
    if (!selection) return;
    setNodes(ns => ns.filter(n => n.id !== selection.id));
    setEdges(es => es.filter(e => e.source !== selection.id && e.target !== selection.id));
    setSelection(null);
  }, [selection, setNodes, setEdges]);

  const onEdgeDoubleClick = useCallback(
    (_: React.MouseEvent, edge: Edge) => {
      setEdges(es => es.filter(e => e.id !== edge.id));
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
  };
}