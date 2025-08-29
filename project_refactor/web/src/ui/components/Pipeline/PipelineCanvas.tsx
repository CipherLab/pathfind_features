import React from 'react';
import '@xyflow/react/dist/style.css';
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Node,
  Edge,
  Connection,
  OnNodesChange,
  OnEdgesChange,
} from '@xyflow/react';
import type { NodeTypes, ReactFlowInstance } from '@xyflow/react';
import { NodeData } from '../Flow/types';
import NodeCard from '../Flow/NodeCard';

const nodeTypes: NodeTypes = { appNode: NodeCard as any };

interface PipelineCanvasProps {
  nodes: Node<NodeData>[];
  edges: Edge[];
  onNodesChange: OnNodesChange<Node<NodeData>>;
  onEdgesChange: OnEdgesChange;
  onConnect: (connection: Connection) => void;
  onEdgeDoubleClick: (event: React.MouseEvent, edge: Edge) => void;
  onNodeClick: (event: React.MouseEvent, node: Node<NodeData>) => void;
  onNodesDelete: (nodes: Node<NodeData>[]) => void;
  setRf: (instance: ReactFlowInstance<Node<NodeData>, Edge>) => void;
  onDrop: (event: React.DragEvent) => void;
  onDragOver: (event: React.DragEvent) => void;
}

export function PipelineCanvas({ 
  nodes, 
  edges, 
  onNodesChange, 
  onEdgesChange, 
  onConnect, 
  onEdgeDoubleClick, 
  onNodeClick,
  onNodesDelete,
  setRf,
  onDrop,
  onDragOver,
}: PipelineCanvasProps) {
  return (
    <ReactFlow<Node<NodeData>, Edge>
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onEdgeDoubleClick={onEdgeDoubleClick}
      deleteKeyCode={[ 'Backspace', 'Delete' ]}
      onNodesDelete={onNodesDelete}
      nodeTypes={nodeTypes}
      onNodeClick={onNodeClick}
      onInit={setRf}
      onDrop={onDrop}
      onDragOver={onDragOver}
      fitView
    >
      <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
      <MiniMap pannable zoomable />
      <Controls />
    </ReactFlow>
  );
}
