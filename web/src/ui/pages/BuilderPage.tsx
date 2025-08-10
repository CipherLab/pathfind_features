import * as React from 'react';
import { useState, useCallback } from 'react';
import type { ReactFlowInstance } from '@xyflow/react';
import { Node } from '@xyflow/react';
import { usePipelineState } from '../hooks/usePipelineState';
import { usePipelineRunner } from '../hooks/usePipelineRunner';
import { PipelineCanvas } from '../components/Pipeline/PipelineCanvas';
import Sidebar from '../components/Flow/Sidebar';
import NodePalette from '../components/Flow/NodePalette';
import PipelineToolbar from '../components/Flow/PipelineToolbar';
import { NodeData, NodeKind } from '../components/Flow/types';

export default function BuilderPage() {
  const {
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
    setSelection,
  } = usePipelineState();

  const {
    progress,
    runNode,
    onRunPipeline,
  } = usePipelineRunner(nodes, edges, setNodes);

  const [rf, setRf] = useState<ReactFlowInstance<Node<NodeData>, Edge> | null>(null);

  const onRunNode = useCallback(async () => {
    if (!selection) return;
    await runNode(selection);
  }, [selection, runNode]);

  const onClear = useCallback(() => {
    setNodes([]);
    setSelection(null);
  }, [setNodes, setSelection]);

  const onNodesDelete = useCallback(() => {
    if (selection) {
      deleteSelection();
    }
  }, [selection, deleteSelection]);

  const onNodeClick = useCallback((_: React.MouseEvent, n: Node<NodeData>) => {
    setSelection(n);
  }, [setSelection]);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const kind = event.dataTransfer.getData('application/reactflow') as NodeKind;
      if (!kind || !rf) return;
      const pos = rf.screenToFlowPosition({ x: event.clientX, y: event.clientY });
      addNode(kind, pos);
    },
    [rf, addNode]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  return (
    <div className="flex h-[calc(100vh-120px)] gap-2">
      <div className="w-56 shrink-0">
        <NodePalette onAdd={addNode} />
      </div>
      <div className="flex min-w-0 flex-1 flex-col rounded-lg border border-slate-700 bg-slate-900/40">
        <PipelineToolbar onRunPipeline={onRunPipeline} onClear={onClear} progress={progress} />
        <div className="relative min-h-0 flex-1">
          <PipelineCanvas
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onEdgeDoubleClick={onEdgeDoubleClick}
            onNodeClick={onNodeClick}
            onNodesDelete={onNodesDelete}
            setRf={setRf}
            onDrop={onDrop}
            onDragOver={onDragOver}
          />
        </div>
      </div>
      <div className="w-[420px] shrink-0 rounded-lg border border-slate-700 bg-slate-900/60">
        <Sidebar
          selection={selection}
          edges={edges}
          onUpdate={onUpdateSelection}
          onRun={onRunNode}
          onDelete={deleteSelection}
        />
      </div>
    </div>
  );
}
