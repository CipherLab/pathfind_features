import * as React from 'react';
import { useState, useCallback, useEffect } from 'react';
import type { ReactFlowInstance } from '@xyflow/react';
import { Node, Edge } from '@xyflow/react';
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
    setEdges,
    setSelection,
  } = usePipelineState();

  const [experimentName, setExperimentName] = useState('my_experiment');
  const [seed, setSeed] = useState(42);

  const {
    progress,
    runNode,
    onRunPipeline,
  } = usePipelineRunner(nodes, edges, setNodes, setEdges, experimentName, seed);

  const [rf, setRf] = useState<ReactFlowInstance<Node<NodeData>, Edge> | null>(null);

  useEffect(() => {
    if (selection) {
      const selectedNode = nodes.find(n => n.id === selection.id);
      if (selectedNode) {
        setSelection(selectedNode);
      }
    }
  }, [nodes, selection, setSelection]);

  const onRunNode = useCallback(async () => {
    if (!selection) return;
    await runNode(selection);
  }, [selection, runNode]);

  const onClear = useCallback(() => {
    setNodes([]);
    setEdges([]);
    setSelection(null);
  }, [setNodes, setEdges, setSelection]);

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

  const onSave = useCallback(() => {
    const state = { nodes, edges, experimentName, seed };
    const json = JSON.stringify(state, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${experimentName}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [nodes, edges, experimentName, seed]);

  const onLoad = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result;
        if (typeof content === 'string') {
          const { nodes: loadedNodes, edges: loadedEdges, experimentName: loadedExperimentName, seed: loadedSeed } = JSON.parse(content);
          setNodes(loadedNodes || []);
          setEdges(loadedEdges || []);
          setExperimentName(loadedExperimentName || 'my_experiment');
          setSeed(loadedSeed || 42);
        }
      };
      reader.readAsText(file);
    }
  }, [setNodes, setEdges, setExperimentName, setSeed]);

  return (
    <div className="flex h-[calc(100vh-120px)] gap-2">
      <div className="w-56 shrink-0">
        <NodePalette onAdd={addNode} />
      </div>
      <div className="flex min-w-0 flex-1 flex-col rounded-lg border border-slate-700 bg-slate-900/40">
        <PipelineToolbar 
          experimentName={experimentName}
          onExperimentNameChange={setExperimentName}
          seed={seed}
          onSeedChange={setSeed}
          onRunPipeline={onRunPipeline} 
          onClear={onClear} 
          onSave={onSave} 
          onLoad={onLoad} 
          progress={progress} 
        />
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