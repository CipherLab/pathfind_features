import { useState, useCallback, useEffect } from 'react';
import { Node, Edge } from '@xyflow/react';
import { jpost } from '../lib/api';
import { NodeData, NodeStatus } from '../components/Flow/types';
import { validatePipeline } from '../components/Flow/validation';
import { NodeConstraints } from '../components/Flow/node-spec';

export function usePipelineRunner(nodes: Node<NodeData>[], edges: Edge[], setNodes: (nodes: Node<NodeData>[] | ((nodes: Node<NodeData>[]) => Node<NodeData>[])) => void) {
  const [progress, setProgress] = useState({ total: 0, completed: 0 });

  const propagateArtifacts = useCallback(
    (src: Node<NodeData>) => {
      setNodes((ns: Node<NodeData>[]) => {
        let updated = ns;
        if (src.data.kind === 'target-discovery') {
          const downstream = edges.filter(e => e.source === src.id).map(e => e.target);
          updated = ns.map(n => {
            if (downstream.includes(n.id) && n.data.kind === 'pathfinding') {
              const cfg = {
                ...n.data.config,
                inheritTargetsFrom: src.id,
                targetsJson: 'target_discovery.json',
              };
              const status = (n.data.status === 'idle' ? 'configured' : n.data.status) as NodeStatus;
              return { ...n, data: { ...n.data, config: cfg, status } };
            }
            return n;
          });
        } else if (src.data.kind === 'pathfinding') {
          const downstream = edges.filter(e => e.source === src.id).map(e => e.target);
          updated = ns.map(n => {
            if (downstream.includes(n.id) && n.data.kind === 'feature-engineering') {
              const cfg = {
                ...n.data.config,
                inheritRelationshipsFrom: src.id,
                relationshipsJson: 'relationships.json',
              };
              const status = (n.data.status === 'idle' ? 'configured' : n.data.status) as NodeStatus;
              return { ...n, data: { ...n.data, config: cfg, status } };
            }
            return n;
          });
        } else if (src.data.kind === 'feature-selection') {
          const downstream = edges.filter(e => e.source === src.id).map(e => e.target);
          updated = ns.map(n => {
            if (downstream.includes(n.id) && n.data.kind === 'target-discovery') {
              const cfg = {
                ...n.data.config,
                inheritFeaturesFrom: src.id,
                featuresJson: src.data.config?.featuresJson,
              };
              const status = (n.data.status === 'idle' ? 'configured' : n.data.status) as NodeStatus;
              return { ...n, data: { ...n.data, config: cfg, status } };
            }
            return n;
          });
        } else if (src.data.kind === 'transform') {
          const downstream = edges.filter(e => e.source === src.id).map(e => e.target);
          updated = ns.map(n => {
            if (downstream.includes(n.id)) {
              const cfg = {
                ...n.data.config,
                inputData: `pipeline_runs/transformed_data_${src.id}.parquet`,
              };
              const status = (n.data.status === 'idle' ? 'configured' : n.data.status) as NodeStatus;
              return { ...n, data: { ...n.data, config: cfg, status } };
            }
            return n;
          });
        }
        return updated;
      });
    },
    [edges, setNodes]
  );

  const runNode = useCallback(
    async (node: Node<NodeData>) => {
      const { id, data } = node;
      const cfg = data.config || {};

      setNodes(ns =>
        ns.map(n =>
          n.id === id
            ? {
                ...n,
                data: {
                  ...n.data,
                  status: 'running' as NodeStatus,
                  statusText:
                    data.kind === 'target-discovery'
                      ? 'Running target discovery...'
                      : data.kind === 'feature-selection'
                      ? 'Loading features...'
                      : data.kind === 'pathfinding'
                      ? 'Exploring relationships...'
                      : data.kind === 'feature-engineering'
                      ? 'Brewing features...'
                      : data.kind === 'transform'
                      ? 'Applying transform...'
                      : 'Working...',
                },
              }
            : n
        )
      );

      try {
        if (data.kind === 'target-discovery') {
          const payload = {
            input_data: cfg.inputData || 'v5.0/train.parquet',
            features_json: cfg.featuresJson || 'v5.0/features.json',
            run_name: cfg.runName || 'wizard',
            max_new_features: cfg.maxNew ?? 8,
            disable_pathfinding: true,
            pretty: cfg.pretty ?? true,
            smoke_mode: cfg.smoke ?? true,
            smoke_max_eras: cfg.smokeEras,
            smoke_row_limit: cfg.smokeRows,
            smoke_feature_limit: cfg.smokeFeat,
            seed: cfg.seed ?? 42,
          };
          await jpost('/runs', payload);
        } else if (data.kind === 'transform') {
          const inputEdge = edges.find(e => e.target === id);
          if (!inputEdge) {
            throw new Error("Transform node requires an input connection");
          }
          const inputNode = nodes.find(n => n.id === inputEdge.source);
          if (!inputNode) {
            throw new Error("Input node not found for transform");
          }

          const inputDataPath = "v5.0/train.parquet";

          const payload = {
            input_data: inputDataPath,
            transform_script: cfg.script,
            output_data: `pipeline_runs/transformed_data_${id}.parquet`,
          };
          const result = await jpost('/transforms/execute', payload);
          setNodes(ns =>
            ns.map(n =>
              n.id === id
                ? {
                    ...n,
                    data: {
                      ...n.data,
                      config: { ...n.data.config, outputPath: result.output },
                    },
                  }
                : n
            )
          );
        } else {
          await new Promise(res => setTimeout(res, 300));
        }

        setNodes(ns =>
          ns.map(n =>
            n.id === id
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    status: 'complete' as NodeStatus,
                    statusText:
                      data.kind === 'target-discovery'
                        ? '✅ Targets discovered'
                        : data.kind === 'feature-selection'
                        ? '✅ Features selected'
                        : data.kind === 'pathfinding'
                        ? '✅ Relationships mapped'
                        : data.kind === 'feature-engineering'
                        ? '✅ Features generated'
                        : data.kind === 'transform'
                        ? '✅ Transform applied'
                        : '✅ Done',
                  },
                }
              : n
          )
        );

        if (
          data.kind === 'target-discovery' ||
          data.kind === 'pathfinding' ||
          data.kind === 'feature-selection' ||
          data.kind === 'transform'
        ) {
          propagateArtifacts(node);
        }
      } catch (e: any) {
        const msg =
          e && typeof e === 'object' && 'message' in e ? String(e.message) : 'Unknown error';
        setNodes(ns =>
          ns.map(n =>
            n.id === id
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    status: 'failed' as NodeStatus,
                    statusText: `❌ ${msg}`,
                  },
                }
              : n
          )
        );
        throw e;
      }
    },
    [nodes, edges, setNodes, propagateArtifacts]
  );

  const onRunPipeline = useCallback(async () => {
    const order = validatePipeline(nodes, edges);
    if (!order) return;
    setProgress({ total: order.length, completed: 0 });
    const map = new Map(nodes.map(n => [n.id, n]));
    for (const id of order) {
      const node = map.get(id);
      if (!node) continue;
      try {
        await runNode(node);
        setProgress(p => ({ total: p.total, completed: p.completed + 1 }));
      } catch {
        alert(`Node "${node.data.title}" failed. Pipeline stopped.`);
        break;
      }
    }
  }, [nodes, edges, runNode]);

  useEffect(() => {
    setNodes((ns: Node<NodeData>[]) => {
      const nodeMap = new Map(ns.map(n => [n.id, n]));
      const incomingByTarget = new Map<string, Edge[]>(
        ns.map(n => [n.id, edges.filter(e => e.target === n.id)])
      );
      const next = ns.map(n => {
        const cons = NodeConstraints[n.data.kind];
        const missing: string[] = [];
        for (const inp of cons.inputs) {
          const inc = (incomingByTarget.get(n.id) || []).find(e => e.targetHandle === inp.id);
          if (!inc) {
            missing.push(inp.label);
            continue;
          }
          const up = nodeMap.get(inc.source);
          if (!up || (up.data.status !== 'complete' && up.data.status !== 'configured')) {
            missing.push(inp.label);
          }
        }
        if (n.data.status === 'running' || n.data.status === 'complete' || n.data.status === 'failed') return n;
        const configured = n.data.config && Object.keys(n.data.config).length > 0;
        if (missing.length > 0) {
          return {
            ...n,
            data: { ...n.data, status: 'blocked' as NodeStatus, statusText: `Missing: ${missing.join(', ')}` },
          };
        }
        return {
          ...n,
          data: { ...n.data, status: (configured ? 'configured' : 'idle') as NodeStatus, statusText: '' },
        };
      });

      if (JSON.stringify(ns.map(n => n.data)) === JSON.stringify(next.map(n => n.data))) {
        return ns;
      }
      
      return next;
    });
  }, [nodes, edges, setNodes]);

  return {
    progress,
    runNode,
    onRunPipeline,
    setProgress,
  };
}
