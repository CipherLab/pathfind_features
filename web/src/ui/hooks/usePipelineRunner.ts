import { useState, useCallback, useEffect } from "react";
import { Node, Edge } from "@xyflow/react";
import { jpost } from "../lib/api";
import { NodeData, NodeStatus } from "../components/Flow/types";
import { validatePipeline } from "../components/Flow/validation";
import { NodeConstraints } from "../components/Flow/node-spec";

export function usePipelineRunner(
  nodes: Node<NodeData>[],
  edges: Edge[],
  setNodes: (
    nodes: Node<NodeData>[] | ((nodes: Node<NodeData>[]) => Node<NodeData>[])
  ) => void,
  setEdges: (edges: Edge[] | ((edges: Edge[]) => Edge[])) => void,
  experimentName: string,
  seed: number
) {
  const [progress, setProgress] = useState({ total: 0, completed: 0 });

  const propagateArtifacts = useCallback(
    (src: Node<NodeData>) => {
      setNodes((ns: Node<NodeData>[]) => {
        let updated = ns;
        const downIds = (kinds?: NodeData["kind"][]) =>
          edges.filter((e) => e.source === src.id).map((e) => e.target);

        if (src.data.kind === "data-source") {
          const downstream = downIds();
          updated = ns.map((n) => {
            if (downstream.includes(n.id)) {
              const cfg = {
                ...n.data.config,
                inputData: src.data.config?.inputData,
              };
              const status = (
                n.data.status === "idle" ? "configured" : n.data.status
              ) as NodeStatus;
              return { ...n, data: { ...n.data, config: cfg, status } };
            }
            return n;
          });
        } else if (src.data.kind === "feature-selection") {
          const downstream = downIds();
          updated = ns.map((n) => {
            if (
              downstream.includes(n.id) &&
              n.data.kind === "target-discovery"
            ) {
              const cfg = {
                ...n.data.config,
                inheritFeaturesFrom: src.id,
                featuresJson: src.data.config?.featuresJson,
              };
              const status = (
                n.data.status === "idle" ? "configured" : n.data.status
              ) as NodeStatus;
              return { ...n, data: { ...n.data, config: cfg, status } };
            }
            return n;
          });
        } else if (src.data.kind === "target-discovery") {
          const downstream = downIds();
          updated = ns.map((n) => {
            if (downstream.includes(n.id) && n.data.kind === "pathfinding") {
              const desiredName = src.data.config?.targetsName as
                | string
                | undefined;
              const fname =
                desiredName && desiredName.trim().length > 0
                  ? desiredName
                  : `targets_${experimentName}.json`;
              const targetsPath = `pipeline_runs/${experimentName}/${fname}`;
              const cfg = {
                ...n.data.config,
                inheritTargetsFrom: src.id,
                targetsJson: targetsPath,
              };
              const status = (
                n.data.status === "idle" ? "configured" : n.data.status
              ) as NodeStatus;
              return { ...n, data: { ...n.data, config: cfg, status } };
            }
            return n;
          });
        } else if (src.data.kind === "pathfinding") {
          const downstream = downIds();
          updated = ns.map((n) => {
            if (
              downstream.includes(n.id) &&
              n.data.kind === "feature-engineering"
            ) {
              const cfg = {
                ...n.data.config,
                inheritRelationshipsFrom: src.id,
                relationshipsJson: "relationships.json",
              };
              const status = (
                n.data.status === "idle" ? "configured" : n.data.status
              ) as NodeStatus;
              return { ...n, data: { ...n.data, config: cfg, status } };
            }
            return n;
          });
        } else if (
          src.data.kind === "transform" ||
          src.data.kind === "feature-engineering"
        ) {
          const downstream = downIds();
          updated = ns.map((n) => {
            if (downstream.includes(n.id)) {
              const cfg = {
                ...(n.data.config || {}),
                inputData: src.data.config.outputPath,
                // if this node expects features.json (target-discovery), pass along derived features when present
                ...(n.data.kind === "target-discovery" &&
                src.data.config.derivedFeatures
                  ? { featuresJson: src.data.config.derivedFeatures }
                  : {}),
              };
              const status = (
                n.data.status === "idle" ? "configured" : n.data.status
              ) as NodeStatus;
              return { ...n, data: { ...n.data, config: cfg, status } };
            }
            return n;
          });
        }
        return updated;
      });
    },
    [edges, setNodes, experimentName]
  );

  const runNode = useCallback(
    async (node: Node<NodeData>) => {
      const { id, data } = node;
      const cfg = data.config || {};

      setNodes((ns) =>
        ns.map((n) =>
          n.id === id
            ? {
                ...n,
                data: {
                  ...n.data,
                  status: "running" as NodeStatus,
                  statusText:
                    data.kind === "target-discovery"
                      ? "Running target discovery..."
                      : data.kind === "feature-selection"
                      ? "Loading features..."
                      : data.kind === "pathfinding"
                      ? "Exploring relationships..."
                      : data.kind === "feature-engineering"
                      ? "Brewing features..."
                      : data.kind === "transform"
                      ? "Applying transform..."
                      : data.kind === "output"
                      ? "Finalizing output..."
                      : "Working...",
                },
              }
            : n
        )
      );

      try {
        if (data.kind === "target-discovery") {
          const output_file = `pipeline_runs/${experimentName}/01_adaptive_targets_${id}.parquet`;
          const discovery_file = `pipeline_runs/${experimentName}/01_target_discovery_${id}.json`;
          const payload = {
            input_file: cfg.inputData || "v5.0/train.parquet",
            features_json_file: cfg.featuresJson || "v5.0/features.json",
            output_file: output_file,
            discovery_file: discovery_file,
            skip_walk_forward: !(cfg.walkForward ?? true),
            max_eras: cfg.smoke ? cfg.smokeEras : undefined,
            row_limit: cfg.smoke ? cfg.smokeRows : undefined,
            target_limit: cfg.smoke ? cfg.smokeFeat : undefined, // Note: UI calls this 'smokeFeat'
          };
          const result = await jpost("/steps/target-discovery", payload);
          
          setNodes((ns) =>
            ns.map((n) =>
              n.id === id
                ? {
                    ...n,
                    data: {
                      ...n.data,
                      config: {
                        ...n.data.config,
                        outputPath: result.output_file,
                        discoveryPath: result.discovery_file,
                        logs: result.stdout,
                      },
                    },
                  }
                : n
            )
          );
        } else if (data.kind === "transform") {
          const inputEdge = edges.find((e) => e.target === id);
          if (!inputEdge) {
            throw new Error("Transform node requires an input connection");
          }
          const inputNode = nodes.find((n) => n.id === inputEdge.source);
          if (!inputNode) {
            throw new Error("Input node not found for transform");
          }

          const inputDataPath =
            inputNode.data.config.outputPath ||
            inputNode.data.config.inputData ||
            "v5.0/train.parquet";

          const payload = {
            input_data: inputDataPath,
            transform_script: cfg.script,
            output_data: `pipeline_runs/${experimentName}/transformed_data_${id}.parquet`,
          };
          const result = await jpost("/transforms/execute", payload);
          // Try to derive a features.json artifact from the transformed parquet
          let derivedFeaturesPath: string | undefined = undefined;
          try {
            const fpath = `pipeline_runs/${experimentName}/features_${id}.json`;
            const r = await jpost("/features/derive", {
              input_data: result.output,
              output_json: fpath,
            });
            derivedFeaturesPath = r.output || fpath;
          } catch (e) {
            // ignore
          }
          const outPath: string = result.output;
          setNodes((ns) =>
            ns.map((n) =>
              n.id === id
                ? {
                    ...n,
                    data: {
                      ...n.data,
                      config: {
                        ...n.data.config,
                        outputPath: outPath,
                        logs: result.stdout,
                        derivedFeatures: derivedFeaturesPath,
                      },
                    },
                  }
                : n
            )
          );
          // Update edge labels from this node to reflect real filenames
          setEdges((es) =>
            es.map((e) => {
              if (e.source !== id) return e;
              const sh: any = (e as any).data?.sourceHandle || e.sourceHandle;
              const label =
                sh === "out-features"
                  ? derivedFeaturesPath
                    ? derivedFeaturesPath.split("/").pop()!
                    : "features.json"
                  : outPath
                  ? outPath.split("/").pop()!
                  : "transformed.parquet";
              const color = (e.style && (e.style as any).stroke) || "#a855f7";
              return {
                ...e,
                label,
                labelStyle: { fill: color, fontSize: 10 },
              } as any;
            })
          );
        } else if (data.kind === "output") {
          const inputEdge = edges.find((e) => e.target === id);
          if (!inputEdge) {
            throw new Error("Output node requires an input connection");
          }
          const inputNode = nodes.find((n) => n.id === inputEdge.source);
          if (!inputNode) {
            throw new Error("Input node not found for output");
          }

          const sourcePath = inputNode.data.config.outputPath;
          const destPath = cfg.outputPath;

          if (sourcePath && destPath && sourcePath !== destPath) {
            await jpost("/files/move", {
              source: sourcePath,
              destination: destPath,
            });
          }
        } else {
          await new Promise((res) => setTimeout(res, 300));
        }

        setNodes((ns) =>
          ns.map((n) =>
            n.id === id
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    status: "complete" as NodeStatus,
                    statusText:
                      data.kind === "target-discovery"
                        ? "✅ Targets discovered"
                        : data.kind === "feature-selection"
                        ? "✅ Features selected"
                        : data.kind === "pathfinding"
                        ? "✅ Relationships mapped"
                        : data.kind === "feature-engineering"
                        ? "✅ Features generated"
                        : data.kind === "transform"
                        ? "✅ Transform applied"
                        : data.kind === "output"
                        ? "✅ Output saved"
                        : "✅ Done",
                  },
                }
              : n
          )
        );

        if (
          data.kind === "target-discovery" ||
          data.kind === "pathfinding" ||
          data.kind === "feature-selection" ||
          data.kind === "transform" ||
          data.kind === "feature-engineering"
        ) {
          propagateArtifacts(node);
        }
      } catch (e: any) {
        const msg =
          e && typeof e === "object" && "message" in e
            ? String(e.message)
            : "Unknown error";
        setNodes((ns) =>
          ns.map((n) =>
            n.id === id
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    status: "failed" as NodeStatus,
                    statusText: `❌ ${msg}`,
                  },
                }
              : n
          )
        );
        throw e;
      }
    },
    [nodes, edges, setNodes, propagateArtifacts, experimentName, seed]
  );

  const onRunPipeline = useCallback(async () => {
    // Validate current graph as-is; do NOT force output to 'idle' before validation,
    // as that incorrectly triggers the "not configured" alert.
    const order = validatePipeline(nodes, edges);
    if (!order) return;

    setProgress({ total: order.length, completed: 0 });
    const map = new Map(nodes.map((n) => [n.id, n]));
    for (const id of order) {
      const node = map.get(id);
      if (!node) continue;
      try {
        await runNode(node);
        setProgress((p) => ({ total: p.total, completed: p.completed + 1 }));
      } catch {
        alert(`Node "${node.data.title}" failed. Pipeline stopped.`);
        break;
      }
    }
  }, [nodes, edges, runNode, setNodes]);

  useEffect(() => {
    setNodes((ns: Node<NodeData>[]) => {
      const nodeMap = new Map(ns.map((n) => [n.id, n]));
      const incomingByTarget = new Map<string, Edge[]>(
        ns.map((n) => [n.id, edges.filter((e) => e.target === n.id)])
      );
      return ns.map((n) => {
        // Do not override nodes that are actively running or have finished/failed.
        // This prevents the effect from demoting their status during the run loop.
        if (
          n.data.status === "running" ||
          n.data.status === "complete" ||
          n.data.status === "failed"
        ) {
          return n;
        }

        const cons = NodeConstraints[n.data.kind];
        const missing: string[] = [];
        const waiting: string[] = [];
        for (const inp of cons.inputs) {
          const inc = (incomingByTarget.get(n.id) || []).find(
            (e) => e.targetHandle === inp.id
          );
          if (!inc) {
            missing.push(inp.label);
            continue;
          }
          const up = nodeMap.get(inc.source);
          if (!up) {
            missing.push(inp.label);
            continue;
          }
          // Consider upstream 'running' as present (waiting), not missing
          if (
            up.data.status === "failed" ||
            up.data.status === "idle" ||
            up.data.status === "blocked"
          ) {
            missing.push(inp.label);
          } else if (up.data.status === "running") {
            waiting.push(inp.label);
          } // complete/configured are satisfied
        }

        const configured =
          n.data.config && Object.keys(n.data.config).length > 0;
        let newStatus: NodeStatus = "idle";
        let newStatusText = "";
        if (missing.length > 0) {
          newStatus = "blocked";
          newStatusText = `Missing: ${missing.join(", ")}`;
        } else if (waiting.length > 0) {
          newStatus = "pending";
          newStatusText = ""; // keep UI clean; badge color conveys pending
        } else {
          newStatus = configured ? "configured" : "idle";
          newStatusText = "";
        }

        if (
          n.data.status === newStatus &&
          n.data.statusText === newStatusText
        ) {
          return n;
        }

        return {
          ...n,
          data: { ...n.data, status: newStatus, statusText: newStatusText },
        };
      });
    });
  }, [nodes, edges, setNodes]);

  return {
    progress,
    runNode,
    onRunPipeline,
    setProgress,
  };
}
