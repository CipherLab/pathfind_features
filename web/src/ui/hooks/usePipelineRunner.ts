import { useState, useCallback, useEffect, useRef } from "react";
import { Node, Edge } from "@xyflow/react";
import { jpost, jget } from "../lib/api";
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
  const pollersRef = useRef<Record<string, number>>({});

  const stopPoller = useCallback((id: string) => {
    const p = pollersRef.current[id];
    if (p) {
      clearInterval(p);
      delete pollersRef.current[id];
    }
  }, []);

  const persistJob = useCallback(
    (nodeId: string, jobId: string) => {
      try {
        localStorage.setItem(`job:${experimentName}:${nodeId}`, jobId);
      } catch {}
    },
    [experimentName]
  );

  const clearJob = useCallback(
    (nodeId: string) => {
      try {
        localStorage.removeItem(`job:${experimentName}:${nodeId}`);
      } catch {}
      setNodes((ns) =>
        ns.map((n) =>
          n.id === nodeId
            ? {
                ...n,
                data: {
                  ...n.data,
                  config: { ...n.data.config, jobId: undefined },
                },
              }
            : n
        )
      );
    },
    [experimentName, setNodes]
  );

  useEffect(() => {
    return () => {
      // Cleanup any outstanding pollers on unmount
      Object.values(pollersRef.current).forEach((h) => clearInterval(h));
      pollersRef.current = {};
    };
  }, []);

  // Auto-resume polling if a job id exists on the node or in localStorage
  useEffect(() => {
    nodes.forEach((n) => {
      const cfg = (n.data && n.data.config) || ({} as any);
      let jobId: string | undefined = cfg.jobId;
      if (!jobId) {
        try {
          jobId =
            localStorage.getItem(`job:${experimentName}:${n.id}`) || undefined;
        } catch {}
      }
      if (!jobId) return;
      if (pollersRef.current[n.id]) return; // already polling

      const nodeId = n.id;
      const kind = n.data.kind;
      const outTargets = `pipeline_runs/${experimentName}/01_adaptive_targets_${nodeId}.parquet`;
      const outDiscovery = `pipeline_runs/${experimentName}/01_target_discovery_${nodeId}.json`;
      const outRelationships = `pipeline_runs/${experimentName}/02_relationships_${nodeId}.json`;

      const handleSuccess = () => {
        if (kind === "target-discovery") {
          setNodes((ns) =>
            ns.map((x) =>
              x.id === nodeId
                ? {
                    ...x,
                    data: {
                      ...x.data,
                      config: {
                        ...x.data.config,
                        outputPath: outTargets,
                        discoveryPath: outDiscovery,
                        jobId: undefined,
                      },
                    },
                  }
                : x
            )
          );
        } else if (kind === "pathfinding") {
          setNodes((ns) =>
            ns.map((x) =>
              x.id === nodeId
                ? {
                    ...x,
                    data: {
                      ...x.data,
                      config: {
                        ...x.data.config,
                        relationshipsPath: outRelationships,
                        jobId: undefined,
                      },
                    },
                  }
                : x
            )
          );
          setEdges((es) =>
            es.map((e) => {
              if (e.source !== nodeId) return e;
              const outName = outRelationships.split("/").pop()!;
              const color = (e.style && (e.style as any).stroke) || "#ef4444";
              return {
                ...e,
                label: outName,
                labelStyle: { fill: color, fontSize: 10 },
              } as any;
            })
          );
        }
        clearJob(nodeId);
      };

      pollersRef.current[nodeId] = window.setInterval(async () => {
        try {
          const job = await jget<any>(`/jobs/${encodeURIComponent(jobId!)}`);
          const logs = await jget<any>(
            `/jobs/${encodeURIComponent(jobId!)}/logs`
          ).catch(() => ({ content: "" }));
          setNodes((ns) =>
            ns.map((x) =>
              x.id === nodeId
                ? {
                    ...x,
                    data: {
                      ...x.data,
                      config: { ...x.data.config, logs: logs.content },
                    },
                  }
                : x
            )
          );
          if (job && (job.status === "SUCCESS" || job.status === "ERROR")) {
            stopPoller(nodeId);
            if (job.status === "SUCCESS") {
              handleSuccess();
            }
          }
        } catch (e: any) {
          // Stop polling on 404s (job lost) to avoid duplicate retries
          const msg = e && typeof e.message === "string" ? e.message : "";
          if (msg.includes("404")) {
            stopPoller(nodeId);
            clearJob(nodeId);
          }
        }
      }, 4000);
    });
  }, [nodes, experimentName, stopPoller, setNodes, setEdges, clearJob]);

  const propagateArtifacts = useCallback(
    (src: Node<NodeData>) => {
      setNodes((ns) => {
        let updated = ns;
        const downstreamIds = edges
          .filter((e) => e.source === src.id)
          .map((e) => e.target);

        if (src.data.kind === "data-source") {
          updated = ns.map((n) => {
            if (downstreamIds.includes(n.id)) {
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
          updated = ns.map((n) => {
            if (
              downstreamIds.includes(n.id) &&
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
          updated = ns.map((n) => {
            if (downstreamIds.includes(n.id) && n.data.kind === "pathfinding") {
              const parquetPath = src.data.config?.outputPath as
                | string
                | undefined;
              const discoveryPath = src.data.config?.discoveryPath as
                | string
                | undefined;
              const cfg = {
                ...n.data.config,
                inheritTargetsFrom: src.id,
                ...(parquetPath ? { inputData: parquetPath } : {}),
                ...(discoveryPath ? { targetsJson: discoveryPath } : {}),
              };
              const status = (
                n.data.status === "idle" ? "configured" : n.data.status
              ) as NodeStatus;
              return { ...n, data: { ...n.data, config: cfg, status } };
            }
            return n;
          });
        } else if (src.data.kind === "pathfinding") {
          updated = ns.map((n) => {
            if (
              downstreamIds.includes(n.id) &&
              n.data.kind === "feature-engineering"
            ) {
              const producedRelPath = src.data.config?.relationshipsPath as
                | string
                | undefined;
              const cfg = {
                ...n.data.config,
                inheritRelationshipsFrom: src.id,
                ...(producedRelPath
                  ? { relationshipsJson: producedRelPath }
                  : {}),
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
          updated = ns.map((n) => {
            if (downstreamIds.includes(n.id)) {
              const cfg = {
                ...(n.data.config || {}),
                inputData: src.data.config.outputPath,
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
    [edges, setNodes]
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
          const eras = cfg.smoke
            ? cfg.smokeEras && cfg.smokeEras > 0
              ? cfg.smokeEras
              : undefined
            : undefined;
          const rows = cfg.smoke
            ? cfg.smokeRows && cfg.smokeRows > 0
              ? cfg.smokeRows
              : undefined
            : undefined;
          const rawTargetsCap =
            (cfg as any).smokeTargets ?? (cfg as any).smokeFeat;
          const targetsCap = cfg.smoke
            ? rawTargetsCap && rawTargetsCap > 0
              ? rawTargetsCap
              : undefined
            : undefined;
          const payload = {
            input_file: cfg.inputData || "v5.0/train.parquet",
            features_json_file: cfg.featuresJson || "v5.0/features.json",
            output_file: output_file,
            discovery_file: discovery_file,
            skip_walk_forward: !(cfg.walkForward ?? true),
            max_eras: eras,
            row_limit: rows,
            target_limit: targetsCap,
            td_eval_mode: cfg.td_eval_mode,
            td_top_full_models: cfg.td_top_full_models,
            td_ridge_lambda: cfg.td_ridge_lambda,
            td_sample_per_era: cfg.td_sample_per_era,
            td_max_combinations: cfg.td_max_combinations,
            td_feature_fraction: cfg.td_feature_fraction,
            td_num_boost_round: cfg.td_num_boost_round,
            td_max_era_cache: cfg.td_max_era_cache,
            td_clear_cache_every: cfg.td_clear_cache_every,
            td_pre_cache_dir: cfg.td_pre_cache_dir,
            td_persist_pre_cache: cfg.td_persist_pre_cache,
          };
          // Start as background job and wait for it to finish before proceeding
          const start = await jpost(
            "/steps/target-discovery/background",
            payload
          );
          const jobId: string = start.job_id;
          // Store jobId for potential resume
          setNodes((ns) =>
            ns.map((n) =>
              n.id === id
                ? {
                    ...n,
                    data: { ...n.data, config: { ...n.data.config, jobId } },
                  }
                : n
            )
          );
          persistJob(id, jobId);
          // Wait synchronously (with small sleeps) for job completion
          // while streaming logs into the node config
          // Do not register a periodic poller here; that's for auto-resume
          // on page refresh.
          // eslint-disable-next-line no-constant-condition
          while (true) {
            const job = await jget<any>(`/jobs/${encodeURIComponent(jobId)}`);
            const logs = await jget<any>(
              `/jobs/${encodeURIComponent(jobId!)}/logs`
            ).catch(() => ({ content: "" }));
            setNodes((ns) =>
              ns.map((n) =>
                n.id === id
                  ? {
                      ...n,
                      data: {
                        ...n.data,
                        config: { ...n.data.config, logs: logs.content },
                      },
                    }
                  : n
              )
            );
            if (job && (job.status === "SUCCESS" || job.status === "ERROR")) {
              if (job.status === "SUCCESS") {
                setNodes((ns) =>
                  ns.map((n) =>
                    n.id === id
                      ? {
                          ...n,
                          data: {
                            ...n.data,
                            config: {
                              ...n.data.config,
                              outputPath: output_file,
                              discoveryPath: discovery_file,
                              jobId: undefined,
                            },
                          },
                        }
                      : n
                  )
                );
                clearJob(id);
                break;
              } else {
                throw new Error(
                  `Target discovery failed (exit=${job.return_code})`
                );
              }
            }
            await new Promise((res) => setTimeout(res, 2000));
          }
        } else if (data.kind === "pathfinding") {
          // Resolve required inputs: parquet and targets json
          let parquetPath: string | undefined = cfg.inputData;
          let targetsJsonPath: string | undefined = cfg.targetsJson;
          if (!parquetPath || !targetsJsonPath) {
            const incoming = edges.filter((e) => e.target === id);
            for (const e of incoming) {
              const up = nodes.find((n) => n.id === e.source);
              if (!up) continue;
              if (e.sourceHandle === "out-parquet") {
                parquetPath =
                  up.data.config.outputPath ||
                  up.data.config.inputData ||
                  parquetPath;
              }
              if (e.sourceHandle === "out-discovery") {
                targetsJsonPath =
                  up.data.config.discoveryPath ||
                  up.data.config.targetsJson ||
                  targetsJsonPath;
              }
            }
          }
          if (!parquetPath || !targetsJsonPath) {
            throw new Error(
              "Pathfinding requires both parquet and targets.json inputs"
            );
          }
          const relationships_file = `pipeline_runs/${experimentName}/02_relationships_${id}.json`;
          // Map config to new script params
          const yolo_mode = Boolean(cfg.yolo);
          const feature_limit =
            cfg.featureLimit && cfg.featureLimit > 0
              ? cfg.featureLimit
              : undefined;
          const debug = Boolean(cfg.debug);
          const pf_feature_cap =
            cfg.smokeFeat && cfg.smokeFeat > 0 ? cfg.smokeFeat : undefined;
          const last_n_eras =
            cfg.lastNEras && cfg.lastNEras > 0 ? cfg.lastNEras : undefined;
          const cache_dir = cfg.cacheDir || "cache/pathfinding_cache";
          const run_sanity_check = cfg.sanityCheck ?? false;
          const n_paths = cfg.nPaths && cfg.nPaths > 0 ? cfg.nPaths : undefined;
          const max_path_length =
            cfg.maxPathLen && cfg.maxPathLen > 0 ? cfg.maxPathLen : undefined;
          const min_strength =
            cfg.minStrength && cfg.minStrength > 0
              ? cfg.minStrength
              : undefined;
          const top_k = cfg.topK && cfg.topK > 0 ? cfg.topK : undefined;
          const batch_size =
            cfg.batchSize && cfg.batchSize > 0 ? cfg.batchSize : undefined;
          const payload = {
            input_file: parquetPath,
            target_col: "adaptive_target",
            output_relationships_file: relationships_file,
            yolo_mode,
            feature_limit,
            debug,
            cache_dir,
            run_sanity_check,
            pf_feature_cap,
            last_n_eras,
            n_paths,
            max_path_length,
            min_strength,
            top_k,
            batch_size,
          } as any;
          const start = await jpost("/steps/pathfinding/background", payload);
          const jobId: string = start.job_id;
          setNodes((ns) =>
            ns.map((n) =>
              n.id === id
                ? {
                    ...n,
                    data: { ...n.data, config: { ...n.data.config, jobId } },
                  }
                : n
            )
          );
          persistJob(id, jobId);
          // Wait synchronously for job completion while updating logs
          // eslint-disable-next-line no-constant-condition
          while (true) {
            const job = await jget<any>(`/jobs/${encodeURIComponent(jobId)}`);
            const logs = await jget<any>(
              `/jobs/${encodeURIComponent(jobId)}/logs`
            ).catch(() => ({ content: "" }));
            setNodes((ns) =>
              ns.map((n) =>
                n.id === id
                  ? {
                      ...n,
                      data: {
                        ...n.data,
                        config: { ...n.data.config, logs: logs.content },
                      },
                    }
                  : n
              )
            );
            if (job && (job.status === "SUCCESS" || job.status === "ERROR")) {
              if (job.status === "SUCCESS") {
                setNodes((ns) =>
                  ns.map((n) =>
                    n.id === id
                      ? {
                          ...n,
                          data: {
                            ...n.data,
                            config: {
                              ...n.data.config,
                              relationshipsPath: relationships_file,
                              jobId: undefined,
                            },
                          },
                        }
                      : n
                  )
                );
                setEdges((es) =>
                  es.map((e) => {
                    if (e.source !== id) return e;
                    const outName = relationships_file.split("/").pop()!;
                    const color =
                      (e.style && (e.style as any).stroke) || "#ef4444";
                    return {
                      ...e,
                      label: outName,
                      labelStyle: { fill: color, fontSize: 10 },
                    } as any;
                  })
                );
                clearJob(id);
                break;
              } else {
                throw new Error(`Pathfinding failed (exit=${job.return_code})`);
              }
            }
            await new Promise((res) => setTimeout(res, 2000));
          }
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
          // Allow optional script args from config (either array or a raw string to parse)
          const parseArgs = (s: string): string[] => {
            const out: string[] = [];
            let cur = "";
            let quote: '"' | "'" | null = null;
            for (let i = 0; i < s.length; i++) {
              const ch = s[i];
              if (quote) {
                if (ch === quote) {
                  quote = null;
                } else if (
                  ch === "\\" &&
                  i + 1 < s.length &&
                  s[i + 1] === quote
                ) {
                  cur += quote;
                  i++;
                } else {
                  cur += ch;
                }
              } else {
                if (ch === '"' || ch === "'") {
                  quote = ch as any;
                } else if (ch === " " || ch === "\n" || ch === "\t") {
                  if (cur) {
                    out.push(cur);
                    cur = "";
                  }
                } else {
                  cur += ch;
                }
              }
            }
            if (cur) out.push(cur);
            return out;
          };
          const argsArray: string[] = Array.isArray(cfg.scriptArgs)
            ? (cfg.scriptArgs as string[])
            : typeof (cfg as any).scriptArgsStr === "string" &&
              (cfg as any).scriptArgsStr
            ? parseArgs((cfg as any).scriptArgsStr as string)
            : [];

          const payload = {
            input_data: inputDataPath,
            transform_script: cfg.script,
            output_data: `pipeline_runs/${experimentName}/transformed_data_${id}.parquet`,
            script_args: argsArray,
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

        // For background jobs we set completion status on success above.
        if (data.kind !== "target-discovery" && data.kind !== "pathfinding") {
          setNodes((ns) =>
            ns.map((n) =>
              n.id === id
                ? {
                    ...n,
                    data: {
                      ...n.data,
                      status: "complete" as NodeStatus,
                      statusText:
                        data.kind === "feature-selection"
                          ? "✅ Features selected"
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
            data.kind === "feature-selection" ||
            data.kind === "transform" ||
            data.kind === "feature-engineering"
          ) {
            propagateArtifacts(node);
          }
        } else {
          // For background jobs, mark status complete now and propagate
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
                          : "✅ Relationships mapped",
                    },
                  }
                : n
            )
          );
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
