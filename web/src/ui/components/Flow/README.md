# Flow Helpers

This folder will contain custom flow nodes and small helpers. For now the BuilderPage uses a generic NodeCard.
## Checklist

React Flow canvas and layout: Done
Node palette with draggable/addable types: Done (buttons to add for now)
Custom node visuals and status dots: Done
Config sidebar (reuse ParameterForm, artifact pickers): Done
Execute Target Discovery node via existing /runs API: Done
Keep backend unchanged: Done
Wire new page and navigation: Done
Build/lint smoke test: Done (web build passes)
What I changed and added

Updated dependency
web/package.json: added @xyflow/react
App navigation and routes
web/src/ui/App.tsx: added “Visual Builder” nav link
web/src/main.tsx: added /builder route
New visual builder page (Phase 1 foundation)
BuilderPage.tsx
React Flow canvas with Background/MiniMap/Controls
Node palette: add “Data”, “Targets”, “Pathfind”, “Features”, “Output”
Node card (icon + status dot + left/right handles) with types:
data-source, target-discovery, pathfinding, feature-engineering, output
Sidebar reusing ParameterForm for configuration (inputs, features.json, seed, smoke, etc.)
Run Node invokes /runs with disable_pathfinding: true for Target Discovery (Phase 1 scope)
“Run Pipeline” triggers Target Discovery node run for now
Small doc stub
README.md
How to try it

Build passed locally; preview ran at http://localhost:4173 during validation.
Open the app, click Visual Builder:
Add nodes via palette (e.g., Data, Target Discovery).
Select “Target Discovery” to configure it in the sidebar (reuses your form).
Click “Run Node” (or “Run Pipeline”) to start a Phase 1 run using your existing /runs endpoint.
Notes on design choices

Phase 1 follows your migration plan precisely:
Canvas + palette + selection sidebar are in place.
Target Discovery is wired to execute; other nodes are scaffolded visually with status.
ParameterForm/Global pickers are reused so no new validation logic is required.
Node visuals: minimal card with status dot and left/right handles, ready to receive richer badges (progress, Sharpe, etc.).
Type safety: nodes are typed as Node<NodeData> to align with @xyflow/react v12.
Phase 2 next steps (small, clear increments)

Implement node-specific config panels (split the shared ParameterForm into per-node minimal forms that wrap the existing widgets).
Connection validation: only allow data-source → target-discovery → pathfinding → feature-engineering → output; invalid shows red.
Execution DAG: topological order over edges with per-node /runs payloads:
Pathfinding: disable_pathfinding=false, max_new_features from config; require target-discovery artifact via connected upstream.
Feature Engineering: require relationships artifact; control new features and strategies.
Artifact passing: map outputs to downstream node config automatically (edges carry artifact references).
Phase 3 ideas

Templates (Quick Discovery, Full Exploration, Debug Mode) as saved node/edge JSON.
Import/export pipelines to/from JSON.
Real-time status via ActiveRuns polling mapped to selected node by experiment_name, with in-node progress summaries.
Quality gates

Build: PASS (vite build)
Lint/typecheck: Addressed TypeScript and typing for @xyflow/react v12
Smoke: Preview served successfully; canvas, palette, selection, and run action exercised
Requirements coverage

React Flow canvas: Done
Node types scaffolded (data-source, target-discovery, pathfinding, feature-engineering, output): Done (render + handles)
Configuration sidebar reusing existing components: Done (ParameterForm/GlobalPickerModal)
Execute at least Target Discovery with existing backend: Done
Backend unchanged: Done
Persistence/templates/import-export: Deferred (Phase 3)
Files changed

web/package.json: add @xyflow/react
web/src/ui/App.tsx: add nav link
web/src/main.tsx: route /builder
web/src/ui/pages/BuilderPage.tsx: new visual builder
web/src/ui/components/Flow/README.md: doc stub
Wrap-up You now have a working Visual Builder page backed by React Flow, reusing your wizard parameters, with a palette, a canvas, and executable Target Discovery node. Next, we can wire up the remaining node configs, connection validation, and artifact flow to complete the node empire.