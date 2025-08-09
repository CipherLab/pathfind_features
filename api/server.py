from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict
from pathlib import Path
import os
from math import ceil
import json
try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

from .runner import RUNS
from . import ops
from .models import RunRequest

app = FastAPI(title="Pathfind Features API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunReqModel(BaseModel):
    input_data: str
    features_json: str
    run_name: Optional[str] = None
    stage1_from: Optional[str] = None
    stage2_from: Optional[str] = None
    force: bool = False
    skip_walk_forward: bool = False
    max_new_features: int = 20
    yolo_mode: bool = False
    pf_debug: bool = False
    pf_debug_every_rows: int = 10000
    disable_pathfinding: bool = False
    pretty: bool = True
    smoke_mode: bool = False
    smoke_max_eras: Optional[int] = None
    smoke_row_limit: Optional[int] = None
    smoke_feature_limit: Optional[int] = None
    seed: int = 42

class PreflightReq(BaseModel):
    input_data: str
    features_json: str
    stage1_from: Optional[str] = None
    stage2_from: Optional[str] = None
    skip_walk_forward: bool = False
    max_new_features: int = 20
    disable_pathfinding: bool = False
    smoke_mode: bool = False
    smoke_max_eras: Optional[int] = None
    smoke_row_limit: Optional[int] = None
    smoke_feature_limit: Optional[int] = None
    seed: int = 42

class InspectReq(BaseModel):
    path: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/ping")
async def ping():
    return {"status": "pong"}

@app.get("/runs")
async def list_runs():
    return RUNS.list_runs()

# NOTE: Declare static and specific FS-based routes BEFORE dynamic parameter routes
@app.get("/runs/list-fs")
async def list_runs_fs_compat():
    return RUNS.list_runs_fs()

@app.get("/runs/fs")
async def list_runs_fs_alias():
    return RUNS.list_runs_fs()

@app.get("/runs/{run_name}/summary")
async def get_run_summary(run_name: str):
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / "pipeline_runs" / run_name
    p = base / "run_summary.json"
    if not p.exists():
        raise HTTPException(404, detail="summary not found")
    return p.read_text(encoding="utf-8", errors="ignore")

@app.get("/runs/{run_name}/logs")
async def get_run_logs(run_name: str, tail: int = 4000):
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / "pipeline_runs" / run_name
    p = base / "logs.log"
    if not p.exists():
        raise HTTPException(404, detail="logs not found")
    txt = p.read_text(encoding="utf-8", errors="ignore")
    return {"content": txt[-tail:]}

# Back-compat aliases used by CLI/UI
@app.get("/runs/list-fs/{run_name}/logs")
async def get_run_logs_compat(run_name: str, tail: int = 4000):
    return await get_run_logs(run_name, tail)

@app.get("/runs/list-fs/{run_name}/summary")
async def get_run_summary_compat(run_name: str):
    return await get_run_summary(run_name)

@app.get("/runs/{run_name}/artifacts")
async def list_run_artifacts(run_name: str):
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / "pipeline_runs" / run_name
    if not base.exists() or not base.is_dir():
        raise HTTPException(404, detail="run not found")
    # Backfill: if this run has engineered features but no merged features.json, create it now
    try:
        merged_fjson = base / "features.json"
        new_list = base / "new_feature_names.json"
        if (not merged_fjson.exists()) and new_list.exists():
            # Merge baseline features.json with new features list
            root = Path(__file__).resolve().parents[1]
            baseline = root / "v5.0" / "features.json"
            try:
                base_dict = {}
                if baseline.exists():
                    base_dict = json.loads(baseline.read_text(encoding="utf-8", errors="ignore"))
                new_names = json.loads(new_list.read_text(encoding="utf-8", errors="ignore")) if new_list.exists() else []
                if not isinstance(base_dict, dict):
                    base_dict = {}
                fs = base_dict.setdefault("feature_sets", {}) if isinstance(base_dict, dict) else {}
                medium = fs.get("medium") if isinstance(fs, dict) else None
                if not isinstance(medium, list):
                    medium = []
                    if isinstance(fs, dict):
                        fs["medium"] = medium
                # Normalize existing names
                existing_names = []
                for item in list(medium):
                    if isinstance(item, str):
                        existing_names.append(item)
                    elif isinstance(item, dict) and "name" in item:
                        existing_names.append(str(item["name"]))
                # Append new names uniquely
                for nm in (new_names or []):
                    if isinstance(nm, str) and nm not in existing_names:
                        medium.append(nm)
                        existing_names.append(nm)
                merged_fjson.write_text(json.dumps(base_dict, indent=2), encoding="utf-8")
            except Exception:
                # As a last resort, write a minimal features.json with just the new names
                try:
                    new_names = json.loads(new_list.read_text(encoding="utf-8", errors="ignore")) if new_list.exists() else []
                    minimal = {"feature_sets": {"medium": list(new_names or [])}}
                    merged_fjson.write_text(json.dumps(minimal, indent=2), encoding="utf-8")
                except Exception:
                    pass
    except Exception:
        # Never fail listing due to backfill attempt
        pass
    out = []
    for p in base.iterdir():
        if p.is_file():
            try:
                stat = p.stat()
                out.append({
                    "name": p.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                })
            except Exception:
                out.append({"name": p.name})
    return sorted(out, key=lambda x: x.get("name", ""))

@app.get("/runs/{run_name}/performance")
async def get_run_performance(run_name: str):
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / "pipeline_runs" / run_name
    p = base / "performance_report.txt"
    if not p.exists():
        # Return empty content instead of 404 to avoid noisy logs during early run stages
        return {"content": ""}
    return {"content": p.read_text(encoding="utf-8", errors="ignore")}

@app.post("/runs", status_code=202)
async def start_run(body: RunReqModel):
    rec = RUNS.start(RunRequest(**body.model_dump()))
    return rec.to_dict()

@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    rec = RUNS.get(run_id)
    if not rec:
        raise HTTPException(status_code=404, detail="run not found")
    return rec

@app.get("/runs/{run_id}/logs")
async def get_logs(run_id: str):
    rec = RUNS.get(run_id)
    if not rec:
        raise HTTPException(status_code=404, detail="run not found")
    path = Path(rec["logs_path"])
    if not path.exists():
        return {"content": ""}
    return {"content": path.read_text(encoding="utf-8", errors="ignore")[-200000:]}


def _analyze_parquet(path: Path):
    info = {
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "rows": None,
        "row_groups": None,
        "columns": None,
        "num_features": None,
        "num_targets": None,
        "has_era": None,
        "schema": None,
    }
    if not path.exists():
        return info
    if pq is None:
        return info
    try:
        pf = pq.ParquetFile(str(path))
        info["rows"] = pf.metadata.num_rows if pf.metadata else None
        info["row_groups"] = pf.metadata.num_row_groups if pf.metadata else None
        names = list(pf.schema.names) if pf.schema else []
        info["columns"] = len(names)
        info["num_features"] = sum(1 for n in names if str(n).startswith("feature"))
        info["num_targets"] = sum(1 for n in names if str(n).startswith("target"))
        info["has_era"] = any(str(n) == "era" for n in names)
        info["schema"] = names
    except Exception:
        # Leave partial info
        pass
    return info


@app.post("/preflight")
async def preflight(req: PreflightReq):
    root = Path(__file__).resolve().parents[1]
    def resolve(p: str) -> Path:
        # Absolute path or project-relative
        pp = Path(p)
        return pp if pp.is_absolute() else (root / p)

    issues = []
    ds = _analyze_parquet(resolve(req.input_data))

    if not ds["exists"]:
        issues.append({"severity": "error", "field": "input_data", "message": f"Input parquet not found: {req.input_data}"})
    features_info = {"exists": False, "kind": "unknown", "valid_for_pipeline": False, "medium_count": None}
    features_diff = {"baseline_exists": False, "added": [], "removed": []}
    fj_path = resolve(req.features_json)
    if not req.features_json.lower().endswith("features.json"):
        issues.append({"severity": "error", "field": "features_json", "message": "Features JSON must be a features.json file (not a new_features list)."})
    elif not fj_path.exists():
        issues.append({"severity": "error", "field": "features_json", "message": f"Features file not found: {req.features_json}"})
    else:
        features_info["exists"] = True
        # Inspect structure
        try:
            data = json.loads(fj_path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict) and isinstance(data.get("feature_sets"), dict):
                features_info["kind"] = "main"
                medium = data.get("feature_sets", {}).get("medium")
                # medium can be list of strings or objects with name
                names = []
                if isinstance(medium, list):
                    for item in medium:
                        if isinstance(item, str):
                            names.append(item)
                        elif isinstance(item, dict) and "name" in item:
                            names.append(str(item.get("name")))
                features_info["medium_count"] = len(names) if names else 0
                features_info["valid_for_pipeline"] = True if names is not None else False
                # Diff vs baseline
                baseline = root / "v5.0" / "features.json"
                if baseline.exists():
                    features_diff["baseline_exists"] = True
                    try:
                        base_data = json.loads(baseline.read_text(encoding="utf-8", errors="ignore"))
                        base_names = []
                        base_medium = base_data.get("feature_sets", {}).get("medium") if isinstance(base_data, dict) else None
                        if isinstance(base_medium, list):
                            for it in base_medium:
                                if isinstance(it, str):
                                    base_names.append(it)
                                elif isinstance(it, dict) and "name" in it:
                                    base_names.append(str(it.get("name")))
                        set_new = set(names)
                        set_base = set(base_names)
                        features_diff["added"] = sorted(list(set_new - set_base))[:200]
                        features_diff["removed"] = sorted(list(set_base - set_new))[:200]
                    except Exception:
                        pass
            elif isinstance(data, list):
                features_info["kind"] = "new_features_list"
                features_info["valid_for_pipeline"] = False
                features_info["medium_count"] = len(data)
                issues.append({"severity": "error", "field": "features_json", "message": "Selected file looks like a new_features list, not a full features.json. Pick the original features.json instead."})
            else:
                features_info["kind"] = "unknown"
                issues.append({"severity": "error", "field": "features_json", "message": "Unrecognized features JSON structure."})
        except Exception as ex:
            issues.append({"severity": "error", "field": "features_json", "message": f"Failed to parse JSON: {ex}"})

    # Info about dataset
    if ds["exists"]:
        issues.append({"severity": "info", "field": "input_data", "message": f"Detected rows≈{ds['rows']}, columns={ds['columns']}, features≈{ds['num_features']}, targets≈{ds['num_targets']}, era_col={ds['has_era']}"})

    # Effective parameters
    rows_total = ds["rows"] or 0
    effective_rows = rows_total
    if req.smoke_mode and req.smoke_row_limit:
        effective_rows = min(effective_rows, int(req.smoke_row_limit))
    features_total = ds["num_features"] or 0
    features_considered = 0 if req.disable_pathfinding else features_total
    if req.smoke_mode and req.smoke_feature_limit is not None:
        features_considered = min(features_considered, int(req.smoke_feature_limit))

    # Heuristic warnings
    if not req.smoke_mode:
        if not req.disable_pathfinding and features_considered > 1500:
            issues.append({"severity": "warning", "field": "smoke_feature_limit", "message": f"Pathfinding with {features_considered} features may be slow. Consider using smoke_feature_limit or reducing features."})
        if effective_rows > 5_000_000:
            issues.append({"severity": "warning", "field": "smoke_row_limit", "message": f"Processing {effective_rows:,} rows may be slow on limited RAM."})
    else:
        if req.smoke_feature_limit is None:
            issues.append({"severity": "warning", "field": "smoke_feature_limit", "message": "Smoke mode is ON but smoke_feature_limit is not set. Defaulting to all features may negate speed gains."})
        if req.smoke_max_eras is None:
            issues.append({"severity": "warning", "field": "smoke_max_eras", "message": "Smoke mode is ON but smoke_max_eras is not set. Consider setting to 30–120."})
        if req.smoke_row_limit is None:
            issues.append({"severity": "warning", "field": "smoke_row_limit", "message": "Smoke mode is ON but smoke_row_limit is not set. Consider 100k–250k."})

    if req.max_new_features > 40:
        issues.append({"severity": "warning", "field": "max_new_features", "message": f"High max_new_features={req.max_new_features} can increase overfit risk and runtime."})

    # Estimates (very rough heuristics)
    # Base units chosen empirically; these are conservative and meant for ballpark guidance
    stage1_minutes = max(0.5, (effective_rows / 3_000_000.0)) * (0.6 if req.skip_walk_forward else 1.0)
    stage2_minutes = 0.0 if req.disable_pathfinding else (effective_rows * max(1, features_considered) / 1_200_000_000.0 + 0.5)
    stage3_minutes = 0.0 if req.disable_pathfinding else (req.max_new_features * (effective_rows / 6_000_000.0) + max(0.2, req.max_new_features * 0.02))
    total_minutes = float(stage1_minutes + stage2_minutes + stage3_minutes)

    # Memory/Disk estimates
    size_bytes = ds["size_bytes"] or 0
    memory_gb = float(max(0.5, (effective_rows * 200.0) / 1e9 + (features_considered * 0.002) + 0.5))
    disk_gb = float(max(0.3, size_bytes / 1e9 * 0.3 + req.max_new_features * 0.02 + 0.05))

    result = {
        "valid": not any(i["severity"] == "error" for i in issues),
        "issues": issues,
        "dataset": ds,
        "features": features_info,
        "features_diff": features_diff,
        "estimates": {
            "effective_rows": effective_rows,
            "features_considered": features_considered,
            "stage_minutes": {
                "stage1": round(stage1_minutes, 2),
                "stage2": round(stage2_minutes, 2),
                "stage3": round(stage3_minutes, 2),
            },
            "total_minutes": round(total_minutes, 2),
            "memory_gb": round(memory_gb, 2),
            "disk_gb": round(disk_gb, 2),
        },
    }
    return result


@app.post("/inspect")
async def inspect(req: InspectReq):
    root = Path(__file__).resolve().parents[1]
    pp = Path(req.path)
    path = pp if pp.is_absolute() else (root / req.path)
    if not path.exists():
        raise HTTPException(404, detail=f"not found: {req.path}")
    name = path.name.lower()
    out: Dict[str, Any] = {"path": str(path), "name": path.name}
    if name.endswith('.parquet'):
        out.update({"type": "parquet", **_analyze_parquet(path)})
    elif name == 'features.json':
        # Reuse preflight parser
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            info = {"exists": True, "kind": "unknown", "valid_for_pipeline": False, "medium_count": None}
            if isinstance(data, dict) and isinstance(data.get("feature_sets"), dict):
                info["kind"] = "main"
                medium = data.get("feature_sets", {}).get("medium")
                names = []
                if isinstance(medium, list):
                    for item in medium:
                        if isinstance(item, str):
                            names.append(item)
                        elif isinstance(item, dict) and "name" in item:
                            names.append(str(item.get("name")))
                info["medium_count"] = len(names) if names else 0
                info["valid_for_pipeline"] = True if names is not None else False
                # diff vs baseline
                baseline = root / "v5.0" / "features.json"
                diff = {"baseline_exists": baseline.exists(), "added": [], "removed": []}
                if baseline.exists():
                    try:
                        base_data = json.loads(baseline.read_text(encoding="utf-8", errors="ignore"))
                        base_names = []
                        base_medium = base_data.get("feature_sets", {}).get("medium") if isinstance(base_data, dict) else None
                        if isinstance(base_medium, list):
                            for it in base_medium:
                                if isinstance(it, str): base_names.append(it)
                                elif isinstance(it, dict) and "name" in it: base_names.append(str(it.get("name")))
                        set_new, set_base = set(names), set(base_names)
                        diff["added"] = sorted(list(set_new - set_base))[:200]
                        diff["removed"] = sorted(list(set_base - set_new))[:200]
                    except Exception:
                        pass
                out.update({"type": "features.json", "features": info, "features_diff": diff})
            else:
                out.update({"type": "features.json", "features": {"exists": True, "kind": "unknown", "valid_for_pipeline": False}})
        except Exception as ex:
            raise HTTPException(400, detail=f"invalid json: {ex}")
    else:
        out.update({"type": "file"})
    return out


class ApplyValidationReq(BaseModel):
    input_data: str
    era_weights: str
    relationships_file: str | None = None
    output_data: str
    max_new_features: int = 40
    row_limit: int | None = None


@app.post("/validation/apply")
async def apply_validation(body: ApplyValidationReq):
    code = ops.apply_to_validation(
        body.input_data,
        body.era_weights,
        body.relationships_file,
        body.output_data,
        body.max_new_features,
        body.row_limit,
    )
    if code != 0:
        raise HTTPException(500, detail=f"apply_bootstrap_to_validation failed with exit {code}")
    return {"status": "ok", "output": body.output_data}


class PredictReq(BaseModel):
    model: str
    data: str
    output: str
    batch_size: int = 100_000


@app.post("/predict")
async def predict(body: PredictReq):
    code = ops.generate_predictions(body.model, body.data, body.output, body.batch_size)
    if code != 0:
        raise HTTPException(500, detail=f"prediction failed with exit {code}")
    return {"status": "ok", "output": body.output}


class CompareReq(BaseModel):
    control_predictions: str
    experimental_predictions: str
    validation_data: str
    output_analysis: str
    target_col: str = "target"
    experimental_target_col: str = "adaptive_target"


@app.post("/compare")
async def compare(body: CompareReq):
    code = ops.compare_models(
        body.control_predictions,
        body.experimental_predictions,
        body.validation_data,
        body.output_analysis,
        body.target_col,
        body.experimental_target_col,
    )
    if code != 0:
        raise HTTPException(500, detail=f"comparison failed with exit {code}")
    return {"status": "ok", "output": body.output_analysis}

