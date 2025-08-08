from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path

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

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/runs")
async def list_runs():
    return RUNS.list_runs()

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

@app.get("/runs/fs")
async def list_runs_fs():
    return RUNS.list_runs_fs()

@app.get("/runs/fs/{run_name}/summary")
async def get_run_summary(run_name: str):
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / "pipeline_runs" / run_name
    p = base / "run_summary.json"
    if not p.exists():
        raise HTTPException(404, detail="summary not found")
    return p.read_text(encoding="utf-8", errors="ignore")

@app.get("/runs/fs/{run_name}/logs")
async def get_run_logs(run_name: str, tail: int = 4000):
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / "pipeline_runs" / run_name
    p = base / "logs.log"
    if not p.exists():
        raise HTTPException(404, detail="logs not found")
    txt = p.read_text(encoding="utf-8", errors="ignore")
    return {"content": txt[-tail:]}


@app.get("/runs/fs/{run_name}/artifacts")
async def list_run_artifacts(run_name: str):
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / "pipeline_runs" / run_name
    if not base.exists() or not base.is_dir():
        raise HTTPException(404, detail="run not found")
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


@app.get("/runs/fs/{run_name}/performance")
async def get_run_performance(run_name: str):
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / "pipeline_runs" / run_name
    p = base / "performance_report.txt"
    if not p.exists():
        raise HTTPException(404, detail="performance report not found")
    return {"content": p.read_text(encoding="utf-8", errors="ignore")}


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

