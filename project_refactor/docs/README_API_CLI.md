# Pathfind API + CLI

This repo now includes a minimal FastAPI server to orchestrate runs and a simple Ink CLI to submit and monitor them.

## API

- Start the server:

```bash
uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload
```

- Endpoints:
  - GET /health
  - GET /runs
  - POST /runs {RunRequest}
  - GET /runs/{id}
  - GET /runs/{id}/logs
  - POST /validation/apply
  - POST /predict
  - POST /compare

## CLI

```bash
cd cli
npm install
npm start
```

Press Enter to submit a smoke run, Esc to exit. Set PATHFIND_API to point the CLI to a remote API.

## Notes

- The API delegates to existing Python scripts, keeping behavior identical to the Streamlit UI.
- Run artifacts and logs are written under pipeline_runs/ as before.
