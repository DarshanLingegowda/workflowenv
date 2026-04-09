"""
server/app.py — WorkflowEnv FastAPI server
-------------------------------------------
Exposes the OpenEnv HTTP API:
  POST /reset        → WorkflowObservation
  POST /step         → {observation, reward, done, info}
  GET  /state        → WorkflowState
  GET  /health       → {"status": "ok"}
  GET  /tasks        → list of available task IDs and prompts
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from .environment import WorkflowEnvironment
from .models import WorkflowAction, WorkflowObservation, WorkflowState, TASKS

app = FastAPI(
    title="WorkflowEnv",
    description=(
        "OpenEnv-compliant environment for training AI agents "
        "on real-world workflow automation tasks."
    ),
    version="1.0.0",
)

# Single environment instance (one session per container)
_env = WorkflowEnvironment()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"


class StepResponse(BaseModel):
    observation: WorkflowObservation
    reward:      float
    done:        bool
    info:        dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "env": "WorkflowEnv", "version": "1.0.0"}


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {"task_id": tid, "prompt": meta["prompt"]}
            for tid, meta in TASKS.items()
        ]
    }


@app.post("/reset", response_model=WorkflowObservation)
async def reset(req: ResetRequest = None):
    task_id = (req.task_id if req else None) or "easy"
    try:
        obs = _env.reset(task_id=task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(action: WorkflowAction):
    try:
        obs, reward, done, info = _env.step(action)
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=WorkflowState)
async def state():
    return _env.state()


# ---------------------------------------------------------------------------
# Entry point (for local dev)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
