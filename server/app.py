"""
OpenEnv-compatible HTTP server for the Code Review Assistant environment.

Endpoints:
    GET  /health     — liveness check
    GET  /metadata   — environment name and description
    GET  /schema     — action / observation / state JSON schemas
    POST /reset      — reset the environment, returns initial observation
    POST /step       — execute one action, returns observation + reward + done
    GET  /state      — current environment state

Usage:
    uv run --project . server
    python -m server.app
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import os
import sys

# Allow imports from the project root when run standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException

from environment import CodeReviewEnv
from models import Action, Observation

app = FastAPI(
    title="Code Review Assistant",
    description="OpenEnv environment for AI code review agent training",
    version="1.0.0",
)

_env = CodeReviewEnv()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "code-review-assistant",
        "description": (
            "AI agent training environment for automated code review — "
            "detects security vulnerabilities, concurrency bugs, and "
            "performance issues across difficulty-tiered scenarios."
        ),
    }


@app.get("/schema")
def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": {"type": "object", "description": "Current environment state dict"},
    }


@app.post("/reset")
def reset(task_id: Optional[str] = None):
    try:
        obs = _env.reset(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return obs.model_dump()


@app.post("/step")
def step(action: Action):
    if _env.current_state is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    obs, reward, done, info = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return _env.state()


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for uv run / python -m server.app."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
