from fastapi import FastAPI, HTTPException
from typing import Dict, Any, Optional
from environment import CodeReviewEnv
from models import Action, Observation
from tasks import TASKS

app = FastAPI(
    title="Code Review Environment API",
    description="OpenEnv environment for AI code review agent training",
    version="1.0.0"
)

# Global environment instance (stateful per server)
env = CodeReviewEnv()


@app.get("/")
def health_check():
    return {"status": "ok", "environment": "code-review-assistant", "version": "1.0.0"}


@app.post("/reset", response_model=Observation)
def reset(task_id: Optional[str] = None):
    """Reset the environment. Optionally specify a task_id."""
    try:
        obs = env.reset(task_id=task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(action: Action) -> Dict[str, Any]:
    """Execute one environment step with the given action."""
    if env.current_state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }


@app.get("/state")
def get_state() -> Dict[str, Any]:
    """Get current environment state."""
    return env.state()


@app.get("/tasks")
def list_tasks():
    """List all available tasks with difficulty and description."""
    return {
        task_id: {
            "difficulty": task["difficulty"],
            "description": task["description"],
            "issue_count": len(task["issues"])
        }
        for task_id, task in TASKS.items()
    }


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    """Get details for a specific task (without revealing issues)."""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    task = TASKS[task_id]
    return {
        "task_id": task_id,
        "difficulty": task["difficulty"],
        "description": task["description"],
        "code": task["code"],
        "issue_count": len(task["issues"])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
