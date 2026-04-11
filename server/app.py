"""
OpenEnv-compatible REST API server for AgroSarthi RL Environment.
Exposes: POST /reset, POST /step, GET /state, GET /health, GET+POST /grade
Runs on port 7860 for Hugging Face Spaces.
"""
import sys
import os

# Ensure the repo root (parent of server/) is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from agrosarthi_rl_env import AgroEnv
from agrosarthi_rl_env.models import Action, ActionType
from agrosarthi_rl_env.grader import grade as _grade

app = FastAPI(title="AgroSarthi RL Environment", version="1.0.0")

env: Optional[AgroEnv] = None


def ensure_env():
    global env
    if env is None:
        env = AgroEnv(seed=42)
        env.reset()


def normalize_score(score: float) -> float:
    return round(max(0.01, min(0.99, float(score))), 4)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: int = 42


class StepRequest(BaseModel):
    action_type: str
    crop_index: Optional[int] = None
    n_delta: Optional[float] = None
    p_delta: Optional[float] = None
    k_delta: Optional[float] = None
    irrigation_mm: Optional[float] = None
    ph_delta: Optional[float] = None
    task_index: Optional[int] = None


class GradeRequest(BaseModel):
    task: str = "hard"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global env
    env = AgroEnv(seed=req.seed)
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    ensure_env()
    try:
        atype = ActionType[req.action_type.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown action_type: {req.action_type}")

    action = Action(
        action_type=atype,
        crop_index=req.crop_index,
        n_delta=req.n_delta,
        p_delta=req.p_delta,
        k_delta=req.k_delta,
        irrigation_mm=req.irrigation_mm,
        ph_delta=req.ph_delta,
        task_index=req.task_index,
    )
    obs, reward, done, truncated, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "truncated": truncated,
        "info": info,
    }


@app.get("/state")
def state():
    ensure_env()
    return env.state().model_dump()


@app.get("/score")
def score():
    ensure_env()
    return {"score": env.score()}


@app.get("/grade/{task_name}")
def grade_get(task_name: str):
    ensure_env()
    s = normalize_score(_grade(env, task_name.lower()))
    return {"score": s}


@app.post("/grade")
def grade_post(body: GradeRequest = GradeRequest()):
    ensure_env()
    s = normalize_score(_grade(env, body.task.lower()))
    return {"score": s}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
