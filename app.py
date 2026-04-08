"""
OpenEnv-compatible REST API server for AgroSarthi RL Environment.
Exposes: POST /reset, POST /step, GET /state, GET /health
Runs on port 7860 for Hugging Face Spaces.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from agrosarthi_rl_env import AgroEnv
from agrosarthi_rl_env.models import Action, ActionType

app = FastAPI(title="AgroSarthi RL Environment", version="1.0.0")

# Global env instance (single-session; HF Spaces is single-container)
env: Optional[AgroEnv] = None


# ---------------------------------------------------------------------------
# Request / Response schemas
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
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    # Map string action_type to ActionType enum
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
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    obs = env.state()
    return obs.model_dump()


@app.get("/score")
def score():
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return {"score": env.score()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
