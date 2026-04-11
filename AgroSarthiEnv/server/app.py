"""FastAPI application for AgroSarthiEnv."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv-core required. Run: uv sync") from e

try:
    from ..models import AgroSarthiEnvAction, AgroSarthiEnvObservation
    from .AgroSarthiEnv_environment import AgrosarthienvEnvironment
except (ModuleNotFoundError, ImportError):
    from models import AgroSarthiEnvAction, AgroSarthiEnvObservation
    from server.AgroSarthiEnv_environment import AgrosarthienvEnvironment

from fastapi import FastAPI
from pydantic import BaseModel

app = create_app(
    AgrosarthienvEnvironment,
    AgroSarthiEnvAction,
    AgroSarthiEnvObservation,
    env_name="AgroSarthiEnv",
    max_concurrent_envs=1,
)

# --- Grading endpoints ---

_grade_env = None

def _ensure_grade_env():
    global _grade_env
    if _grade_env is None:
        _grade_env = AgrosarthienvEnvironment()
        _grade_env.reset()
    return _grade_env


def _normalize(score: float) -> float:
    return round(max(0.01, min(0.99, float(score))), 4)


class GradeRequest(BaseModel):
    task: str = "hard"


@app.get("/grade/{task_name}")
def grade_get(task_name: str):
    env = _ensure_grade_env()
    from agrosarthi_rl_env.grader import grade
    return {"score": _normalize(grade(env._env, task_name.lower()))}


@app.post("/grade")
def grade_post(req: GradeRequest = GradeRequest()):
    env = _ensure_grade_env()
    from agrosarthi_rl_env.grader import grade
    return {"score": _normalize(grade(env._env, req.task.lower()))}


@app.get("/score")
def get_score():
    env = _ensure_grade_env()
    return {"score": env.score()}


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
