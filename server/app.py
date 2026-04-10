"""
AgroSarthi RL Environment — OpenEnv-compatible server.
Uses openenv-core create_app() for full validator compliance.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.env_server.mcp_environment import MCPEnvironment
from fastmcp import FastMCP

from agrosarthi_rl_env import AgroEnv
from agrosarthi_rl_env.models import Action as AgroAction, ActionType
from agrosarthi_rl_env.constants import STAGE_TASKS
from agrosarthi_rl_env.grader import grade_easy, grade_medium, grade_hard


class AgroSarthiEnvironment(MCPEnvironment):
    """
    AgroSarthi RL Environment wrapped as an OpenEnv MCPEnvironment.
    Exposes farming actions as MCP tools and graders for 3 tasks.
    """

    def __init__(self):
        mcp = FastMCP("AgroSarthiEnv")

        @mcp.tool
        def apply_fertilizer(n_delta: float = 20.0, p_delta: float = 10.0, k_delta: float = 10.0) -> dict:
            """Apply fertilizer to soil. Adds N/P/K nutrients (kg/ha)."""
            return self._do_step(AgroAction(
                action_type=ActionType.APPLY_FERTILIZER,
                n_delta=n_delta, p_delta=p_delta, k_delta=k_delta
            ))

        @mcp.tool
        def select_crop(crop_index: int = 1) -> dict:
            """Select a crop (1-21). Only valid in stage 0."""
            return self._do_step(AgroAction(
                action_type=ActionType.SELECT_CROP, crop_index=crop_index
            ))

        @mcp.tool
        def irrigate(irrigation_mm: float = 15.0) -> dict:
            """Add water to soil (mm)."""
            return self._do_step(AgroAction(
                action_type=ActionType.IRRIGATE, irrigation_mm=irrigation_mm
            ))

        @mcp.tool
        def complete_task(task_index: int = 0) -> dict:
            """Complete a stage task by index (0-4)."""
            return self._do_step(AgroAction(
                action_type=ActionType.COMPLETE_TASK, task_index=task_index
            ))

        @mcp.tool
        def advance_stage() -> dict:
            """Advance to the next cultivation stage."""
            return self._do_step(AgroAction(action_type=ActionType.ADVANCE_STAGE))

        @mcp.tool
        def apply_treatment() -> dict:
            """Treat active disease immediately."""
            return self._do_step(AgroAction(action_type=ActionType.APPLY_TREATMENT))

        @mcp.tool
        def amend_ph(ph_delta: float = 0.5) -> dict:
            """Adjust soil pH. Positive = lime, negative = sulfur."""
            return self._do_step(AgroAction(
                action_type=ActionType.AMEND_PH, ph_delta=ph_delta
            ))

        @mcp.tool
        def wait() -> dict:
            """Do nothing for one step."""
            return self._do_step(AgroAction(action_type=ActionType.WAIT))

        @mcp.tool
        def get_score(task: str = "hard") -> dict:
            """Get the graded score for a task: easy, medium, or hard."""
            scores = {
                "easy": grade_easy(self._env),
                "medium": grade_medium(self._env),
                "hard": grade_hard(self._env),
            }
            s = scores.get(task.lower(), grade_hard(self._env))
            return {"task": task, "score": s}

        super().__init__(mcp)
        self._env = AgroEnv(seed=42)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._last_obs = None
        self._rewards = []
        self._done = False

    def _do_step(self, agro_action: AgroAction) -> dict:
        """Execute one env step and return result dict."""
        if self._last_obs is None:
            return {"error": "Call reset first"}
        try:
            obs, reward, done, truncated, info = self._env.step(agro_action)
            self._last_obs = obs
            self._rewards.append(reward)
            self._done = done or truncated
            return {
                "observation": obs.model_dump(),
                "reward": reward,
                "done": self._done,
                "info": info,
                "score_easy": grade_easy(self._env),
                "score_medium": grade_medium(self._env),
                "score_hard": grade_hard(self._env),
            }
        except Exception as e:
            return {"error": str(e)}

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> Observation:
        self._env = AgroEnv(seed=seed or 42)
        self._last_obs = self._env.reset()
        self._rewards = []
        self._done = False
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "observation": self._last_obs.model_dump(),
                "score_easy": grade_easy(self._env),
                "score_medium": grade_medium(self._env),
                "score_hard": grade_hard(self._env),
            }
        )

    def _step_impl(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        return Observation(done=False, reward=0.0, metadata={"error": "Use MCP tools"})

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._state


app = create_app(AgroSarthiEnvironment, env_name="AgroSarthiEnv")


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
