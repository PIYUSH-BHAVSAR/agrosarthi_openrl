# =============================================================================
# agrosarthi_rl_env/tasks/medium.py
# MEDIUM TASK: Complete Land Prep + Sowing stages with all high tasks done
# =============================================================================
"""
Objective
---------
Starting from degraded soil (low N, wrong pH), the agent must:
  1. Amend soil (fertilize + pH correction) to reach crop-optimal range.
  2. Select a suitable crop.
  3. Complete all HIGH-importance tasks in Stage 0 (Land Prep) and Stage 1 (Sowing).
  4. Advance to Stage 2 (Vegetative) without skipping any high tasks.

Episode length : max 25 steps
Reward         : delayed — full reward only on reaching Stage 2 with all high tasks done
Success        : stage == 2 AND all high tasks in stages 0+1 completed
Failure        : max steps exceeded OR advanced stage with skipped high tasks twice
"""
from agrosarthi_rl_env.env import AgroEnv
from agrosarthi_rl_env.models import Action, ActionType
from agrosarthi_rl_env.constants import STAGE_TASKS

MEDIUM_INIT = {
    # Degraded soil — needs amendment before crop selection
    "N": 20.0, "P": 10.0, "K": 15.0,
    "ph": 5.0, "temperature": 22.0, "rainfall": 60.0,
}

MAX_STEPS_MEDIUM = 25
PENALTY_SKIP_LIMIT = 2  # fail after 2 stage advances with incomplete high tasks


class MediumTask:

    def __init__(self, seed: int = 1):
        self.env = AgroEnv(seed=seed, init_state=MEDIUM_INIT)
        self._skip_count = 0

    def reset(self):
        self._skip_count = 0
        return self.env.reset(init_state=MEDIUM_INIT)

    def step(self, action: Action):
        obs, reward, done, info = self.env.step(action)

        # Track bad advances
        if action.action_type == ActionType.ADVANCE_STAGE:
            stage_tasks = STAGE_TASKS[obs.stage - 1] if obs.stage > 0 else []
            high_count = sum(1 for t in stage_tasks if t["importance"] == "high")
            if obs.tasks_done < high_count:
                self._skip_count += 1

        # Success: reached stage 2 with no skips
        if obs.stage == 2 and self._skip_count == 0:
            done = True
            info["task_result"] = "success"

        # Failure: too many skips or max steps
        if self._skip_count >= PENALTY_SKIP_LIMIT:
            done = True
            info["task_result"] = "failure"
        elif self.env._state.step_count >= MAX_STEPS_MEDIUM:
            done = True
            info["task_result"] = "failure"

        return obs, reward, done, info

    def success_condition(self, info: dict) -> bool:
        return info.get("task_result") == "success"

    def failure_condition(self, info: dict) -> bool:
        return info.get("task_result") == "failure"
