# =============================================================================
# agrosarthi_rl_env/tasks/hard.py
# HARD TASK: Full season — maximise yield under resource and disease pressure
# =============================================================================
"""
Objective
---------
Run a complete growing season (all 5 stages) under the following constraints:
  - Fertilizer budget: total N+P+K applied <= 200 kg/ha across the episode
  - Disease events will occur (seeded to guarantee at least 2 events)
  - Agent must treat diseases promptly (untreated > 3 steps = yield penalty)
  - Final score must be >= 0.65 to succeed

Trade-offs the agent must balance:
  - Spend fertilizer budget early (good for crop selection) vs. later stages
  - Treat disease immediately (costs a step) vs. completing tasks
  - Advance stages quickly (fewer steps) vs. completing all tasks

Episode length : MAX_STEPS (60)
Reward         : full step-wise + terminal
Success        : score() >= 0.65 at episode end
Failure        : score() < 0.40 OR crop never selected
"""
from agrosarthi_rl_env.env import AgroEnv
from agrosarthi_rl_env.models import Action, ActionType
from agrosarthi_rl_env.constants import MAX_STEPS

HARD_INIT = {
    # Moderate soil — not terrible, not great
    "N": 40.0, "P": 25.0, "K": 30.0,
    "ph": 6.8, "temperature": 28.0, "rainfall": 100.0,
}

FERTILIZER_BUDGET = 200.0  # total kg/ha across all applications
SUCCESS_SCORE_THRESHOLD = 0.65
FAILURE_SCORE_THRESHOLD = 0.40


class HardTask:

    def __init__(self, seed: int = 7):
        self.env = AgroEnv(seed=seed, init_state=HARD_INIT)
        self._fertilizer_used = 0.0

    def reset(self):
        self._fertilizer_used = 0.0
        return self.env.reset(seed=7, init_state=HARD_INIT)

    def step(self, action: Action):
        # Track fertilizer budget
        if action.action_type == ActionType.APPLY_FERTILIZER:
            used = (
                (action.n_delta or 0.0)
                + (action.p_delta or 0.0)
                + (action.k_delta or 0.0)
            )
            self._fertilizer_used += used
            # Block over-budget applications
            if self._fertilizer_used > FERTILIZER_BUDGET:
                # Nullify the fertilizer action — convert to WAIT
                action = Action(action_type=ActionType.WAIT)

        obs, reward, done, info = self.env.step(action)
        info["fertilizer_used"] = self._fertilizer_used
        info["fertilizer_remaining"] = max(0.0, FERTILIZER_BUDGET - self._fertilizer_used)

        # Evaluate score at episode end
        if done:
            final_score = self.env.score()
            info["final_score"] = final_score
            if final_score >= SUCCESS_SCORE_THRESHOLD:
                info["task_result"] = "success"
            else:
                info["task_result"] = "failure"

        return obs, reward, done, info

    def success_condition(self, info: dict) -> bool:
        return info.get("task_result") == "success"

    def failure_condition(self, info: dict) -> bool:
        return info.get("task_result") == "failure"
