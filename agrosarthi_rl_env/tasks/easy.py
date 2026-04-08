# =============================================================================
# agrosarthi_rl_env/tasks/easy.py
# EASY TASK: Select the correct crop given pre-optimised soil conditions
# =============================================================================
"""
Objective
---------
Given soil/climate conditions that are already optimal for exactly one crop,
the agent must call SELECT_CROP with the correct crop_index within 5 steps.

Episode length : max 5 steps
Reward         : immediate (+1.0 on correct selection, -1.0 on wrong)
Success        : crop selected with suitability_score >= 0.75
Failure        : 5 steps elapsed without a valid selection
"""
from agrosarthi_rl_env.env import AgroEnv
from agrosarthi_rl_env.models import Action, ActionType


EASY_INIT = {
    # Conditions tuned for "rice" (see CROP_OPTIMA)
    "N": 90.0, "P": 45.0, "K": 45.0,
    "ph": 6.2, "temperature": 27.0, "rainfall": 200.0,
}

TARGET_CROP_INDEX = 1  # "rice"
MAX_STEPS_EASY = 5


class EasyTask:
    """
    Wraps AgroEnv with easy-task constraints.
    """

    def __init__(self, seed: int = 0):
        self.env = AgroEnv(seed=seed, init_state=EASY_INIT)
        self.env.metadata["max_steps"] = MAX_STEPS_EASY

    def reset(self):
        return self.env.reset(init_state=EASY_INIT)

    def step(self, action: Action):
        obs, reward, done, info = self.env.step(action)

        # Override done: success if correct crop selected with high confidence
        if (action.action_type == ActionType.SELECT_CROP
                and info["crop_confidence"] >= 0.75):
            done = True
            info["task_result"] = "success"
        elif self.env._state.step_count >= MAX_STEPS_EASY:
            done = True
            info["task_result"] = "failure"

        return obs, reward, done, info

    def success_condition(self, info: dict) -> bool:
        return info.get("task_result") == "success"

    def failure_condition(self, info: dict) -> bool:
        return info.get("task_result") == "failure"
