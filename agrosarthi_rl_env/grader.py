from agrosarthi_rl_env.constants import STAGE_TASKS


def grade_easy(env) -> float:
    """Return 1.0 if selected crop has suitability >= 0.85, else 0.0."""
    if env._state is None:
        return 0.0
    suitability = env._state.crop_confidence
    return 1.0 if suitability >= 0.85 else 0.0


def grade_medium(env) -> float:
    """Return tasks completed / total tasks, clamped to [0.0, 1.0]."""
    if env._state is None:
        return 0.0
    total_tasks = sum(len(s) for s in STAGE_TASKS)
    tasks_done = env._state.total_tasks_completed
    return min(1.0, tasks_done / max(total_tasks, 1))


def grade_hard(env) -> float:
    """Return the final normalized score from env.score()."""
    return env.score()


def grade(env, task_name: str = "hard") -> float:
    """Dispatch to task-specific grader. Returns float in [0.0, 1.0]."""
    if task_name == "easy":
        return grade_easy(env)
    elif task_name == "medium":
        return grade_medium(env)
    else:
        return grade_hard(env)
