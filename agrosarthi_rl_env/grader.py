from agrosarthi_rl_env.constants import STAGE_TASKS, BASELINE_YIELD, CROP_LIST


def normalize_score(score: float) -> float:
    """Clamp score to (0.01, 0.99) — never exactly 0 or 1."""
    return round(max(0.01, min(0.99, float(score))), 4)


def grade_easy(env) -> float:
    """
    Crop Selection Intelligence.
    Evaluates how well the agent matched crop to soil/climate conditions.
    Higher suitability → higher score. Smooth, not binary.
    """
    if env._state is None:
        return normalize_score(0.01)

    suitability = env._state.crop_confidence  # float in [0.0, 1.0]

    # Smooth score: weighted blend of raw suitability + threshold bonus
    threshold_bonus = 1.0 if suitability > 0.7 else (suitability / 0.7) * 0.3
    score = 0.5 * suitability + 0.5 * threshold_bonus

    return normalize_score(score)


def grade_medium(env) -> float:
    """
    Execution Quality.
    Evaluates task completion rate weighted by task priority.
    Rewards agents that complete high-importance tasks first.
    """
    if env._state is None:
        return normalize_score(0.01)

    state = env._state
    total_tasks = sum(len(s) for s in STAGE_TASKS)
    tasks_done = state.total_tasks_completed

    # Count total high-priority tasks across all stages
    high_priority_total = sum(
        1 for stage in STAGE_TASKS for t in stage if t["importance"] == "high"
    )

    # Estimate high-priority tasks completed from skipped stages
    skipped_stages = state.skipped_high_task_stages
    high_priority_skipped = skipped_stages * 2  # avg 2 high tasks per stage
    high_priority_done = max(0, min(high_priority_total,
                                   high_priority_total - high_priority_skipped))

    task_score = tasks_done / max(total_tasks, 1)
    priority_score = high_priority_done / max(high_priority_total, 1)

    score = 0.6 * task_score + 0.4 * priority_score

    return normalize_score(score)


def grade_hard(env) -> float:
    """
    Real-World Agricultural Outcome.
    Multi-dimensional: yield quality + disease management + task efficiency.
    Reflects how a real farmer would be evaluated at season end.
    """
    if env._state is None:
        return normalize_score(0.01)

    state = env._state

    # --- Yield score ---
    crop_index = state.obs.crop_index
    if crop_index > 0:
        crop_name = CROP_LIST[crop_index]
        baseline = BASELINE_YIELD.get(crop_name, 2.0)
        yield_frac = state.yield_at_harvest / max(baseline, 0.001)
        yield_score = min(1.0, yield_frac)
    else:
        yield_score = 0.0

    # --- Disease penalty ---
    # More untreated steps = worse disease management
    from agrosarthi_rl_env.constants import MAX_STEPS
    disease_penalty = min(1.0, state.disease_untreated_steps / max(MAX_STEPS, 1) * 3.0)

    # --- Efficiency score ---
    # Reward completing more tasks relative to steps used
    total_tasks = sum(len(s) for s in STAGE_TASKS)
    task_ratio = state.total_tasks_completed / max(total_tasks, 1)
    step_ratio = state.step_count / max(MAX_STEPS, 1)
    # High tasks done quickly = efficient
    efficiency = min(1.0, task_ratio / max(step_ratio, 0.01) * 0.5)

    score = (
        0.5 * yield_score
        + 0.3 * (1.0 - disease_penalty)
        + 0.2 * efficiency
    )

    return normalize_score(score)


def grade(env, task_name: str = "hard") -> float:
    """Dispatch to task-specific grader. Returns float in (0.0, 1.0)."""
    if task_name == "easy":
        return grade_easy(env)
    elif task_name == "medium":
        return grade_medium(env)
    else:
        return grade_hard(env)
