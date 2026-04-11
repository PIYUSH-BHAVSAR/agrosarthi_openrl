# =============================================================================
# agrosarthi_rl_env/reward.py
# Discriminative step-wise reward — widens gap between random and heuristic
# =============================================================================
from agrosarthi_rl_env.constants import CROP_LIST, CROP_OPTIMA, STAGE_TASKS, BASELINE_YIELD
from agrosarthi_rl_env.models import Observation, Action, ActionType

# ---------------------------------------------------------------------------
# Thresholds that define collapse / bonus zones
# ---------------------------------------------------------------------------

# Crop selection
CONF_GREAT   = 0.75   # suitability >= this → large reward
CONF_OK      = 0.50   # suitability >= this → small reward
CONF_BAD     = 0.40   # suitability <  this → penalty

# Nutrient collapse: if any nutrient drops below this fraction of crop minimum
# the yield multiplier is permanently capped
NUTRIENT_COLLAPSE_FRAC = 0.40   # < 40% of crop minimum → collapse flag

# Over-fertilization: total N+P+K applied in one step above this → disease spike
OVERFERT_STEP_LIMIT = 80.0      # kg/ha in a single APPLY_FERTILIZER call

# Disease escalation: untreated steps before exponential penalty kicks in
DISEASE_ESCALATION_START = 3    # after 3 untreated steps, penalty doubles each step

# Stage sequence bonus: completing ALL tasks (not just high) before advancing
STAGE_PERFECT_BONUS = 0.8       # all 5 tasks done → bonus on top of advance reward

# Efficiency: penalise each step beyond this count per stage
STEPS_PER_STAGE_BUDGET = 8      # > 8 steps in a stage → inefficiency penalty per extra step
INEFFICIENCY_PENALTY   = -0.15  # per step over budget

# Repeated WAIT penalty (tracked externally via consecutive_waits)
CONSECUTIVE_WAIT_LIMIT = 2      # > 2 consecutive WAITs → escalating penalty
WAIT_ESCALATION        = -0.15  # per wait beyond limit (stacks)

# Terminal thresholds (non-linear)
YIELD_GREAT_FRAC = 0.80   # yield >= 80% of baseline → large bonus
YIELD_OK_FRAC    = 0.50   # yield >= 50% of baseline → small bonus
YIELD_FAIL_FRAC  = 0.30   # yield <  30% of baseline → strong penalty


# ---------------------------------------------------------------------------
# Step reward
# ---------------------------------------------------------------------------

def step_reward(
    action: Action,
    obs_before: Observation,
    obs_after: Observation,
    crop_confidence: float,
    task_importance: str | None,
    all_high_tasks_done: bool,
    all_tasks_done: bool,           # NEW: True if ALL tasks (not just high) are done
    disease_untreated_steps: int,
    stage_step_count: int,          # NEW: steps spent in current stage so far
    consecutive_waits: int,         # NEW: consecutive WAIT actions
    nutrient_collapsed: bool,       # NEW: True if nutrient collapse already triggered
    overfert_this_step: float,      # NEW: total kg/ha applied this step
) -> tuple[float, dict]:
    """
    Returns (reward, breakdown_dict).
    breakdown_dict keys: action_reward, disease_penalty, nutrient_penalty,
                         inefficiency_penalty, overfert_penalty
    """
    reward = 0.0
    breakdown = {
        "action_reward": 0.0,
        "disease_penalty": 0.0,
        "nutrient_penalty": 0.0,
        "inefficiency_penalty": 0.0,
        "overfert_penalty": 0.0,
    }
    atype = action.action_type

    # -----------------------------------------------------------------------
    # 1. Action-specific rewards
    # -----------------------------------------------------------------------

    # --- SELECT_CROP: strict boundaries — only clearly optimal gets positive reward ---
    if atype == ActionType.SELECT_CROP:
        s = crop_confidence  # always in [0.0, 1.0] from score_crop()
        if s >= 0.85:
            ar = 0.8
        elif s >= 0.7:
            ar = 0.3
        else:
            ar = -0.5   # anything < 0.7 is penalised — no exceptions
        reward += ar
        breakdown["action_reward"] = ar
        breakdown["crop_suitability"] = s

    # --- COMPLETE_TASK: reward scales with importance, bonus for sequence ---
    elif atype == ActionType.COMPLETE_TASK and task_importance is not None:
        if task_importance == "high":
            ar = 0.8    # raised from 0.5
        elif task_importance == "medium":
            ar = 0.3    # raised from 0.2
        else:
            ar = 0.1
        reward += ar
        breakdown["action_reward"] = ar

    # --- ADVANCE_STAGE: non-linear — perfect completion gives multiplier ---
    elif atype == ActionType.ADVANCE_STAGE:
        if all_tasks_done:
            # All 5 tasks done: base + perfect bonus
            ar = 0.5 + STAGE_PERFECT_BONUS
        elif all_high_tasks_done:
            # Only high tasks done: base reward
            ar = 0.5
        else:
            # Skipped high tasks: stacked penalty
            ar = -1.5   # raised from -0.7 — this is the key discriminator
        reward += ar
        breakdown["action_reward"] = ar

    # --- APPLY_TREATMENT: immediate reward, but only when disease is active ---
    elif atype == ActionType.APPLY_TREATMENT:
        if obs_before.disease_active == 1:
            # Bonus scales with how quickly treatment was applied
            if disease_untreated_steps <= 1:
                ar = 1.0    # treated immediately
            elif disease_untreated_steps <= 3:
                ar = 0.6    # treated promptly
            else:
                ar = 0.3    # treated late
            reward += ar
            breakdown["action_reward"] = ar
        else:
            # Treating when healthy wastes a step
            reward += -0.2
            breakdown["action_reward"] = -0.2

    # --- WAIT: escalating penalty for consecutive waits ---
    elif atype == ActionType.WAIT:
        if consecutive_waits > CONSECUTIVE_WAIT_LIMIT:
            w = -0.05 + WAIT_ESCALATION * (consecutive_waits - CONSECUTIVE_WAIT_LIMIT)
            w = max(w, -0.5)    # cap at -0.5 per step
        else:
            w = -0.05
        reward += w
        breakdown["action_reward"] = w

    # -----------------------------------------------------------------------
    # 2. Disease penalty — exponential escalation after threshold
    # -----------------------------------------------------------------------
    if obs_after.disease_active == 1:
        if disease_untreated_steps <= DISEASE_ESCALATION_START:
            dp = -0.3
        else:
            # Doubles every 2 steps beyond threshold, capped at -2.0
            extra = disease_untreated_steps - DISEASE_ESCALATION_START
            dp = -0.3 * (1.5 ** (extra // 2))
            dp = max(dp, -2.0)
        reward += dp
        breakdown["disease_penalty"] = dp

    # -----------------------------------------------------------------------
    # 3. Nutrient imbalance penalty — harder threshold
    # -----------------------------------------------------------------------
    if obs_after.crop_index > 0 and not nutrient_collapsed:
        crop_name = CROP_LIST[obs_after.crop_index]
        optima = CROP_OPTIMA.get(crop_name, {})
        if optima:
            n_min = optima["N"][0]
            p_min = optima["P"][0]
            k_min = optima["K"][0]
            # Collapse zone: < 40% of minimum
            if (obs_after.N < n_min * NUTRIENT_COLLAPSE_FRAC or
                    obs_after.P < p_min * NUTRIENT_COLLAPSE_FRAC or
                    obs_after.K < k_min * NUTRIENT_COLLAPSE_FRAC):
                np_ = -0.5   # severe — collapse will be flagged in env
            # Warning zone: < 60% of minimum
            elif (obs_after.N < n_min * 0.6 or
                  obs_after.P < p_min * 0.6 or
                  obs_after.K < k_min * 0.6):
                np_ = -0.2
            else:
                np_ = 0.0
            reward += np_
            breakdown["nutrient_penalty"] = np_

    # -----------------------------------------------------------------------
    # 4. Over-fertilization penalty (delayed disease spike handled in env)
    # -----------------------------------------------------------------------
    if overfert_this_step > OVERFERT_STEP_LIMIT:
        ofp = -0.8   # immediate signal; disease spike comes next step
        reward += ofp
        breakdown["overfert_penalty"] = ofp

    # -----------------------------------------------------------------------
    # 5. Inefficiency penalty — too many steps in one stage
    # -----------------------------------------------------------------------
    if stage_step_count > STEPS_PER_STAGE_BUDGET:
        ip = INEFFICIENCY_PENALTY
        reward += ip
        breakdown["inefficiency_penalty"] = ip

    return round(reward, 4), breakdown


# ---------------------------------------------------------------------------
# Terminal reward — non-linear thresholds
# ---------------------------------------------------------------------------

def terminal_reward(
    crop_index: int,
    yield_ton_ha: float,
    total_tasks: int,
    tasks_completed: int,
    disease_untreated_steps: int,
    max_steps: int,
    nutrient_collapsed: bool,
    skipped_high_task_stages: int,   # NEW: count of stages where high tasks were skipped
) -> float:
    """
    Non-linear terminal reward.
    Good policy gets large positive; random gets near-zero or negative.
    """
    if crop_index == 0:
        return -5.0   # never selected a crop — hard failure

    crop_name = CROP_LIST[crop_index]
    baseline = BASELINE_YIELD.get(crop_name, 2.0)
    yield_frac = yield_ton_ha / (baseline + 1e-6)

    # --- Yield: step-function bonus/penalty ---
    if yield_frac >= YIELD_GREAT_FRAC:
        yield_term = 6.0 + (yield_frac - YIELD_GREAT_FRAC) * 4.0   # up to +10
    elif yield_frac >= YIELD_OK_FRAC:
        yield_term = 2.0
    elif yield_frac >= YIELD_FAIL_FRAC:
        yield_term = -1.0
    else:
        yield_term = -4.0   # collapse zone

    # --- Task adherence: non-linear ---
    task_ratio = tasks_completed / max(total_tasks, 1)
    if task_ratio >= 0.9:
        task_term = 3.0
    elif task_ratio >= 0.7:
        task_term = 1.0
    elif task_ratio >= 0.5:
        task_term = 0.0
    else:
        task_term = -2.0

    # --- Disease: flat penalty per untreated step ---
    disease_term = -0.4 * disease_untreated_steps

    # --- Nutrient collapse: hard penalty ---
    collapse_term = -3.0 if nutrient_collapsed else 0.0

    # --- Skipped high-task stages: compounding penalty ---
    skip_term = -1.0 * skipped_high_task_stages

    terminal = yield_term + task_term + disease_term + collapse_term + skip_term

    # --- High-performance bonus: rewards strategies that achieve strong yield ---
    if yield_frac >= 0.85:
        terminal += 0.4
    elif yield_frac >= 0.75:
        terminal += 0.2

    return round(terminal, 4)


# ---------------------------------------------------------------------------
# Grader — score in [0.0, 1.0]
# ---------------------------------------------------------------------------

def compute_score(
    crop_index: int,
    yield_ton_ha: float,
    tasks_completed: int,
    total_tasks: int,
    disease_untreated_steps: int,
    max_steps: int,
    total_reward: float,
    steps_taken: int,
    nutrient_collapsed: bool,
    skipped_high_task_stages: int,
) -> float:
    """
    Deterministic score in [0.0, 1.0].

    Weights:
        yield_score    0.40  — non-linear: 0 below 30% baseline, 1.0 above 80%
        task_score     0.30  — non-linear: 0 below 50%, 1.0 above 90%
        disease_score  0.20  — linear decay, hard 0 if collapsed
        efficiency     0.10  — steps taken vs budget
    """
    if crop_index == 0:
        return 0.0

    crop_name = CROP_LIST[crop_index]
    baseline = BASELINE_YIELD.get(crop_name, 2.0)
    yield_frac = yield_ton_ha / (baseline + 1e-6)

    # Yield score: piecewise
    if yield_frac >= YIELD_GREAT_FRAC:
        yield_score = min(0.5 + (yield_frac - YIELD_GREAT_FRAC) * 2.5, 1.0)
    elif yield_frac >= YIELD_OK_FRAC:
        yield_score = 0.3 + (yield_frac - YIELD_OK_FRAC) / (YIELD_GREAT_FRAC - YIELD_OK_FRAC) * 0.2
    elif yield_frac >= YIELD_FAIL_FRAC:
        yield_score = 0.1
    else:
        yield_score = 0.0   # collapse — no yield credit

    # Task score: piecewise
    task_ratio = tasks_completed / max(total_tasks, 1)
    if task_ratio >= 0.9:
        task_score = 1.0
    elif task_ratio >= 0.7:
        task_score = 0.6
    elif task_ratio >= 0.5:
        task_score = 0.3
    else:
        task_score = 0.0

    # Disease score: hard 0 if nutrient collapsed (compound failure)
    if nutrient_collapsed:
        disease_score = 0.0
    else:
        disease_score = max(0.0, 1.0 - (disease_untreated_steps / max(max_steps, 1)) * 3.0)

    # Efficiency: penalise wasted steps
    efficiency = max(0.0, 1.0 - (steps_taken / max(max_steps, 1)) * 1.5)

    # Skipped high tasks: hard deduction
    skip_deduction = min(skipped_high_task_stages * 0.1, 0.3)

    # Efficiency bonus: reward finishing in fewer steps
    efficiency_bonus = max(0.0, (max_steps - steps_taken) / max_steps * 0.2)

    score = (
        0.40 * yield_score
        + 0.30 * task_score
        + 0.20 * disease_score
        + 0.10 * efficiency
        + efficiency_bonus
        - skip_deduction
    )
    return round(min(max(score, 0.0), 1.0), 4)
