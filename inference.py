import os
from agrosarthi_rl_env import AgroEnv
from agrosarthi_rl_env.models import Action, ActionType
from agrosarthi_rl_env.constants import STAGE_TASKS

# --- ENV VARS ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = "gpt-4o-mini"  # hard-locked, no env override
HF_TOKEN     = os.getenv("HF_TOKEN")  # read but only used when USE_LLM=True

# --- CONFIG FLAGS ---
USE_LLM      = False  # default for submission (manual control only)
LLM_FALLBACK = True   # always fall back to heuristic on any LLM failure

# --- CONFIG ---
MAX_STEPS = 20
TASK_NAME = "agri-hard"
BENCHMARK = "agrosarthi_env"

# --- MODEL VALIDATION ---
if USE_LLM and not MODEL_NAME.startswith("gpt"):
    raise ValueError(f"Invalid model detected: {MODEL_NAME}. Only OpenAI GPT models allowed.")

print(f"[INFO] Using model: {MODEL_NAME}")

# --- SAFE LLM INIT ---
client = None
if USE_LLM:
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN required when USE_LLM=True")
    from openai import OpenAI
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------

def action_to_str(a: Action) -> str:
    parts = [a.action_type.name]
    if a.crop_index is not None:
        parts.append(f"crop_index={a.crop_index}")
    if a.n_delta is not None:
        parts.append(f"n={a.n_delta:.1f}")
    if a.p_delta is not None:
        parts.append(f"p={a.p_delta:.1f}")
    if a.k_delta is not None:
        parts.append(f"k={a.k_delta:.1f}")
    if a.irrigation_mm is not None:
        parts.append(f"irrigation={a.irrigation_mm:.1f}")
    if a.task_index is not None:
        parts.append(f"task={a.task_index}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Heuristic policy (default, deterministic)
# ---------------------------------------------------------------------------

def choose_action(step: int, obs, info: dict | None = None) -> Action:
    # 0. Conditional event response
    if info is not None:
        if info.get("event") == "drought" and obs.rainfall < 40.0:
            return Action(action_type=ActionType.IRRIGATE, irrigation_mm=15.0)
        if info.get("event") == "disease_outbreak" and info.get("disease_risk", 0.0) > 0.5:
            return Action(action_type=ActionType.APPLY_TREATMENT)

    # 1. Treat active disease immediately
    if obs.disease_active == 1:
        return Action(action_type=ActionType.APPLY_TREATMENT)

    # 2. Stage 0: fertilize then select crop
    if obs.stage == 0 and obs.crop_index == 0:
        if obs.N < 60:
            return Action(action_type=ActionType.APPLY_FERTILIZER,
                          n_delta=20.0, p_delta=10.0, k_delta=10.0)
        return Action(action_type=ActionType.SELECT_CROP, crop_index=1)

    # 3. Complete tasks in order
    stage_tasks = STAGE_TASKS[obs.stage]
    if obs.tasks_done < len(stage_tasks):
        return Action(action_type=ActionType.COMPLETE_TASK, task_index=obs.tasks_done)

    # 4. All tasks done — advance
    return Action(action_type=ActionType.ADVANCE_STAGE)


# ---------------------------------------------------------------------------
# LLM policy (optional, experimental)
# ---------------------------------------------------------------------------

def llm_policy(obs, info: dict) -> str:
    """Call LLM and return raw action text. Raises on any failure."""
    prompt = (
        "You are an agricultural decision agent.\n\n"
        f"State: N={obs.N}, P={obs.P}, K={obs.K}, pH={obs.ph}, "
        f"moisture={obs.rainfall}, disease_risk={info.get('disease_risk', 0.0)}, "
        f"stage={obs.stage}, tasks_done={obs.tasks_done}, "
        f"disease_active={obs.disease_active}\n\n"
        "Choose ONE action from: SELECT_CROP, APPLY_FERTILIZER, IRRIGATE, "
        "APPLY_TREATMENT, COMPLETE_TASK, ADVANCE_STAGE\n\n"
        "Respond with ONLY the action name."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip().upper()


def parse_llm_action(action_text: str, obs) -> Action:
    """Convert LLM text to a typed Action. Defaults to ADVANCE_STAGE."""
    if "SELECT_CROP" in action_text:
        return Action(action_type=ActionType.SELECT_CROP, crop_index=1)
    elif "APPLY_FERTILIZER" in action_text:
        return Action(action_type=ActionType.APPLY_FERTILIZER,
                      n_delta=20.0, p_delta=10.0, k_delta=10.0)
    elif "IRRIGATE" in action_text:
        return Action(action_type=ActionType.IRRIGATE, irrigation_mm=10.0)
    elif "APPLY_TREATMENT" in action_text:
        return Action(action_type=ActionType.APPLY_TREATMENT)
    elif "COMPLETE_TASK" in action_text:
        return Action(action_type=ActionType.COMPLETE_TASK, task_index=obs.tasks_done)
    else:
        return Action(action_type=ActionType.ADVANCE_STAGE)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    env = AgroEnv(seed=42)
    rewards = []
    steps_taken = 0
    success = False

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

    try:
        obs = env.reset()
        info: dict = {}

        for step in range(1, MAX_STEPS + 1):

            # --- Action selection ---
            if USE_LLM:
                try:
                    action_text = llm_policy(obs, info)
                    action = parse_llm_action(action_text, obs)
                except Exception:
                    action = choose_action(step, obs, info)
            else:
                action = choose_action(step, obs, info)

            # --- Step environment ---
            try:
                obs, reward, done, truncated, info = env.step(action)
                error = None
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)
                info = {}

            rewards.append(reward)
            steps_taken = step

            print(
                f"[STEP] step={step} action={action_to_str(action)} "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={error if error else 'null'}"
            )

            if done:
                break

        score = env.score()
        success = score >= 0.6

    finally:
        print(
            f"[END] success={str(success).lower()} "
            f"steps={steps_taken} "
            f"rewards={','.join([f'{r:.2f}' for r in rewards])}"
        )


if __name__ == "__main__":
    main()
