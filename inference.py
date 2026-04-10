import os

# Load .env file if present (before reading any env vars)
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

from openai import OpenAI
from agrosarthi_rl_env import AgroEnv
from agrosarthi_rl_env.models import Action, ActionType
from agrosarthi_rl_env.constants import STAGE_TASKS

# --- ENV VARS ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")

# --- CONFIG ---
MAX_STEPS = 20
TASK_NAME = "agri-hard"
BENCHMARK = "AgroSarthiEnv"
USE_LLM   = bool(API_KEY)

# --- LLM CLIENT ---
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if USE_LLM else None

VALID_ACTIONS = [
    "SELECT_CROP", "APPLY_FERTILIZER", "IRRIGATE", "AMEND_PH",
    "COMPLETE_TASK", "ADVANCE_STAGE", "APPLY_TREATMENT", "WAIT"
]


def llm_policy(obs, info: dict) -> str:
    """Ask LLM to choose the next action. Returns action name string."""
    prompt = (
        "You are an expert agricultural decision agent managing a crop season.\n\n"
        f"Current State:\n"
        f"  Stage: {obs.stage} (0=LandPrep, 1=Sowing, 2=Vegetative, 3=Flowering, 4=Harvest)\n"
        f"  Crop selected: {'none' if obs.crop_index == 0 else obs.crop_index}\n"
        f"  Tasks done this stage: {obs.tasks_done}\n"
        f"  Soil N={obs.N:.1f} P={obs.P:.1f} K={obs.K:.1f} pH={obs.ph:.1f}\n"
        f"  Temperature={obs.temperature:.1f}°C  Moisture={obs.rainfall:.1f}mm\n"
        f"  Disease active: {'YES' if obs.disease_active else 'no'}\n"
        f"  Disease risk: {info.get('disease_risk', 0.0):.2f}\n"
        f"  Event: {info.get('event', 'none')}\n\n"
        "Available actions:\n"
        "  SELECT_CROP      - choose crop (only in stage 0, before crop selected)\n"
        "  APPLY_FERTILIZER - add N/P/K nutrients to soil\n"
        "  IRRIGATE         - add water to soil\n"
        "  AMEND_PH         - adjust soil pH\n"
        "  COMPLETE_TASK    - complete next pending task in current stage\n"
        "  ADVANCE_STAGE    - move to next cultivation stage\n"
        "  APPLY_TREATMENT  - treat active disease (use immediately if disease active)\n"
        "  WAIT             - do nothing (avoid unless necessary)\n\n"
        "Rules:\n"
        "- If disease is active, respond with APPLY_TREATMENT immediately.\n"
        "- If stage=0 and no crop selected, use SELECT_CROP.\n"
        "- Complete all tasks before advancing stage.\n"
        "- Avoid WAIT unless no other action is valid.\n\n"
        "Respond with ONLY the action name, nothing else."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip().upper()


def fallback_policy(obs, info: dict | None = None) -> Action:
    """Deterministic heuristic fallback."""
    if info is not None:
        if info.get("event") == "drought" and obs.rainfall < 40.0:
            return Action(action_type=ActionType.IRRIGATE, irrigation_mm=15.0)
        if info.get("event") == "disease_outbreak" and info.get("disease_risk", 0.0) > 0.5:
            return Action(action_type=ActionType.APPLY_TREATMENT)

    if obs.disease_active == 1:
        return Action(action_type=ActionType.APPLY_TREATMENT)

    if obs.stage == 0 and obs.crop_index == 0:
        if obs.N < 60:
            return Action(action_type=ActionType.APPLY_FERTILIZER,
                          n_delta=20.0, p_delta=10.0, k_delta=10.0)
        return Action(action_type=ActionType.SELECT_CROP, crop_index=1)

    stage_tasks = STAGE_TASKS[obs.stage]
    if obs.tasks_done < len(stage_tasks):
        return Action(action_type=ActionType.COMPLETE_TASK, task_index=obs.tasks_done)

    return Action(action_type=ActionType.ADVANCE_STAGE)


def parse_llm_action(action_text: str, obs) -> Action:
    """Map LLM output string to a typed Action. Falls back on invalid output."""
    text = action_text.strip().upper()

    # Extract first matching valid action name
    matched = next((a for a in VALID_ACTIONS if a in text), None)
    if matched is None:
        return None  # signal caller to use fallback

    if matched == "SELECT_CROP":
        return Action(action_type=ActionType.SELECT_CROP, crop_index=1)
    elif matched == "APPLY_FERTILIZER":
        return Action(action_type=ActionType.APPLY_FERTILIZER,
                      n_delta=20.0, p_delta=10.0, k_delta=10.0)
    elif matched == "IRRIGATE":
        return Action(action_type=ActionType.IRRIGATE, irrigation_mm=15.0)
    elif matched == "AMEND_PH":
        return Action(action_type=ActionType.AMEND_PH, ph_delta=0.5)
    elif matched == "COMPLETE_TASK":
        return Action(action_type=ActionType.COMPLETE_TASK, task_index=obs.tasks_done)
    elif matched == "ADVANCE_STAGE":
        return Action(action_type=ActionType.ADVANCE_STAGE)
    elif matched == "APPLY_TREATMENT":
        return Action(action_type=ActionType.APPLY_TREATMENT)
    elif matched == "WAIT":
        return Action(action_type=ActionType.WAIT)
    return None


def choose_action(step: int, obs, info: dict) -> Action:
    """Try LLM first, fall back to heuristic on any failure."""
    if USE_LLM:
        try:
            action_text = llm_policy(obs, info)
            action = parse_llm_action(action_text, obs)
            if action is not None:
                return action
        except Exception:
            pass
    return fallback_policy(obs, info)


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


def main():
    env = AgroEnv(seed=42)
    rewards = []
    steps_taken = 0
    success = False
    score = 0.0

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

    try:
        obs = env.reset()
        info: dict = {}

        for step in range(1, MAX_STEPS + 1):
            action = choose_action(step, obs, info)

            try:
                obs, reward, done, truncated, info = env.step(action)
                error = None
            except Exception as e:
                reward = 0.0
                done = True
                truncated = False
                error = str(e)
                info = {}

            rewards.append(reward)
            steps_taken = step

            print(
                f"[STEP] step={step} action={action.action_type.name} "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={error if error else 'null'}"
            )

            if done or truncated:
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
    try:
        main()
    except Exception:
        print("[END] success=false steps=0 rewards=")
