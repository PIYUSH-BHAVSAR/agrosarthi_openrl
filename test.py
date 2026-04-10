"""
AgroSarthiEnv — LLM-based test suite
Loads credentials from .env and uses the LLM policy from inference.py.
"""
import sys
import os

# Load .env file manually (no dotenv dependency needed)
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

# Now import inference — it reads env vars at import time
import inference as inf
from agrosarthi_rl_env.env import AgroEnv
from agrosarthi_rl_env.models import Action, ActionType
from agrosarthi_rl_env.grader import grade, normalize_score
from agrosarthi_rl_env.constants import STAGE_TASKS

PASS = "✔"
FAIL = "✗"
results = []


def check(label: str, condition: bool):
    tag = PASS if condition else FAIL
    print(f"  {tag} {label}")
    results.append(condition)


print(f"\n  LLM enabled: {inf.USE_LLM}")
print(f"  Model: {inf.MODEL_NAME}")
print(f"  API base: {inf.API_BASE_URL}")


# ---------------------------------------------------------------------------
# Helper: run episode using LLM policy from inference.py
# ---------------------------------------------------------------------------

def run_episode(seed: int = 42, max_steps: int = 20) -> AgroEnv:
    env = AgroEnv(seed=seed)
    obs = env.reset()
    info: dict = {}

    for step in range(1, max_steps + 1):
        action = inf.choose_action(step, obs, info)

        result = env.step(action)
        check("step() returns 5-tuple", len(result) == 5)
        obs, reward, done, truncated, info = result

        check("reward is float", isinstance(reward, float))
        check("done is bool", isinstance(done, bool))
        check("truncated is bool", isinstance(truncated, bool))
        check("info is dict", isinstance(info, dict))

        if done or truncated:
            break

    return env


# ---------------------------------------------------------------------------
# 1. reset()
# ---------------------------------------------------------------------------
print("\n[1] reset()")
env = AgroEnv(seed=42)
obs = env.reset()
check("reset() returns Observation", hasattr(obs, "N") and hasattr(obs, "stage"))
check("stage starts at 0", obs.stage == 0)
check("crop_index starts at 0", obs.crop_index == 0)
check("disease_active starts at 0", obs.disease_active == 0)

# ---------------------------------------------------------------------------
# 2. LLM action selection
# ---------------------------------------------------------------------------
print("\n[2] LLM / fallback action selection")
obs = env.reset()
action = inf.choose_action(1, obs, {})
check("choose_action() returns Action", isinstance(action, Action))
check("action_type is valid ActionType", isinstance(action.action_type, ActionType))

# ---------------------------------------------------------------------------
# 3. Full episode with LLM policy
# ---------------------------------------------------------------------------
print("\n[4] Full episode (LLM policy)")
env3 = run_episode(seed=42, max_steps=20)
score = env3.score()
check("score() returns float", isinstance(score, float))
check("score() in [0.0, 1.0]", 0.0 <= score <= 1.0)
print(f"     Episode score: {score:.4f}")

# ---------------------------------------------------------------------------
# 4. Grader — all three tasks
# ---------------------------------------------------------------------------
print("\n[5] Grader scores")
for task in ["easy", "medium", "hard"]:
    env_g = run_episode(seed=42, max_steps=20)
    s = grade(env_g, task)
    check(f"grade({task}) is float", isinstance(s, float))
    check(f"grade({task}) in (0, 1) exclusive", 0.0 < s < 1.0)
    print(f"     {task} score = {s}")

# ---------------------------------------------------------------------------
# 5. normalize_score()
# ---------------------------------------------------------------------------
print("\n[6] normalize_score()")
check("normalize_score(0.0) > 0", normalize_score(0.0) > 0.0)
check("normalize_score(1.0) < 1", normalize_score(1.0) < 1.0)
check("normalize_score(0.5) == 0.5", normalize_score(0.5) == 0.5)

# ---------------------------------------------------------------------------
# 6. Fallback safety — LLM failure should not crash
# ---------------------------------------------------------------------------
print("\n[7] Fallback safety")
obs = env.reset()
# Force fallback by calling fallback_policy directly
fallback_action = inf.fallback_policy(obs, {})
check("fallback_policy() returns Action", isinstance(fallback_action, Action))
check("fallback action_type valid", isinstance(fallback_action.action_type, ActionType))

# ---------------------------------------------------------------------------
# 7. Determinism — same seed = same initial obs
# ---------------------------------------------------------------------------
print("\n[8] Determinism")
env_a = AgroEnv(seed=7)
obs_a = env_a.reset()
env_b = AgroEnv(seed=7)
obs_b = env_b.reset()
check("Same seed → same initial obs", obs_a.model_dump() == obs_b.model_dump())

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
passed = sum(results)
total = len(results)
print(f"\n{'='*40}")
print(f"  {passed}/{total} checks passed")
if passed == total:
    print("  ALL TESTS PASSED")
else:
    print(f"  {total - passed} FAILED")
    sys.exit(1)
