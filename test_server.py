"""
test_server.py — Local OpenEnv server compliance validator
Run the server first: python server/app.py
Then: python test_server.py
"""
import sys
import requests

BASE_URL = "http://localhost:7860"

PASS = "[✔]"
FAIL = "[✗]"
results = []

OBS_FIELDS = {"N", "P", "K", "ph", "temperature", "rainfall",
              "stage", "tasks_done", "disease_active", "crop_index"}


def check(label: str, condition: bool, detail: str = ""):
    tag = PASS if condition else FAIL
    msg = f"{tag} {label}"
    if detail:
        msg += f": {detail}"
    print(msg)
    results.append(condition)
    return condition


def section(title: str):
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# 1. /reset
# ---------------------------------------------------------------------------
section("/reset")
try:
    r = requests.post(f"{BASE_URL}/reset", json={"seed": 42}, timeout=10)
    check("/reset status 200", r.status_code == 200, str(r.status_code))
    data = r.json()
    check("/reset returns JSON", isinstance(data, dict))
    missing = OBS_FIELDS - set(data.keys())
    check("/reset observation has all fields", len(missing) == 0,
          f"missing: {missing}" if missing else "ok")
    obs = data
except Exception as e:
    check("/reset reachable", False, str(e))
    print("Server not reachable — aborting.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. /step — single step
# ---------------------------------------------------------------------------
section("/step single")
try:
    payload = {"action_type": "APPLY_FERTILIZER", "n_delta": 10.0,
               "p_delta": 5.0, "k_delta": 5.0}
    r = requests.post(f"{BASE_URL}/step", json=payload, timeout=10)
    check("/step status 200", r.status_code == 200, str(r.status_code))
    data = r.json()
    check("/step returns JSON", isinstance(data, dict))
    check("/step has observation", "observation" in data)
    check("/step has reward (float)", isinstance(data.get("reward"), (int, float)))
    check("/step has done (bool)", isinstance(data.get("done"), bool))
    check("/step has truncated (bool)", isinstance(data.get("truncated"), bool))
    check("/step has info (dict)", isinstance(data.get("info"), dict))
    if "observation" in data:
        missing = OBS_FIELDS - set(data["observation"].keys())
        check("/step observation has all fields", len(missing) == 0,
              f"missing: {missing}" if missing else "ok")
except Exception as e:
    check("/step reachable", False, str(e))

# ---------------------------------------------------------------------------
# 3. /reset again + run 15 steps
# ---------------------------------------------------------------------------
section("/step multi-step (15 steps)")
try:
    requests.post(f"{BASE_URL}/reset", json={"seed": 42}, timeout=10)
    done = False
    step_actions = [
        {"action_type": "APPLY_FERTILIZER", "n_delta": 20.0, "p_delta": 10.0, "k_delta": 10.0},
        {"action_type": "SELECT_CROP", "crop_index": 1},
        {"action_type": "COMPLETE_TASK", "task_index": 0},
        {"action_type": "COMPLETE_TASK", "task_index": 1},
        {"action_type": "COMPLETE_TASK", "task_index": 2},
        {"action_type": "COMPLETE_TASK", "task_index": 3},
        {"action_type": "COMPLETE_TASK", "task_index": 4},
        {"action_type": "ADVANCE_STAGE"},
        {"action_type": "COMPLETE_TASK", "task_index": 0},
        {"action_type": "COMPLETE_TASK", "task_index": 1},
        {"action_type": "COMPLETE_TASK", "task_index": 2},
        {"action_type": "COMPLETE_TASK", "task_index": 3},
        {"action_type": "COMPLETE_TASK", "task_index": 4},
        {"action_type": "ADVANCE_STAGE"},
        {"action_type": "ADVANCE_STAGE"},
    ]
    crashed = False
    for i, payload in enumerate(step_actions):
        if done:
            break
        r = requests.post(f"{BASE_URL}/step", json=payload, timeout=10)
        if r.status_code != 200:
            check(f"step {i+1} status", False, str(r.status_code))
            crashed = True
            break
        data = r.json()
        done = data.get("done", False) or data.get("truncated", False)
    if not crashed:
        check("15-step run completed without crash", True)
except Exception as e:
    check("multi-step run", False, str(e))

# ---------------------------------------------------------------------------
# 4. /grade/{task}
# ---------------------------------------------------------------------------
section("/grade endpoints")
for task in ["easy", "medium", "hard"]:
    try:
        r = requests.get(f"{BASE_URL}/grade/{task}", timeout=10)
        ok_status = check(f"/grade/{task} status 200", r.status_code == 200,
                          str(r.status_code))
        if ok_status:
            data = r.json()
            has_score = "score" in data
            check(f"/grade/{task} has score field", has_score)
            if has_score:
                s = float(data["score"])
                in_range = 0.0 < s < 1.0
                check(f"/grade/{task} score in (0,1)", in_range,
                      f"{s:.4f}" if in_range else f"INVALID: {s:.4f}")
    except Exception as e:
        check(f"/grade/{task} reachable", False, str(e))

# ---------------------------------------------------------------------------
# 5. /score
# ---------------------------------------------------------------------------
section("/score")
try:
    r = requests.get(f"{BASE_URL}/score", timeout=10)
    check("/score status 200", r.status_code == 200, str(r.status_code))
    data = r.json()
    check("/score has score field", "score" in data)
    if "score" in data:
        s = float(data["score"])
        check("/score in [0.0, 1.0]", 0.0 <= s <= 1.0, f"{s:.4f}")
except Exception as e:
    check("/score reachable", False, str(e))

# ---------------------------------------------------------------------------
# 6. /health
# ---------------------------------------------------------------------------
section("/health")
try:
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    check("/health status 200", r.status_code == 200)
    check("/health returns ok", r.json().get("status") == "ok")
except Exception as e:
    check("/health reachable", False, str(e))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
passed = sum(results)
total = len(results)
print(f"\n{'='*40}")
print(f"  {passed}/{total} checks passed")
if passed == total:
    print("  ALL TESTS PASSED")
    sys.exit(0)
else:
    print(f"  SOME TESTS FAILED ({total - passed} failures)")
    sys.exit(1)
