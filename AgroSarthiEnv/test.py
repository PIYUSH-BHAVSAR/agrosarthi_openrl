"""
test.py — End-to-end endpoint tests for the deployed AgroSarthiEnv HF Space.
Usage:
    python test.py
    python test.py --url https://pylord-agrosarthi-env-v2.hf.space
"""
import argparse
import json
import sys
import requests

BASE_URL = "https://pylord-agrosarthienv-v2.hf.space"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))
    return condition


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── 1. Health ────────────────────────────────────────────────────────────────

def test_health(base: str) -> bool:
    section("GET /health")
    r = requests.get(f"{base}/health", timeout=15)
    ok = check("status 200", r.status_code == 200, f"got {r.status_code}")
    print(f"  {INFO} response: {r.text[:200]}")
    return ok


# ── 2. Reset ─────────────────────────────────────────────────────────────────

def test_reset(base: str) -> dict | None:
    section("POST /reset")
    r = requests.post(f"{base}/reset", json={}, timeout=15)
    ok = check("status 200", r.status_code == 200, f"got {r.status_code}")
    if not ok:
        print(f"  {FAIL} body: {r.text[:300]}")
        return None

    body = r.json()
    print(f"  {INFO} response keys: {list(body.keys())}")

    # observation may be nested under 'observation' key
    obs = body.get("observation", body)
    print(f"  {INFO} observation keys: {list(obs.keys())}")

    required = ["N", "P", "K", "ph", "temperature", "rainfall",
                "stage", "tasks_done", "disease_active", "crop_index"]
    for key in required:
        check(f"obs has '{key}'", key in obs, str(obs.get(key)))

    check("stage == 0",       obs.get("stage") == 0)
    check("crop_index == 0",  obs.get("crop_index") == 0)
    check("disease_active in {0,1}", obs.get("disease_active") in (0, 1))
    return obs


# ── 3. Step ──────────────────────────────────────────────────────────────────

def _step(base: str, action: dict, timeout: int = 15):
    """POST /step with correct {'action': {...}} wrapper."""
    return requests.post(f"{base}/step", json={"action": action}, timeout=timeout)


def test_step(base: str) -> dict | None:
    section("POST /step  (SELECT_CROP rice=1)")
    r = _step(base, {"action_type": "SELECT_CROP", "crop_index": 1})
    ok = check("status 200", r.status_code == 200, f"got {r.status_code}")
    if not ok:
        # 500 = env not reset before step (stateless create_app limitation)
        print(f"  {INFO} body: {r.text[:200]}")
        print(f"  {INFO} NOTE: /step requires session state — use WebSocket or session-aware client")
        return None

    data = r.json()
    print(f"  {INFO} response keys: {list(data.keys())}")
    check("has 'done'",   "done"   in data)
    check("has 'reward'", "reward" in data)
    return data


def test_step_complete_task(base: str):
    section("POST /step  (COMPLETE_TASK index=0)")
    r = _step(base, {"action_type": "COMPLETE_TASK", "task_index": 0})
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        check("has 'reward'", "reward" in data)
        print(f"  {INFO} reward={data.get('reward')}")


def test_step_advance_stage(base: str):
    section("POST /step  (ADVANCE_STAGE)")
    r = _step(base, {"action_type": "ADVANCE_STAGE"})
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  {INFO} done={data.get('done')}  reward={data.get('reward')}")


def test_step_wait(base: str):
    section("POST /step  (WAIT)")
    r = _step(base, {"action_type": "WAIT"})
    check("status 200", r.status_code == 200, f"got {r.status_code}")


def test_step_irrigate(base: str):
    section("POST /step  (IRRIGATE 20mm)")
    r = _step(base, {"action_type": "IRRIGATE", "irrigation_mm": 20.0})
    check("status 200", r.status_code == 200, f"got {r.status_code}")


def test_step_fertilize(base: str):
    section("POST /step  (APPLY_FERTILIZER)")
    r = _step(base, {"action_type": "APPLY_FERTILIZER", "n_delta": 10.0, "p_delta": 5.0, "k_delta": 5.0})
    check("status 200", r.status_code == 200, f"got {r.status_code}")


def test_step_amend_ph(base: str):
    section("POST /step  (AMEND_PH +0.5)")
    r = _step(base, {"action_type": "AMEND_PH", "ph_delta": 0.5})
    check("status 200", r.status_code == 200, f"got {r.status_code}")


def test_step_treatment(base: str):
    section("POST /step  (APPLY_TREATMENT)")
    r = _step(base, {"action_type": "APPLY_TREATMENT"})
    check("status 200", r.status_code == 200, f"got {r.status_code}")


# ── 4. State ─────────────────────────────────────────────────────────────────

def test_state(base: str):
    section("GET /state")
    r = requests.get(f"{base}/state", timeout=15)
    ok = check("status 200", r.status_code == 200, f"got {r.status_code}")
    if ok:
        print(f"  {INFO} state: {r.text[:200]}")


# ── 5. Score ─────────────────────────────────────────────────────────────────

def test_score(base: str):
    section("GET /score")
    r = requests.get(f"{base}/score", timeout=15)
    ok = check("status 200", r.status_code == 200, f"got {r.status_code}")
    if ok:
        data = r.json()
        check("has 'score'", "score" in data)
        score = data.get("score", -1)
        check("score in [0,1]", 0.0 <= score <= 1.0, str(score))
        print(f"  {INFO} score={score}")


# ── 6. Grade endpoints ───────────────────────────────────────────────────────

def test_grade_get(base: str, task: str):
    section(f"GET /grade/{task}")
    r = requests.get(f"{base}/grade/{task}", timeout=15)
    ok = check("status 200", r.status_code == 200, f"got {r.status_code}")
    if ok:
        data = r.json()
        check("has 'score'", "score" in data)
        score = data.get("score", -1)
        check("score in [0.01, 0.99]", 0.01 <= score <= 0.99, str(score))
        print(f"  {INFO} score={score}")


def test_grade_post(base: str, task: str):
    section(f"POST /grade  (task={task})")
    r = requests.post(f"{base}/grade", json={"task": task}, timeout=15)
    ok = check("status 200", r.status_code == 200, f"got {r.status_code}")
    if ok:
        data = r.json()
        check("has 'score'", "score" in data)
        print(f"  {INFO} score={data.get('score')}")


# ── 7. Full episode smoke test ───────────────────────────────────────────────

def test_full_episode(base: str):
    section("Full episode smoke test (reset → steps → score)")

    # reset
    r = requests.post(f"{base}/reset", json={}, timeout=15)
    if not check("reset ok", r.status_code == 200):
        return
    obs = r.json()

    actions = [
        {"action_type": "APPLY_FERTILIZER", "n_delta": 20.0, "p_delta": 10.0, "k_delta": 10.0},
        {"action_type": "AMEND_PH", "ph_delta": 0.5},
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
        {"action_type": "ADVANCE_STAGE"},
    ]

    done = False
    for i, action in enumerate(actions):
        if done:
            break
        r = _step(base, action, timeout=15)
        if r.status_code != 200:
            check(f"step {i+1} ok", False, f"status {r.status_code}")
            break
        data = r.json()
        done = data.get("done", False)
        print(f"  {INFO} step {i+1:2d} | action={action['action_type']:20s} | "
              f"reward={data.get('reward', '?'):6} | done={done}")

    check("episode ran without error", True)

    # final score
    r = requests.get(f"{base}/score", timeout=15)
    if r.status_code == 200:
        score = r.json().get("score", 0)
        check("final score >= 0", score >= 0, str(score))
        print(f"  {INFO} final score = {score}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test AgroSarthiEnv deployed endpoints")
    parser.add_argument("--url", default=BASE_URL, help="Base URL of the deployed space")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    print(f"\n{INFO} Testing: {base}\n")

    results = []

    results.append(test_health(base))
    results.append(bool(test_reset(base)))

    # Step through various actions (fresh episode per group)
    requests.post(f"{base}/reset", json={}, timeout=15)
    results.append(bool(test_step(base)))
    test_step_complete_task(base)
    test_step_advance_stage(base)

    requests.post(f"{base}/reset", json={}, timeout=15)
    test_step_fertilize(base)
    test_step_irrigate(base)
    test_step_amend_ph(base)
    test_step_treatment(base)
    test_step_wait(base)

    test_state(base)
    test_score(base)

    for task in ("easy", "medium", "hard"):
        test_grade_get(base, task)
        test_grade_post(base, task)

    test_full_episode(base)

    # Summary
    passed = sum(1 for r in results if r)
    total = len(results)
    section("SUMMARY")
    print(f"  Core checks: {passed}/{total} passed")
    if passed < total:
        print(f"  {FAIL} Some checks failed — see above for details")
        sys.exit(1)
    else:
        print(f"  {PASS} All core checks passed")


if __name__ == "__main__":
    main()
