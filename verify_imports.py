"""Quick import verification — run before pushing."""
import sys

checks = [
    "agrosarthi_rl_env",
    "agrosarthi_rl_env.env",
    "agrosarthi_rl_env.models",
    "agrosarthi_rl_env.constants",
    "agrosarthi_rl_env.reward",
    "agrosarthi_rl_env.grader",
    "agrosarthi_rl_env.tasks",
    "agrosarthi_rl_env.tasks.easy",
    "agrosarthi_rl_env.tasks.medium",
    "agrosarthi_rl_env.tasks.hard",
    "server.app",
]

failed = []
for mod in checks:
    try:
        __import__(mod)
        print(f"  ✔ {mod}")
    except Exception as e:
        print(f"  ✗ {mod} — {e}")
        failed.append(mod)

print()
if failed:
    print(f"FAILED: {len(failed)} imports broken")
    sys.exit(1)
else:
    print("ALL IMPORTS OK")
