---
title: Agrosarthi Env
emoji: 📈
colorFrom: gray
colorTo: pink
sdk: docker
pinned: false
---

# AgroSarthi RL Environment

A reinforcement learning environment simulating agricultural decision-making across a full crop growing season, built for the OpenEnv hackathon.

## Overview

Smallholder farmers face compounding decisions every season — which crop to plant, how to manage nutrients, when to treat disease, and how to respond to climate events. Poor decisions at any stage compound into yield loss.

This is a sequential decision problem with delayed consequences, irreversible failure states, and stochastic environmental dynamics. A good policy must plan across 5 cultivation stages, balance competing resource constraints, and adapt to unpredictable events.

## Project Structure

```
.
├── inference.py              # Entry point — heuristic + optional LLM policy
├── Dockerfile                # Container definition
├── requirements.txt          # pydantic>=2.0, openai>=1.0
├── test_env.py               # Compliance + discrimination tests
├── openenv.yaml              # OpenEnv environment specification
└── agrosarthi_rl_env/
    ├── env.py                # AgroEnv: reset(), step(), state(), score()
    ├── models.py             # Pydantic schemas: Observation, Action, EpisodeState
    ├── reward.py             # step_reward(), terminal_reward(), compute_score()
    ├── crop_model.py         # Suitability scoring and yield estimation
    ├── weather_sim.py        # Seeded Gaussian weather simulator
    ├── constants.py          # CROP_LIST, CROP_OPTIMA, STAGE_TASKS, BASELINE_YIELD
    ├── grader.py             # grade_easy(), grade_medium(), grade_hard()
    └── tasks/
        ├── easy.py
        ├── medium.py
        └── hard.py
```

## Observation Space

| Variable | Type | Range | Description |
|---|---|---|---|
| `N` | float | [0, 200] | Nitrogen in soil (kg/ha) |
| `P` | float | [0, 200] | Phosphorus in soil (kg/ha) |
| `K` | float | [0, 200] | Potassium in soil (kg/ha) |
| `ph` | float | [0, 14] | Soil pH |
| `rainfall` | float | [0, 500] | Cumulative moisture (mm) |
| `temperature` | float | [5, 50] | Ambient temperature (°C) |
| `stage` | int | [0, 4] | Cultivation stage |
| `tasks_done` | int | [0, 5] | Tasks completed in current stage |
| `disease_active` | int | {0, 1} | Active disease flag |
| `crop_index` | int | [0, 21] | Selected crop (0 = none) |

## Action Space

| Action | Arguments | Effect |
|---|---|---|
| `SELECT_CROP` | `crop_index` [1–21] | Sets crop; computes suitability score |
| `APPLY_FERTILIZER` | `n_delta`, `p_delta`, `k_delta` | Increases N/P/K; over-application spikes disease risk |
| `IRRIGATE` | `irrigation_mm` | Increases moisture |
| `AMEND_PH` | `ph_delta` [-2, +2] | Shifts soil pH |
| `COMPLETE_TASK` | `task_index` [0–4] | Marks a stage task done |
| `ADVANCE_STAGE` | — | Moves to next stage |
| `APPLY_TREATMENT` | — | Clears active disease |
| `WAIT` | — | No-op; escalating penalty |

## Tasks

### Easy — Crop Selection
Select the optimal crop given pre-optimised soil conditions within 5 steps.
- Init: N=90, P=45, K=45, pH=6.2, temp=27°C, rain=200mm
- Success: suitability ≥ 0.85

### Medium — Soil Optimisation + Stage Completion
Amend degraded soil, select a crop, and complete all high-importance tasks before stage 2.
- Init: N=20, P=10, K=15, pH=5.0, temp=22°C, rain=60mm
- Success: `stage == 2` with no skipped high tasks

### Hard — Full Season Planning
Complete a 5-stage season under fertilizer budget with disease events and climate shocks.
- Init: N=40, P=25, K=30, pH=6.8, temp=28°C, rain=100mm
- Success: `env.score() >= 0.65`

## Reward System

Step-wise, non-linear rewards — a random policy cannot accidentally score well.

**Positive:** crop suitability ≥ 0.85 → `+0.8`, high task completion → `+0.8`, disease treated in 1 step → `+1.0`, perfect stage advance → `+1.3`

**Penalties:** skipping high tasks → `-1.5`, disease per step → up to `-2.0`, nutrient collapse → `-0.5/step` + irreversible yield cap, over-fertilization → `-0.8` + disease spike

Final score normalized to `[0.0, 1.0]`:
```
score = 0.40 * yield_score + 0.30 * task_score + 0.20 * disease_score + 0.10 * efficiency_score
```

## Baseline Performance

| Policy | Avg Score | Notes |
|---|---|---|
| Random | ~0.05 | Hits collapse conditions frequently |
| Heuristic | ~0.55–0.62 | Deterministic, no LLM needed |

## Setup & Run

```bash
pip install -r requirements.txt

# Run heuristic — no token needed, fully deterministic
python inference.py

# Run with LLM (optional)
# 1. Set USE_LLM = True in inference.py
# 2. Export your token
export HF_TOKEN=your_openai_compatible_token
python inference.py

# Run tests
python test_env.py
```

## Stdout Format

The script emits exactly three line types:

```
[INFO] Using model: gpt-4o-mini
[START] task=agri-hard env=agrosarthi_env model=gpt-4o-mini
[STEP] step=1 action=APPLY_FERTILIZER n=20.0 p=10.0 k=10.0 reward=0.10 done=false error=null
[STEP] step=2 action=SELECT_CROP crop_index=1 reward=0.80 done=false error=null
...
[END] success=true steps=20 rewards=0.10,0.80,...
```

## Inference Config

| Setting | Value | Notes |
|---|---|---|
| `TASK_NAME` | `agri-hard` | Hard task, full season |
| `BENCHMARK` | `agrosarthi_env` | Environment name |
| `MAX_STEPS` | `20` | Episode step limit |
| `seed` | `42` | Fixed for reproducibility |
| `MODEL_NAME` | `gpt-4o-mini` | Hard-locked in script |
| `USE_LLM` | `False` | Default — heuristic only |
| `LLM_FALLBACK` | `True` | Falls back to heuristic on LLM error |
| `success threshold` | `score >= 0.6` | From `env.score()` |

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | No | `https://api.openai.com/v1` | LLM endpoint |
| `HF_TOKEN` | Only if `USE_LLM=True` | — | API key for LLM calls |

> `MODEL_NAME` is hard-locked to `gpt-4o-mini` in the script and is not read from the environment.

## Docker

```bash
docker build -t agrosarthi-env .

# Heuristic mode (default, no token needed)
docker run agrosarthi-env

# LLM mode (set USE_LLM=True in inference.py first)
docker run -e HF_TOKEN=your_token agrosarthi-env
```

## Key Design Properties

- **Deterministic** — all randomness is seeded; same seed = identical episode
- **Delayed consequences** — over-fertilization spikes disease probability next step; nutrient collapse is irreversible
- **Failure states** — severe disease (8 untreated steps), nutrient collapse at stage ≥ 3, no crop by stage 2
- **Climate shocks** — 15% per-step chance of drought or disease outbreak event
- **Strongly discriminative** — ~0.50 gap between random and heuristic baselines

## License

MIT
