# AgroSarthi RL Environment

A reinforcement learning environment simulating agricultural decision-making across a full crop growing season — built for the OpenEnv hackathon.

---

## Overview

### The Problem

Smallholder farmers in India face compounding decisions every season: which crop to plant given current soil conditions, how to manage nutrients under budget constraints, when to treat disease before it cascades, and how to respond to unpredictable climate events. Poor decisions at any stage compound into yield loss.

### Why RL

This is a sequential decision problem with delayed consequences, irreversible failure states, and stochastic environmental dynamics — exactly the class of problem where reinforcement learning outperforms static heuristics. A good policy must plan across 5 cultivation stages, balance competing resource constraints, and adapt to events it cannot predict.

### Real-World Relevance

India has 140 million smallholder farms. A trained policy from this environment could be distilled into advisory systems that recommend actions at each stage of the season — fertilizer timing, irrigation scheduling, disease response — grounded in the actual soil and climate state of a given farm.

---

## Environment Design

### State Space

The agent observes 10 variables at every step. All variables change each step via environmental dynamics, ensuring the state is never static.

| Variable | Type | Range | Description |
|---|---|---|---|
| `N` | float | [0, 200] | Nitrogen in soil (kg/ha) |
| `P` | float | [0, 200] | Phosphorus in soil (kg/ha) |
| `K` | float | [0, 200] | Potassium in soil (kg/ha) |
| `pH` | float | [0, 14] | Soil pH |
| `moisture` | float | [0, 500] | Cumulative rainfall / irrigation (mm) |
| `disease_risk` | float | [0, 1] | Accumulated risk from outbreak events |
| `temperature` | float | [5, 50] | Ambient temperature (°C) |
| `stage` | int | [0, 4] | Cultivation stage |
| `tasks_done` | int | [0, 5] | Tasks completed in current stage |
| `disease_active` | int | {0, 1} | Active disease flag |

Nutrients decay every step via leaching/uptake. Weather evolves via seeded Gaussian simulation. Disease can activate stochastically, with probability scaling by stage and moisture level.

### Action Space

8 discrete actions, each with typed arguments:

| Action | Arguments | Effect |
|---|---|---|
| `select_crop` | `crop_index` [1–21] | Sets crop; computes suitability score |
| `apply_fertilizer` | `n_delta`, `p_delta`, `k_delta` | Increases N/P/K; over-application spikes disease risk |
| `irrigate` | `irrigation_mm` | Increases moisture; excess raises disease probability |
| `amend_ph` | `ph_delta` [-2, +2] | Shifts soil pH toward crop optimum |
| `complete_task` | `task_index` [0–4] | Marks a cultivation task done in current stage |
| `advance_stage` | — | Moves to next stage; penalised if high tasks incomplete |
| `apply_treatment` | — | Clears active disease; reward scales with response speed |
| `wait` | — | No state change; escalating penalty for repeated use |

### Reward System

Rewards are step-wise, deterministic, and non-linear — designed so that a random policy cannot accidentally accumulate positive reward.

**Positive signals**

- Crop selection with suitability ≥ 0.85 → `+0.8`
- Crop selection with suitability ≥ 0.70 → `+0.3`
- Completing a high-importance task → `+0.8`
- Completing a medium-importance task → `+0.3`
- Advancing stage with all tasks complete → `+1.3` (base + perfect bonus)
- Treating disease within 1 step of onset → `+1.0`
- High-performance yield bonus at harvest → `+0.4`

**Penalties**

- Crop selection with suitability < 0.70 → `-0.5`
- Advancing stage with incomplete high tasks → `-1.5`
- Disease active per step → escalating from `-0.3` to `-2.0`
- Nutrient collapse (< 40% of crop minimum) → `-0.5` per step + irreversible yield cap
- Over-fertilization (> 80 kg/ha in one step) → `-0.8` + disease spike next step
- Consecutive WAIT actions → escalating from `-0.05` to `-0.5`
- Stage budget exceeded (> 8 steps per stage) → `-0.15` per extra step

**Terminal reward** at harvest is non-linear:

```
terminal = yield_term + task_term + disease_term + collapse_term + skip_term
         + high_performance_bonus
```

**Final score** is normalized to `[0.0, 1.0]`:

```
score = 0.40 * yield_score
      + 0.30 * task_score
      + 0.20 * disease_score
      + 0.10 * efficiency_score
      + efficiency_bonus
      - skip_deduction
```

---

## Tasks

### Easy — Crop Selection

The agent is given pre-optimised soil conditions and must select the correct crop within 5 steps.

- **Init state:** N=90, P=45, K=45, pH=6.2, temp=27°C, rain=200mm
- **Success:** `select_crop` with suitability ≥ 0.85 within 5 steps
- **Failure:** 5 steps elapsed without valid selection

### Medium — Soil Optimisation + Stage Completion

Starting from degraded soil, the agent must amend conditions, select a crop, and complete all high-importance tasks in Land Prep and Sowing before advancing to Vegetative Growth.

- **Init state:** N=20, P=10, K=15, pH=5.0, temp=22°C, rain=60mm
- **Success:** `stage == 2` with zero skipped high-importance tasks
- **Failure:** 2+ stage advances with incomplete high tasks, or 25 steps exceeded

### Hard — Full Season Planning

A complete 5-stage season under a fertilizer budget (200 kg/ha total), with seeded disease events and climate shocks. The agent must balance soil management, task completion, disease response, and resource allocation.

- **Init state:** N=40, P=25, K=30, pH=6.8, temp=28°C, rain=100mm
- **Success:** `env.score() >= 0.65`
- **Failure:** `env.score() < 0.40` or crop never selected

---

## Key Features

**Deterministic simulation**
All randomness is seeded. Same seed produces identical episodes. Fully reproducible for evaluation.

**Delayed consequences**
Over-fertilization does not penalise immediately — it sets a flag that spikes disease probability on the next step. Nutrient collapse is irreversible once triggered. Skipping high-importance tasks compounds into terminal reward deductions.

**Collapse conditions**
Three irreversible failure states terminate episodes early with strong negative reward: severe disease (8 untreated steps), nutrient collapse at late stages, and no crop selected by stage 2.

**Event-driven dynamics**
At each step, a 15% probability triggers a climate shock — either a drought (reduces moisture, degrades yield multiplier if critical) or a disease outbreak (spikes disease risk, forces `disease_active=1` if risk exceeds 0.8). Events are seeded and deterministic.

**Strong reward discrimination**
The reward function is calibrated so that a random policy cannot accidentally score well. Non-linear thresholds, escalating penalties, and irreversible collapse conditions ensure that only deliberate, sequential decision-making produces high scores.

---

## Baseline Performance

Evaluated over 15 episodes each, 60 steps max:

| Policy | Avg Score | Description |
|---|---|---|
| Random | ~0.05 | Picks random actions each step |
| Heuristic | ~0.55–0.62 | Treats disease, fertilizes, completes tasks in order, responds to events |

The gap of ~0.50+ demonstrates that the environment is strongly discriminative. A random policy hits collapse conditions (disease failure, nutrient collapse, no crop selected) frequently and accumulates large negative rewards. The heuristic avoids all collapse conditions and reaches harvest with positive yield.

This gap is the key signal that the environment is learnable — a trained RL agent has clear room to improve beyond the heuristic by optimising fertilizer timing, crop selection, and event response.

---

## Setup & Run

**Install dependencies**

```bash
pip install -r agrosarthi_rl_env/requirements.txt
```

**Set required environment variable**

```bash
export HF_TOKEN=your_huggingface_token
```

**Run inference**

```bash
python -m agrosarthi_rl_env.inference
```

**Run test suite**

```bash
python -m agrosarthi_rl_env.test_env
```

Expected test output:

```
✔ reset() returns valid Observation
✔ state() returns current Observation
✔ step() returns (obs, reward, done, info)
✔ Invalid action penalty applied
✔ Bad crop selection penalised
✔ Good crop selection rewarded
✔ Crop reward strictly suitability-based
✔ Episode terminates on failure
✔ score() in [0.0, 1.0]
✔ Over-fertilization disease spike: triggered=True

Random avg score:    0.0xxx
Heuristic avg score: 0.5xxx
Gap:                 0.4xxx
✔ Gap > 0.15 — reward signal is strongly discriminative
```

---

## Docker Support

A `Dockerfile` is included for containerised deployment.

```bash
docker build -t agrosarthi-env -f agrosarthi_rl_env/Dockerfile .
docker run -e HF_TOKEN=your_token agrosarthi-env
```

The image is based on `python:3.10-slim`, installs only `pydantic` and `openai`, and runs via module execution for correct package resolution. Compatible with Hugging Face Spaces (Docker SDK).

---

## Project Structure

```
agrosarthi_rl_env/
├── env.py            # AgroEnv: reset(), step(), state(), score()
├── models.py         # Pydantic schemas: Observation, Action, EpisodeState
├── reward.py         # step_reward(), terminal_reward(), compute_score()
├── crop_model.py     # Rule-based suitability scoring and yield estimation
├── weather_sim.py    # Seeded Gaussian weather simulator
├── constants.py      # CROP_LIST, CROP_OPTIMA, STAGE_TASKS, BASELINE_YIELD
├── inference.py      # Heuristic policy + OpenEnv-compliant output format
├── test_env.py       # Compliance + discrimination tests
├── openenv.yaml      # Environment specification
├── Dockerfile        # Container definition
├── requirements.txt  # pydantic>=2.0, openai>=1.0
└── tasks/
    ├── easy.py       # EasyTask wrapper
    ├── medium.py     # MediumTask wrapper
    └── hard.py       # HardTask wrapper
```

---

## Why This Project Stands Out

**Real-world domain with measurable impact**
Agricultural decision-making affects food security for billions of people. The environment is grounded in actual agronomic data — crop nutrient optima, disease dynamics, seasonal weather patterns — not toy physics.

**Multi-step decision making with genuine trade-offs**
The agent cannot optimise one variable in isolation. Fertilizing too aggressively causes disease. Advancing stages too quickly leaves tasks incomplete. Treating disease costs a step that could complete a task. Every decision has a cost.

**Non-trivial reward structure**
The reward function is not a simple scalar. It combines step-wise signals, irreversible state flags, exponential disease penalties, and a non-linear terminal reward. A policy that learns to navigate this structure has genuinely learned something about sequential resource management.

**Robustness and realism**
Seeded stochastic events (drought, disease outbreak) mean the agent must be robust, not just optimal on a fixed trajectory. The same seed always produces the same episode, ensuring fair evaluation while maintaining environmental uncertainty.

---

## Future Work

- **Multi-agent simulation** — model cooperative and competitive dynamics between neighbouring farms sharing water resources
- **Market pricing integration** — add commodity price signals so the agent must balance agronomic yield against economic return
- **Climate variability modelling** — replace the Gaussian weather simulator with historical climate data replay for region-specific training
- **Continuous action space** — extend fertilizer and irrigation actions to continuous ranges for finer-grained optimisation
- **LLM-guided advisory layer** — use a trained policy to generate natural language recommendations for farmers via the existing HuggingFace service integration
