---

title: AgroSarthi RL Environment
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
-------------

# AgroSarthi RL Environment

A reinforcement learning environment for agricultural decision-making across a full crop lifecycle.
The environment simulates real-world farming tasks such as crop selection, soil management, irrigation, and disease control.

---

## 🚀 Features

* Deterministic RL environment (seeded)
* Step-wise reward system (non-linear)
* Real-world agricultural simulation
* No external API dependency (submission-safe)
* Supports easy, medium, and hard tasks

---

## 🧠 Observation Space

* Soil nutrients: N, P, K
* Soil pH
* Growth stage (0–4)
* Moisture level
* Disease risk

---

## 🎯 Action Space

* Select crop
* Apply fertilizer
* Irrigate
* Adjust pH
* Complete stage tasks
* Advance stage
* Apply treatment
* Wait

---

## 📊 Tasks

### Easy

* Select optimal crop

### Medium

* Soil improvement + stage progression

### Hard

* Full crop lifecycle under constraints

---

## ▶️ Run Locally

```bash
python -m agrosarthi_rl_env.inference
```

---

## 🐳 Docker

The environment runs using Docker:

```bash
docker build -t agrosarthi-env .
docker run agrosarthi-env
```

---

## 📌 Notes

* LLM usage is optional and disabled by default for reproducibility
* The environment is fully deterministic and suitable for RL benchmarking


# AgroSarthi RL Environment

A reinforcement learning environment simulating agricultural decision-making across a full crop growing season — built for the OpenEnv hackathon.

---

## Overview

### The Problem

Smas. Poor decisions at any stage compound into yield loss.

### Why RL

This is a sequential decision problem with delayed consequences, irreversible failure states, and stochastic environmental dynamics — exactly the class of problem where reinforcement learning outperforms static heuristics. A good policy must plan across 5 cultivation stages, balance competing resource constraints, and adapt to events it cannot predict.

### Real-World Relevance

India has 140 million smallholder farms. nd climate state of a given farm.

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
 | Cumulative rainfall / irrigation (mm) |
| `disease_risk` | float | [0, 1] | Accumulated risk from outbreak events |
| `temperature` | float | [5, 50] | Ambient temperature (°C) |
| `stage` | int | [0, 4] | Cultivation stage |
| `tasks_done` | int | [0, 5] | Tasks completed in current stage |
| `disease_active` | int | {0, 1} | Active disease flag |

Nutrients decay every step via leaching/uptake. Weather evolves via seeded Gaussian simuge and moisture level.

### Action Space

8 discrete actions, each with typed arguments:

| Action | Arguments | Effect |
|---|---|---|
| `select_crop` | `crop_index` [1–21] | Sets crop; computes suitability score |
| `apply_fertilizer` | `n_delta`, `p_delta`, `k_delta` | Increases N/P/K; over-application spikes disease risk |
| `irrigate` | `irrigation_mm` | Increases moisture; excess raises disease probability |
| `amend_ph` | `ph_delta` [-2, +2] | Shifts soil pH toward crop optimum |
ask_index` [0–4] | Marks a cultivation task done in current stage |
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
- Disease active per step → escalo `-2.0`
- Nutrient collapse (< 40% of crop minimum) → `-0.5` per step + irreversible yield cap
- Over-fertilization (> 80 kg/ha in one step) → `-0.8` + disease spike next step
- Consecutive WAIT actions → escalating from `-0.05` to `-0.5`
- Stage budget exceeded (> 8 steps per stage) → `-0.15` per extra step

**Terminal reward** at harvest is non-linear:

```
terminal = yield_term + task_term + disease_term + collapse_term + skip_term
         + high_performance_bonus
```

.0, 1.0]`:

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

A complete 5-stage season undget (200 kg/ha total), with seeded disease events and climate shocks. The agent must balance soil management, task completion, disease response, and resource allocation.

- **Init state:** N=40, P=25, K=30, pH=6.8, temp=28°C, rain=100mm
- **Success:** `env.score() >= 0.65`
- **Failure:** `env.score() < 0.40` or crop never selected

---

## Key Features

**Deterministic simulation**
All randomness is seeded. Same seed produces identical episodes. Fully reproducible for evaluation.

**Delayed consequences**
fertilization does not penalise immediately — it sets a flag that spikes disease probability on the next step. Nutrient collapse is irreversible once triggered. Skipping high-importance tasks compounds into terminal reward deductions.

**Collapse conditions**
Three irreversible failure states terminate episodes early with strong negative reward: severe disease (8 untreated steps), nutrient collapse at late stages, and no crop selected by stage 2.

**Event-driven dynamics**
ggers a climate shock — either a drought (reduces moisture, degrades yield multiplier if critical) or a disease outbreak (spikes disease risk, forces `disease_active=1` if risk exceeds 0.8). Events are seeded and deterministic.

**Strong reward discrimination**
The reward function is calibrated so that a random policy cannot accidentally score well. Non-linear thresholds, escalating penalties, and irreversible collapse conditions ensure that only deliberate, sequential decision-making produces high scores.

---

## Baseline Performance

Evaluated over 15 episodes each, 20 steps max:

| Policy | Avg Score | Description |
|---|---|---|
| Random | ~0.05 | Picks random actions each step |
| Heuristic | ~0.55–0.62 | Treats disease, fertilizes, completes tasks in order, responds to events |

The gap of ~0.50+ demonstrates that the environment is strongly discriminative.

---

## Setup & Run

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Run inference (heuristic mode — no token needed)**

```ash
python inference.py
```

**Run with LLM (optional)**

Set `USE_LLM = True` in `inference.py`, then:

```bash
export HF_TOKEN=your_openai_compatible_token
python inference.py
```

**Run test suite**

```bash
python test_env.py
```

---

## Docker / Hugging Face Spaces

The included `Dockerfile` uses `sdk: docker` and runs the heuristic policy by default — no token required.

```bash
# Build locally
docker build -t agrosarthi-env .
docker run agrosarthi-env

# With LLM enabled (optional)
your_token agrosarthi-env
```

On Hugging Face Spaces, the Space will automatically build from the `Dockerfile` and run `python inference.py`.

---

## Project Structure

```
.
├── inference.py              # Entry point — heuristic + optional LLM policy
├── Dockerfile                # HF Spaces / Docker deployment
├── requirements.txt          # pydantic>=2.0, openai>=1.0
├── test_env.py               # Compliance + discrimination tests
├── openenv.yaml              # Environment specification
└── agrosarthi_rl_env/
    ├── env.py                # AgroEnv: reset(), step(), state(), score()
    ├── models.py             # Pydantic schemas: Observation, Action, EpisodeState
    ├── reward.py             # step_reward(), terminal_reward(), compute_score()
    ├── crop_model.py         # Rule-based suitability scoring and yield estimation
    ├── weather_sim.py        # Seeded Gaussian weather simulator
    ├── constants.py          # CROP_LIST, CROP_OPTIMA, STAGE_TASKS, BASELINE_YIELD
    ├── inference.py          # Package-level inference (mirrors root)
    └── tasks/
        ├── easy.py
        ├── medium.py
        └── hard.py
```

---

## Why This Project Stands Out

**Real-world domain with measurable impact**
Agricultural decision-making affects food security for billions of people. The environment is grounded in actual agronomic data — crop nutrient optima, disease dynamics, seasonal weather patterns — not toy physics.

**Multi-step decision making with genuine trade-offs**
The agent cannot optimise one variable in isolation. Fertilizing too aggressively causes disease. Advancing stages too quickly leaves tasks incomplete. Treating disease costs a step that could complete a task. Every decision has a cost.

**Non-trivial reward structure**
The reward function is not a simple scalar. It combines step-wise signals, irreversible state flags, exponential disease penalties, and a non-linear terminal reward. A policy that learns to navigate this structure has genuinely learned ement.

**Robustness and realism**
Seeded stochastic events (drought, disease outbreak) mean the agent must be robust, not just optimal on a fixed trajectory. The same seed always produces the same episode, ensuring fair evaluation while maintaining environmental uncertainty.

---

## Future Work

- **Multi-agent simulation** — model cooperative and competitive dynamics between neighbouring farms sharing water resources
- **Market pri agronomic yield against economic return
- **Climate variability modelling** — replace the Gaussian weather simulator with historical climate data replay for region-specific training
- **Continuous action space** — extend fertilizer and irrigation actions to continuous ranges for finer-grained optimisation
- **LLM-guided advisory layer** — use a trained policy to generate natural language recommendations for farmers via the existing OpenAI-compatible service integration
