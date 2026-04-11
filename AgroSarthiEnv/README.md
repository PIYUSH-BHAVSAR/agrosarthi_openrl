---
title: AgroSarthi Env
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
secrets:
  - API_BASE_URL
  - API_KEY
  - HF_TOKEN
  - MODEL_NAME
---

# AgroSarthiEnv 🌾

**OpenEnv-compatible agricultural RL environment simulating a full crop growing season.**

The agent manages soil nutrients, irrigation, disease response, and task execution across 5 cultivation stages to maximize yield under realistic constraints.

[![HF Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/pylord/AgroSarthiEnv-v2)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/PIYUSH-BHAVSAR/agrosarthi_rl_v2)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Environment Logic](#environment-logic)
- [Tasks](#tasks)
- [API Endpoints](#api-endpoints)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Grading System](#grading-system)
- [Testing](#testing)

---

## 🎯 Overview

AgroSarthiEnv is a production-ready reinforcement learning environment that simulates real-world agricultural decision-making. Built on the OpenEnv framework, it provides:

- **5 cultivation stages**: Land Prep → Sowing → Vegetative → Flowering → Harvest
- **8 action types**: Crop selection, fertilization, irrigation, pH amendment, task completion, stage advancement, disease treatment, and wait
- **10-dimensional observation space**: Soil nutrients (N, P, K, pH), climate (temperature, rainfall), stage progress, tasks, disease status, crop selection
- **Shaped reward system**: Step-wise rewards with terminal bonuses for yield optimization
- **3 difficulty levels**: Easy (crop selection), Medium (farm operations), Hard (full season optimization)

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Python     │  │   HTTP       │  │   WebSocket  │         │
│  │   Client     │  │   Client     │  │   Client     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/JSON
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI SERVER                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  OpenEnv HTTP Server (create_app)                        │  │
│  │  • POST /reset    → Initialize episode                   │  │
│  │  • POST /step     → Execute action                       │  │
│  │  • GET  /state    → Get episode metadata                 │  │
│  │  • GET  /health   → Health check                         │  │
│  │  • GET  /grade/*  → Task grading                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  AgrosarthienvEnvironment (OpenEnv wrapper)              │  │
│  │  • Wraps AgroEnv with OpenEnv interface                  │  │
│  │  • Converts between OpenEnv and internal types           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CORE RL ENVIRONMENT                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  AgroEnv (agrosarthi_rl_env/env.py)                      │  │
│  │  • State management (EpisodeState)                       │  │
│  │  • Action validation and execution                       │  │
│  │  • Environmental dynamics (weather, nutrient decay)      │  │
│  │  • Disease simulation                                    │  │
│  │  • Reward computation                                    │  │
│  │  • Episode termination logic                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Supporting Modules                                       │  │
│  │  • crop_model.py    → Crop suitability scoring           │  │
│  │  • weather_sim.py   → Stochastic weather simulation      │  │
│  │  • reward.py        → Step & terminal reward logic       │  │
│  │  • grader.py        → Task evaluation functions          │  │
│  │  • constants.py     → Domain knowledge (crops, tasks)    │  │
│  │  • models.py        → Pydantic schemas                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Client sends action → POST /step {"action": {...}}
                              ↓
2. OpenEnv HTTP Server validates request
                              ↓
3. AgrosarthienvEnvironment converts to internal Action
                              ↓
4. AgroEnv.step(action) executes:
   a. Validate action (stage constraints, parameter ranges)
   b. Apply action effects (fertilizer, irrigation, crop selection, etc.)
   c. Update environmental dynamics (weather, nutrient decay)
   d. Check disease events (probabilistic + over-fertilization triggers)
   e. Compute step reward (shaped, multi-component)
   f. Check termination conditions (harvest, failure, max steps)
   g. Return observation
                              ↓
5. AgrosarthienvEnvironment converts to OpenEnv Observation
                              ↓
6. HTTP Server serializes to JSON
                              ↓
7. Client receives {"observation": {...}, "reward": ..., "done": ...}
```

---

## 🧠 Environment Logic

### State Space

**Observation (10 dimensions):**
- `N`, `P`, `K` (float): Soil nutrients in kg/ha [0, 200]
- `ph` (float): Soil pH [0, 14]
- `temperature` (float): Temperature in °C [5, 50]
- `rainfall` (float): Cumulative rainfall in mm [0, 500]
- `stage` (int): Cultivation stage [0-4]
- `tasks_done` (int): Tasks completed in current stage [0-5]
- `disease_active` (int): Disease status {0, 1}
- `crop_index` (int): Selected crop [0-21] (0 = none)

**Internal State (not exposed):**
- `crop_confidence` (float): Crop suitability score [0, 1]
- `yield_at_harvest` (float): Final yield in ton/ha
- `disease_untreated_steps` (int): Steps with active untreated disease
- `nutrient_collapsed` (bool): Irreversible nutrient deficiency flag
- `skipped_high_task_stages` (int): Count of stages advanced without completing high-priority tasks
- `stage_step_count` (int): Steps spent in current stage
- `consecutive_waits` (int): Consecutive WAIT actions
- `overfert_pending` (bool): Over-fertilization disease spike flag
- `disease_risk` (float): Accumulated disease risk from climate shocks
- `yield_potential_multiplier` (float): Yield degradation from droughts/outbreaks

### Action Space

**8 Action Types:**

1. **SELECT_CROP** (stage 0 only)
   - `crop_index` (int): Crop to plant [1-21]
   - Effect: Sets crop, computes suitability score
   - Reward: +0.8 if suitability ≥ 0.85, +0.3 if ≥ 0.7, -0.5 otherwise

2. **APPLY_FERTILIZER**
   - `n_delta`, `p_delta`, `k_delta` (float): Nutrients to add [0, 50] kg/ha
   - Effect: Increases soil nutrients (capped at 200)
   - Penalty: -0.8 if total > 80 kg/ha (triggers disease spike)

3. **IRRIGATE**
   - `irrigation_mm` (float): Water to add [0, 100] mm
   - Effect: Increases rainfall (capped at 500)
   - Side effect: Excess (>300mm) doubles disease probability

4. **AMEND_PH**
   - `ph_delta` (float): pH adjustment [-2, 2]
   - Effect: Adjusts soil pH (clamped to [0, 14])

5. **COMPLETE_TASK**
   - `task_index` (int): Task to complete [0-4]
   - Effect: Increments tasks_done
   - Reward: +0.8 (high), +0.3 (medium), +0.1 (low priority)

6. **ADVANCE_STAGE**
   - Effect: Moves to next stage, resets tasks_done
   - Reward: +1.3 if all tasks done, +0.5 if high tasks done, -1.5 if high tasks skipped

7. **APPLY_TREATMENT**
   - Effect: Clears disease_active
   - Reward: +1.0 if treated immediately, +0.6 if ≤3 steps, +0.3 if late

8. **WAIT**
   - Effect: No-op
   - Penalty: -0.05, escalates to -0.5 for consecutive waits

### Environmental Dynamics (Applied Every Step)

1. **Weather Simulation** (stochastic, seeded)
   - Temperature: Gaussian drift based on stage
   - Rainfall: Gaussian accumulation based on stage
   - Clamped to physical bounds

2. **Nutrient Decay** (deterministic)
   - N: -2.0 kg/ha per step
   - P: -0.5 kg/ha per step
   - K: -1.0 kg/ha per step

3. **Disease Events** (probabilistic)
   - Base probability per stage: 2% (stage 0) → 15% (stage 3)
   - Doubled if rainfall > 300mm
   - Spiked to 60% if over-fertilization occurred last step

4. **Climate Shocks** (15% chance per step)
   - **Drought**: -20mm rainfall, -0.2 reward, 30% yield reduction if rainfall < 20mm
   - **Disease Outbreak**: +0.4 disease risk, -0.2 reward, 40% yield reduction if risk > 0.8

### Reward Structure

**Step Reward Components:**
- Action reward: Immediate feedback for action quality
- Disease penalty: -0.3 to -2.0 (exponential escalation)
- Nutrient penalty: -0.2 (warning) to -0.5 (collapse)
- Over-fertilization penalty: -0.8
- Inefficiency penalty: -0.15 per step beyond 8 steps/stage

**Terminal Reward (at harvest):**
- Yield score: +6.0 to +10.0 (≥80% baseline), +2.0 (≥50%), -1.0 (≥30%), -4.0 (<30%)
- Task adherence: +3.0 (≥90%), +1.0 (≥70%), 0.0 (≥50%), -2.0 (<50%)
- Disease management: -0.4 per untreated step
- Nutrient collapse: -3.0
- Skipped high tasks: -1.0 per stage
- High-performance bonus: +0.4 (≥85% yield), +0.2 (≥75%)

### Termination Conditions

**Success (done=True):**
- Reached stage 4 and executed ADVANCE_STAGE (harvest)

**Failure (done=True):**
- Disease untreated for ≥8 steps → -4.0 penalty, 10% yield
- Nutrient collapse at stage ≥3 → -3.0 penalty, 15% yield
- No crop selected by stage 2 → -3.0 penalty
- Max steps (60) exceeded → truncated=True

**Forced Stage Advance:**
- If stuck in one stage for >15 steps → auto-advance with -1.0 penalty

---

## 📊 Tasks

### Task 1: Crop Selection (Easy)

**Objective:** Select the optimal crop given pre-tuned soil conditions.

**Initial Conditions:**
```python
N=90, P=45, K=45, ph=6.2, temp=27°C, rainfall=200mm
# Optimal for rice (crop_index=1)
```

**Success Criteria:**
- Execute SELECT_CROP with suitability score ≥ 0.75
- Within 5 steps

**Expected Score:** 1.0

**Grader:** `agrosarthi_rl_env.grader:grade_easy`
- Evaluates crop suitability score
- Smooth scoring: 50% raw suitability + 50% threshold bonus

---

### Task 2: Farm Operations Planning (Medium)

**Objective:** Amend degraded soil, select crop, complete high-priority tasks in stages 0-1.

**Initial Conditions:**
```python
N=20, P=10, K=15, ph=5.0, temp=22°C, rainfall=60mm
# Degraded soil requiring amendment
```

**Success Criteria:**
- Reach stage 2 (Vegetative)
- Complete all high-importance tasks in stages 0 and 1
- No skipped high-priority tasks

**Expected Score:** 1.0

**Grader:** `agrosarthi_rl_env.grader:grade_medium`
- Evaluates task completion rate weighted by priority
- 60% task completion + 40% high-priority task completion

---

### Task 3: Full Season Optimization (Hard)

**Objective:** Maximize yield across full 5-stage season under constraints.

**Initial Conditions:**
```python
N=40, P=25, K=30, ph=6.8, temp=28°C, rainfall=100mm
# Moderate soil, not optimal
```

**Constraints:**
- Fertilizer budget: 200 kg/ha total (N+P+K) across entire episode
- Disease events guaranteed (seeded)
- Must treat diseases promptly (>3 untreated steps = yield penalty)

**Success Criteria:**
- Final score ≥ 0.65

**Expected Score:** 0.65

**Grader:** `agrosarthi_rl_env.grader:grade_hard`
- 50% yield quality (vs baseline)
- 30% disease management (1 - untreated_steps/max_steps)
- 20% efficiency (tasks/steps ratio)

---

## 🔌 API Endpoints

### Core Endpoints

**POST /reset**
```json
Request: {}
Response: {
  "observation": {
    "N": 50.0, "P": 30.0, "K": 40.0, "ph": 6.5,
    "temperature": 25.0, "rainfall": 80.0,
    "stage": 0, "tasks_done": 0,
    "disease_active": 0, "crop_index": 0
  },
  "reward": 0.0,
  "done": false
}
```

**POST /step**
```json
Request: {
  "action": {
    "action_type": "SELECT_CROP",
    "crop_index": 1
  }
}
Response: {
  "observation": {...},
  "reward": 0.8,
  "done": false
}
```

**GET /state**
```json
Response: {
  "episode_id": "uuid",
  "step_count": 5
}
```

**GET /health**
```json
Response: {"status": "healthy"}
```

### Grading Endpoints

**GET /grade/easy**
```json
Response: {"score": 0.85}
```

**GET /grade/medium**
```json
Response: {"score": 0.70}
```

**GET /grade/hard**
```json
Response: {"score": 0.65}
```

**POST /grade**
```json
Request: {"task": "easy"}
Response: {"score": 0.85}
```

**GET /score**
```json
Response: {"score": 0.4183}
```

---

## 🚀 Installation

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/PIYUSH-BHAVSAR/agrosarthi_rl_v2.git
cd agrosarthi_rl_v2

# Build Docker image
docker build -t agrosarthi-env .

# Run container
docker run -p 7860:7860 agrosarthi-env
```

### Option 2: Local Python

```bash
# Clone repository
git clone https://github.com/PIYUSH-BHAVSAR/agrosarthi_rl_v2.git
cd agrosarthi_rl_v2

# Install dependencies (requires Python 3.10+)
pip install -e .

# Run server
python -m server.app
```

### Option 3: UV (Fast)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync
git clone https://github.com/PIYUSH-BHAVSAR/agrosarthi_rl_v2.git
cd agrosarthi_rl_v2
uv sync

# Run server
uv run python -m server.app
```

---

## 💻 Usage

### Python Client

```python
from client import AgroSarthiEnvEnv
from models import AgroSarthiEnvAction

# Connect to deployed space
env = AgroSarthiEnvEnv(base_url="https://pylord-agrosarthienv-v2.hf.space")

# Reset environment
result = env.reset()
obs = result.observation
print(f"Initial state: N={obs.N}, P={obs.P}, K={obs.K}, pH={obs.ph}")

# Execute actions
action = AgroSarthiEnvAction(action_type="SELECT_CROP", crop_index=1)
result = env.step(action)
print(f"Reward: {result.reward}, Done: {result.done}")

# Complete episode
done = False
while not done:
    action = AgroSarthiEnvAction(action_type="COMPLETE_TASK", task_index=0)
    result = env.step(action)
    done = result.done
```

### HTTP Client (curl)

```bash
# Reset
curl -X POST https://pylord-agrosarthienv-v2.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{}'

# Step
curl -X POST https://pylord-agrosarthienv-v2.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "SELECT_CROP", "crop_index": 1}}'

# Grade
curl https://pylord-agrosarthienv-v2.hf.space/grade/easy
```

### LLM-Driven Inference

```bash
# Set environment variables
export API_KEY="your-openai-key"
export MODEL_NAME="gpt-4o-mini"

# Run inference script
python inference.py
```

---

## 📁 Project Structure

```
AgroSarthiEnv/
├── agrosarthi_rl_env/          # Core RL environment
│   ├── __init__.py
│   ├── env.py                  # Main AgroEnv class
│   ├── models.py               # Pydantic schemas (Action, Observation, State)
│   ├── constants.py            # Domain knowledge (crops, tasks, baselines)
│   ├── crop_model.py           # Crop suitability scoring
│   ├── weather_sim.py          # Stochastic weather simulation
│   ├── reward.py               # Step & terminal reward logic
│   ├── grader.py               # Task evaluation functions
│   └── tasks/                  # Task wrappers
│       ├── easy.py
│       ├── medium.py
│       └── hard.py
│
├── server/                     # FastAPI server
│   ├── __init__.py
│   ├── app.py                  # FastAPI app + grading endpoints
│   ├── AgroSarthiEnv_environment.py  # OpenEnv wrapper
│   ├── Dockerfile              # Server container
│   └── requirements.txt
│
├── models.py                   # OpenEnv action/observation schemas
├── client.py                   # Python HTTP client
├── inference.py                # LLM-driven baseline agent
├── test.py                     # End-to-end endpoint tests
├── openenv.yaml                # OpenEnv specification
├── pyproject.toml              # Python package config
├── Dockerfile                  # Main container
├── README.md                   # This file
└── .env                        # Environment variables (gitignored)
```

---

## 🎯 Grading System

### Grader Functions

All graders are in `agrosarthi_rl_env/grader.py`:

**`grade_easy(env) -> float`**
- Evaluates crop selection intelligence
- Smooth scoring based on suitability score
- Returns normalized score in [0.01, 0.99]

**`grade_medium(env) -> float`**
- Evaluates task execution quality
- Weighted by task priority (60% completion + 40% high-priority)
- Penalizes skipped high-priority tasks

**`grade_hard(env) -> float`**
- Multi-dimensional real-world outcome
- 50% yield quality + 30% disease management + 20% efficiency
- Includes baseline offset to prevent collapse

**`grade(env, task_name) -> float`**
- Universal dispatcher
- Routes to specific grader based on task_name

### Score Normalization

All scores are clamped to [0.01, 0.99] to avoid edge cases:
```python
def normalize_score(score: float) -> float:
    return round(max(0.01, min(0.99, float(score))), 4)
```

---

## 🧪 Testing

### Run Full Test Suite

```bash
python test.py
```

**Tests:**
- ✅ Health check
- ✅ Reset endpoint
- ✅ All 8 action types
- ✅ State endpoint
- ✅ Score endpoint
- ✅ All 3 grade endpoints (GET + POST)
- ✅ Full episode smoke test (14 steps)

### Expected Output

```
[PASS] GET /health
[PASS] POST /reset
[PASS] POST /step (SELECT_CROP)
[PASS] POST /step (COMPLETE_TASK)
[PASS] POST /step (ADVANCE_STAGE)
[PASS] POST /step (APPLY_FERTILIZER)
[PASS] POST /step (IRRIGATE)
[PASS] POST /step (AMEND_PH)
[PASS] POST /step (APPLY_TREATMENT)
[PASS] POST /step (WAIT)
[PASS] GET /state
[PASS] GET /score
[PASS] GET /grade/easy
[PASS] POST /grade (easy)
[PASS] GET /grade/medium
[PASS] POST /grade (medium)
[PASS] GET /grade/hard
[PASS] POST /grade (hard)
[PASS] Full episode smoke test

SUMMARY: Core checks: 3/3 passed
```

---

## 🔧 Configuration

### Environment Variables

Set in `.env` file or HF Space secrets:

```bash
# LLM API Configuration
API_BASE_URL=https://api.openai.com/v1
API_KEY=sk-...
HF_TOKEN=hf_...
MODEL_NAME=gpt-4o-mini
```

### OpenEnv Configuration

Edit `openenv.yaml` to customize:
- Task definitions
- Grader functions
- Observation/action space schemas
- Deployment settings

---

## 📚 References

- **OpenEnv Framework**: [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- **HF Space**: [huggingface.co/spaces/pylord/AgroSarthiEnv-v2](https://huggingface.co/spaces/pylord/AgroSarthiEnv-v2)
- **GitHub Repo**: [github.com/PIYUSH-BHAVSAR/agrosarthi_rl_v2](https://github.com/PIYUSH-BHAVSAR/agrosarthi_rl_v2)

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 👥 Authors

**AgroSarthi Team**
- Built for OpenEnv Hackathon
- Contact: [GitHub Issues](https://github.com/PIYUSH-BHAVSAR/agrosarthi_rl_v2/issues)

---

## 🙏 Acknowledgments

- Meta PyTorch for OpenEnv framework
- Hugging Face for hosting infrastructure
- OpenAI for LLM baseline support
