# AgroEnv — OpenEnv RL Environment Design Document

---

## 1. Environment Overview

**Name:** `AgroEnv-v1`

**Agent role:** The agent acts as a farm manager for one Indian crop season. It must select the right crop for the current soil/climate conditions, manage soil nutrients and irrigation across 5 cultivation stages, complete required tasks, respond to disease events, and maximise final yield.

**One episode = one crop growing season** (Land Prep → Sowing → Vegetative → Flowering → Harvest). Soil state carries over between episodes if the caller passes the final `obs` as `init_state` for the next `reset()`.

---

## 2. State Design (Observation Space)

| # | Variable | Type | Range | Evolves by |
|---|---|---|---|---|
| 1 | `N` | float | [0, 200] kg/ha | Decay each step; +delta on APPLY_FERTILIZER |
| 2 | `P` | float | [0, 200] kg/ha | Decay each step; +delta on APPLY_FERTILIZER |
| 3 | `K` | float | [0, 200] kg/ha | Decay each step; +delta on APPLY_FERTILIZER |
| 4 | `ph` | float | [0, 14] | +delta on AMEND_PH |
| 5 | `temperature` | float | [5, 50] °C | Seeded Gaussian noise each step |
| 6 | `rainfall` | float | [0, 500] mm | Seeded Gaussian noise + IRRIGATE delta |
| 7 | `stage` | int | [0, 4] | +1 on ADVANCE_STAGE |
| 8 | `tasks_done` | int | [0, 5] | +1 on COMPLETE_TASK; reset to 0 on ADVANCE_STAGE |
| 9 | `disease_active` | int | {0, 1} | Stochastic onset; 0 on APPLY_TREATMENT |
| 10 | `crop_index` | int | [0, 21] | Set once on SELECT_CROP |

State changes every step via nutrient decay and weather simulation, even on WAIT.

---

## 3. Action Space

| ID | Action | Arguments | Effect |
|---|---|---|---|
| 0 | `SELECT_CROP` | `crop_index: int [1–21]` | Sets crop; computes suitability score |
| 1 | `APPLY_FERTILIZER` | `n_delta, p_delta, k_delta: float [0–50]` | N/P/K += delta |
| 2 | `IRRIGATE` | `irrigation_mm: float [0–100]` | rainfall += mm |
| 3 | `AMEND_PH` | `ph_delta: float [-2, +2]` | ph += delta, clamped [0,14] |
| 4 | `COMPLETE_TASK` | `task_index: int [0–4]` | tasks_done += 1; returns task importance |
| 5 | `ADVANCE_STAGE` | none | stage += 1; tasks_done reset to 0 |
| 6 | `APPLY_TREATMENT` | none | disease_active → 0 |
| 7 | `WAIT` | none | No state change; -0.05 penalty |

---

## 4. State Transition Logic

```
ON SELECT_CROP(crop_index):
    IF stage == 0:
        obs.crop_index = crop_index
        state.crop_confidence = score_crop(crop_index, N, P, K, ph, temp, rain)

ON APPLY_FERTILIZER(n_delta, p_delta, k_delta):
    obs.N = clamp(obs.N + n_delta, 0, 200)
    obs.P = clamp(obs.P + p_delta, 0, 200)
    obs.K = clamp(obs.K + k_delta, 0, 200)

ON IRRIGATE(mm):
    obs.rainfall = clamp(obs.rainfall + mm, 0, 500)

ON AMEND_PH(delta):
    obs.ph = clamp(obs.ph + delta, 0, 14)

ON COMPLETE_TASK(task_index):
    IF task_index < len(STAGE_TASKS[stage]):
        obs.tasks_done += 1
        task_importance = STAGE_TASKS[stage][task_index].importance

ON ADVANCE_STAGE():
    all_high_done = (tasks_done >= count_high_tasks(stage))
    IF stage < 4:
        obs.stage += 1
        obs.tasks_done = 0
    IF stage == 4 (harvest):
        done = True
        compute terminal reward

ON APPLY_TREATMENT():
    obs.disease_active = 0
    state.disease_untreated_steps = 0

EVERY STEP (environmental dynamics):
    obs.N -= NUTRIENT_DECAY.N          # 2.0 kg/ha
    obs.P -= NUTRIENT_DECAY.P          # 0.5 kg/ha
    obs.K -= NUTRIENT_DECAY.K          # 1.0 kg/ha
    obs.temperature, obs.rainfall = weather_sim.step(stage)

EVERY STEP (disease event):
    p = DISEASE_BASE_PROB[stage]
    IF obs.rainfall > 300: p *= 2
    IF random() < p AND disease_active == 0:
        obs.disease_active = 1
```

---

## 5. Reward Function

### Step-wise rewards

| Event | Reward |
|---|---|
| SELECT_CROP with suitability >= 0.75 | +1.0 |
| SELECT_CROP with suitability 0.5–0.75 | +0.3 |
| SELECT_CROP with suitability < 0.4 | -1.0 |
| COMPLETE_TASK (high importance) | +0.5 |
| COMPLETE_TASK (medium importance) | +0.2 |
| COMPLETE_TASK (low importance) | +0.05 |
| ADVANCE_STAGE with all high tasks done | +0.3 |
| ADVANCE_STAGE with incomplete high tasks | -0.2 + -0.5 |
| APPLY_TREATMENT while disease active | +0.4 |
| Disease active (per step, any action) | -0.3 |
| Nutrient imbalance (any nutrient < 50% of crop min) | -0.1 |
| WAIT | -0.05 |

### Terminal reward (on harvest or truncation)

```
terminal = (yield / baseline)  * 5.0
         + (tasks_done / total) * 2.0
         + (untreated_steps / max_steps) * -1.0
```

---

## 6. Task Design

### EASY — Crop Selection
- **Objective:** Select the correct crop (rice) given pre-optimised soil conditions.
- **Init state:** N=90, P=45, K=45, ph=6.2, temp=27, rain=200
- **Max steps:** 5
- **Success:** SELECT_CROP with suitability_score >= 0.75 within 5 steps
- **Failure:** 5 steps elapsed without valid selection

### MEDIUM — Soil Amendment + Stage Completion
- **Objective:** Amend degraded soil, select crop, complete all high tasks in stages 0+1, advance to stage 2.
- **Init state:** N=20, P=10, K=15, ph=5.0, temp=22, rain=60
- **Max steps:** 25
- **Success:** stage == 2 with zero skipped high tasks
- **Failure:** 2+ stage advances with incomplete high tasks, or max steps exceeded

### HARD — Full Season Under Constraints
- **Objective:** Complete all 5 stages, manage disease, stay within fertilizer budget (200 kg/ha total), achieve score >= 0.65.
- **Init state:** N=40, P=25, K=30, ph=6.8, temp=28, rain=100
- **Max steps:** 60
- **Success:** `env.score() >= 0.65`
- **Failure:** `env.score() < 0.40` or crop never selected

---

## 7. Episode Flow

```
obs = env.reset(seed=42)

while True:
    action = agent.act(obs)
    obs, reward, done, truncated, info = env.step(action)

    if done:
        # Harvest reached — terminal reward already included
        final_score = env.score()
        break

    if truncated:
        # Max steps exceeded
        final_score = env.score()
        break
```

**Max steps:** 60  
**Termination conditions:**
- `done=True`: agent called ADVANCE_STAGE when stage==4 (harvest)
- `truncated=True`: step_count >= 60

---

## 8. Grader Design

```
score = 0.4 * min(yield / (2 * baseline), 1.0)
      + 0.3 * (tasks_completed / total_tasks)
      + 0.2 * (1 - untreated_steps / max_steps)
      + 0.1 * (1 - steps_taken / max_steps)

score ∈ [0.0, 1.0]
```

| Component | Weight | Measures |
|---|---|---|
| Yield ratio | 0.4 | Primary agricultural outcome |
| Task adherence | 0.3 | Following cultivation protocol |
| Disease management | 0.2 | Prompt treatment of disease |
| Efficiency | 0.1 | Completing season in fewer steps |

---

## 9. Simulation Requirements

| Original component | Replacement |
|---|---|
| `WeatherService.get_forecast()` (live API) | `WeatherSimulator` — seeded Gaussian per stage |
| `ModelManager.predict_crop()` (Keras .h5) | `crop_model.score_crop()` — rule-based range scoring |
| `ModelManager.predict_yield()` (sklearn .pkl) | `crop_model.estimate_yield()` — formula-based |
| LLM cultivation plan generation | `STAGE_TASKS` static task graph in `constants.py` |
| `SoilService.find_nearest_soil()` (CSV scan) | `init_state` dict passed to `reset()` |
| Disease detection (image + LLM) | Stochastic Bernoulli event per step |

---

## 10. Reusable Component Mapping

| Original | Status | Usage in AgroEnv |
|---|---|---|
| `CropPredictionRequest` field names (N,P,K,ph,temp,humidity,rainfall) | Reused | Observation space variable names |
| `STAGE_TASKS` structure from `_get_default_steps()` | Reused (static) | `constants.STAGE_TASKS` |
| `validate_crop_request()` bounds | Reused | Observation field validators |
| `STATES`, `DISTRICTS_YIELD`, `COMMODITIES_YIELD` | Discarded | Not needed in simulation |
| `HuggingFaceService` | Discarded | Non-deterministic, too slow |
| `WeatherService` | Discarded | Replaced by `WeatherSimulator` |
| `SoilService` CSV scan | Discarded | Replaced by `init_state` dict |
| `ModelManager.predict_yield()` | Partial | Used only as terminal reward baseline reference |
| `CROP_OPTIMA` (derived from domain knowledge) | New | Core of rule-based scoring |
| `BASELINE_YIELD` (derived from domain knowledge) | New | Terminal reward normalisation |

---

## 11. OpenEnv Interface

### Observation (Pydantic)
```python
class Observation(BaseModel):
    N: float            # [0, 200]
    P: float            # [0, 200]
    K: float            # [0, 200]
    ph: float           # [0, 14]
    temperature: float  # [5, 50]
    rainfall: float     # [0, 500]
    stage: int          # [0, 4]
    tasks_done: int     # [0, 5]
    disease_active: int # {0, 1}
    crop_index: int     # [0, 21]
```

### Action (Pydantic)
```python
class Action(BaseModel):
    action_type: ActionType          # enum 0–7
    crop_index: Optional[int]        # SELECT_CROP
    n_delta: Optional[float]         # APPLY_FERTILIZER
    p_delta: Optional[float]
    k_delta: Optional[float]
    irrigation_mm: Optional[float]   # IRRIGATE
    ph_delta: Optional[float]        # AMEND_PH
    task_index: Optional[int]        # COMPLETE_TASK
```

### `reset()` signature
```python
def reset(seed: int | None, init_state: dict | None) -> Observation
```

### `step()` signature
```python
def step(action: Action) -> tuple[Observation, float, bool, bool, dict]
# returns: (obs, reward, done, truncated, info)
```

---

## 12. File Structure

```
agrosarthi_rl_env/
├── __init__.py          # exports AgroEnv, Observation, Action, ActionType
├── env.py               # AgroEnv class — reset(), step(), score()
├── models.py            # Pydantic: Observation, Action, ActionType, EpisodeState
├── constants.py         # CROP_LIST, CROP_OPTIMA, STAGE_TASKS, BASELINE_YIELD, etc.
├── reward.py            # step_reward(), terminal_reward(), compute_score()
├── crop_model.py        # score_crop(), top_crops(), estimate_yield()
├── weather_sim.py       # WeatherSimulator (seeded Gaussian)
├── tasks/
│   ├── __init__.py
│   ├── easy.py          # EasyTask
│   ├── medium.py        # MediumTask
│   └── hard.py          # HardTask
├── design_doc.md        # this file
└── requirements.txt     # pydantic>=2.0
```

### Development order
1. `constants.py` — all static data, no dependencies
2. `models.py` — Pydantic schemas
3. `weather_sim.py` — isolated, testable
4. `crop_model.py` — isolated, testable
5. `reward.py` — depends on constants + models
6. `env.py` — integrates all above
7. `tasks/easy.py` → `medium.py` → `hard.py`
