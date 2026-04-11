# =============================================================================
# agrosarthi_rl_env/models.py
# Pydantic schemas for observation space, action space, and episode state
# =============================================================================
from __future__ import annotations
from enum import IntEnum
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

class ActionType(IntEnum):
    SELECT_CROP       = 0   # Choose crop for this season
    APPLY_FERTILIZER  = 1   # Add N/P/K nutrients
    IRRIGATE          = 2   # Add water (mm)
    AMEND_PH          = 3   # Apply lime (+) or sulfur (-)
    COMPLETE_TASK     = 4   # Mark current stage task as done
    ADVANCE_STAGE     = 5   # Move to next cultivation stage
    APPLY_TREATMENT   = 6   # Treat detected disease
    WAIT              = 7   # Do nothing (costs 1 step)


class Action(BaseModel):
    action_type: ActionType

    # SELECT_CROP: index into CROP_LIST
    crop_index: Optional[int] = Field(None, ge=0, le=21)

    # APPLY_FERTILIZER: kg/ha deltas, clamped internally
    n_delta: Optional[float] = Field(None, ge=0.0, le=50.0)
    p_delta: Optional[float] = Field(None, ge=0.0, le=50.0)
    k_delta: Optional[float] = Field(None, ge=0.0, le=50.0)

    # IRRIGATE: mm of water added
    irrigation_mm: Optional[float] = Field(None, ge=0.0, le=100.0)

    # AMEND_PH: positive = lime, negative = sulfur
    ph_delta: Optional[float] = Field(None, ge=-2.0, le=2.0)

    # COMPLETE_TASK: which task index in current stage
    task_index: Optional[int] = Field(None, ge=0, le=4)


# ---------------------------------------------------------------------------
# Observation Space  (5–10 variables, all change each step)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    # Soil state (continuous, mutable)
    N: float = Field(..., ge=0.0,   le=200.0,  description="Nitrogen kg/ha")
    P: float = Field(..., ge=0.0,   le=200.0,  description="Phosphorus kg/ha")
    K: float = Field(..., ge=0.0,   le=200.0,  description="Potassium kg/ha")
    ph: float = Field(..., ge=0.0,  le=14.0,   description="Soil pH")

    # Climate state (stochastic, changes each step via weather sim)
    temperature: float = Field(..., ge=5.0,  le=50.0,  description="°C")
    rainfall: float    = Field(..., ge=0.0,  le=500.0, description="mm cumulative this stage")

    # Episode progress
    stage: int = Field(..., ge=0, le=4,
                       description="0=LandPrep 1=Sowing 2=Vegetative 3=Flowering 4=Harvest")
    tasks_done: int  = Field(..., ge=0, le=5,  description="Tasks completed in current stage")
    disease_active: int = Field(..., ge=0, le=1, description="0=healthy 1=disease present")

    # Crop selection (0 = not yet selected)
    crop_index: int = Field(..., ge=0, le=21, description="Index into CROP_LIST; 0=none")


# ---------------------------------------------------------------------------
# Episode internal state (not exposed to agent directly)
# ---------------------------------------------------------------------------

class EpisodeState(BaseModel):
    obs: Observation
    step_count: int = 0
    total_reward: float = 0.0
    disease_untreated_steps: int = 0
    crop_confidence: float = 0.0        # set on SELECT_CROP
    yield_at_harvest: float = 0.0       # set on ADVANCE_STAGE to 4
    done: bool = False
    truncated: bool = False
    info: dict = Field(default_factory=dict)

    # --- Delayed consequence tracking ---
    nutrient_collapsed: bool = False       # True once any nutrient hits collapse zone
    skipped_high_task_stages: int = 0     # count of stages advanced with incomplete high tasks
    stage_step_count: int = 0             # steps taken in the current stage
    consecutive_waits: int = 0            # consecutive WAIT actions
    overfert_pending: bool = False        # True if over-fertilization happened last step
    total_tasks_completed: int = 0        # cumulative across all stages
    # Climate shock tracking
    disease_risk: float = 0.0             # accumulated disease risk from outbreak events [0,1]
    yield_potential_multiplier: float = 1.0  # degraded by drought/outbreak events
