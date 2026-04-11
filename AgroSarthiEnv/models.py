"""Data models for the AgroSarthiEnv Environment."""

from typing import Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class AgroSarthiEnvAction(Action):
    """Action for the AgroSarthiEnv environment."""
    action_type: str = Field(..., description="One of: SELECT_CROP, APPLY_FERTILIZER, IRRIGATE, AMEND_PH, COMPLETE_TASK, ADVANCE_STAGE, APPLY_TREATMENT, WAIT")
    crop_index: Optional[int] = Field(None, description="Crop index [1-21] for SELECT_CROP")
    n_delta: Optional[float] = Field(None, description="Nitrogen to add (kg/ha)")
    p_delta: Optional[float] = Field(None, description="Phosphorus to add (kg/ha)")
    k_delta: Optional[float] = Field(None, description="Potassium to add (kg/ha)")
    irrigation_mm: Optional[float] = Field(None, description="Water to add (mm)")
    ph_delta: Optional[float] = Field(None, description="pH adjustment")
    task_index: Optional[int] = Field(None, description="Task index [0-4] for COMPLETE_TASK")


class AgroSarthiEnvObservation(Observation):
    """Observation from the AgroSarthiEnv environment."""
    N: float = Field(default=50.0, description="Nitrogen kg/ha")
    P: float = Field(default=30.0, description="Phosphorus kg/ha")
    K: float = Field(default=40.0, description="Potassium kg/ha")
    ph: float = Field(default=6.5, description="Soil pH")
    temperature: float = Field(default=25.0, description="Temperature °C")
    rainfall: float = Field(default=80.0, description="Moisture mm")
    stage: int = Field(default=0, description="Cultivation stage 0-4")
    tasks_done: int = Field(default=0, description="Tasks completed in current stage")
    disease_active: int = Field(default=0, description="0=healthy 1=disease")
    crop_index: int = Field(default=0, description="Selected crop (0=none)")
