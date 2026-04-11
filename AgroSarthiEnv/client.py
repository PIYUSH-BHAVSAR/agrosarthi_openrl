"""AgroSarthiEnv Client."""

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .models import AgroSarthiEnvAction, AgroSarthiEnvObservation


class AgroSarthiEnvEnv(EnvClient[AgroSarthiEnvAction, AgroSarthiEnvObservation, State]):
    """Client for the AgroSarthiEnv agricultural RL environment."""

    def _step_payload(self, action: AgroSarthiEnvAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[AgroSarthiEnvObservation]:
        obs_data = payload.get("observation", {})
        observation = AgroSarthiEnvObservation(
            N=obs_data.get("N", 50.0),
            P=obs_data.get("P", 30.0),
            K=obs_data.get("K", 40.0),
            ph=obs_data.get("ph", 6.5),
            temperature=obs_data.get("temperature", 25.0),
            rainfall=obs_data.get("rainfall", 80.0),
            stage=obs_data.get("stage", 0),
            tasks_done=obs_data.get("tasks_done", 0),
            disease_active=obs_data.get("disease_active", 0),
            crop_index=obs_data.get("crop_index", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
