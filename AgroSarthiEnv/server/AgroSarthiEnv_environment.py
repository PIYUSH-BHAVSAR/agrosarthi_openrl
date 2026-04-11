"""AgroSarthiEnv Environment Implementation — full agricultural RL logic."""

from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AgroSarthiEnvAction, AgroSarthiEnvObservation
except ImportError:
    from models import AgroSarthiEnvAction, AgroSarthiEnvObservation

from agrosarthi_rl_env.env import AgroEnv
from agrosarthi_rl_env.models import Action, ActionType


def _to_rl_action(action: AgroSarthiEnvAction) -> Action:
    """Convert OpenEnv action to internal RL Action."""
    try:
        atype = ActionType[action.action_type.upper()]
    except KeyError:
        atype = ActionType.WAIT
    return Action(
        action_type=atype,
        crop_index=action.crop_index,
        n_delta=action.n_delta,
        p_delta=action.p_delta,
        k_delta=action.k_delta,
        irrigation_mm=action.irrigation_mm,
        ph_delta=action.ph_delta,
        task_index=action.task_index,
    )


class AgrosarthienvEnvironment(Environment):
    """Full agricultural RL environment wrapped in OpenEnv interface."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._env = AgroEnv(seed=42)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._last_score = 0.0

    def reset(self) -> AgroSarthiEnvObservation:
        self._env = AgroEnv(seed=42)
        obs = self._env.reset()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._last_score = 0.0
        return AgroSarthiEnvObservation(
            N=obs.N, P=obs.P, K=obs.K, ph=obs.ph,
            temperature=obs.temperature, rainfall=obs.rainfall,
            stage=obs.stage, tasks_done=obs.tasks_done,
            disease_active=obs.disease_active, crop_index=obs.crop_index,
            done=False, reward=0.0,
        )

    def step(self, action: AgroSarthiEnvAction) -> AgroSarthiEnvObservation:
        rl_action = _to_rl_action(action)
        obs, reward, done, truncated, info = self._env.step(rl_action)
        self._state.step_count += 1
        if done or truncated:
            self._last_score = self._env.score()
        return AgroSarthiEnvObservation(
            N=obs.N, P=obs.P, K=obs.K, ph=obs.ph,
            temperature=obs.temperature, rainfall=obs.rainfall,
            stage=obs.stage, tasks_done=obs.tasks_done,
            disease_active=obs.disease_active, crop_index=obs.crop_index,
            done=done or truncated, reward=reward,
            metadata={"score": self._last_score, "info": info},
        )

    def score(self) -> float:
        return self._env.score()

    @property
    def state(self) -> State:
        return self._state
