# =============================================================================
# agrosarthi_rl_env/env.py
# AgroEnv — OpenEnv-compatible RL environment (discriminative upgrade)
# =============================================================================
from __future__ import annotations
import logging
import random

from agrosarthi_rl_env.models import Observation, Action, ActionType, EpisodeState
from agrosarthi_rl_env.constants import (
    CROP_LIST, CROP_OPTIMA, STAGE_TASKS, MAX_STEPS,
    NUTRIENT_DECAY, DISEASE_BASE_PROB
)
from agrosarthi_rl_env.weather_sim import WeatherSimulator
from agrosarthi_rl_env.crop_model import score_crop, estimate_yield
from agrosarthi_rl_env.reward import (
    step_reward, terminal_reward,
    NUTRIENT_COLLAPSE_FRAC, OVERFERT_STEP_LIMIT,
    DISEASE_ESCALATION_START
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Failure thresholds
# ---------------------------------------------------------------------------
DISEASE_FAILURE_STEPS = 8       # lowered from 10 — random hits this more often
STAGE_BUDGET_HARD_CAP = 15      # > 15 steps in one stage → forced advance (penalised)

DEFAULT_INIT = {
    "N": 50.0, "P": 30.0, "K": 40.0, "ph": 6.5,
    "temperature": 25.0, "rainfall": 80.0,
}


class AgroEnv:
    """
    OpenEnv-compatible agricultural RL environment.

    Interface
    ---------
    env = AgroEnv(seed=42)
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    score = env.score()   # call after done=True
    """

    metadata = {"name": "AgroEnv-v1", "max_steps": MAX_STEPS}

    def __init__(self, seed: int = 42, init_state: dict | None = None, debug: bool = False):
        self.seed = seed
        self.init_state = init_state or DEFAULT_INIT.copy()
        self.weather_sim = WeatherSimulator(seed=seed)
        self._rng = random.Random(seed)
        self._state: EpisodeState | None = None
        self.debug = debug

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, init_state: dict | None = None) -> Observation:
        if seed is not None:
            self.seed = seed
        if init_state is not None:
            self.init_state = init_state

        self._rng = random.Random(self.seed)
        self.weather_sim.reset(self.seed)

        s = self.init_state
        obs = Observation(
            N=s.get("N", 50.0),
            P=s.get("P", 30.0),
            K=s.get("K", 40.0),
            ph=s.get("ph", 6.5),
            temperature=s.get("temperature", 25.0),
            rainfall=s.get("rainfall", 80.0),
            stage=0,
            tasks_done=0,
            disease_active=0,
            crop_index=0,
        )
        self._state = EpisodeState(obs=obs)
        return obs.model_copy()

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------
    def state(self) -> Observation:
        if self._state is None:
            raise RuntimeError("Call reset() before state()")
        return self._state.obs.model_copy()

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------
    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        state = self._state
        obs = state.obs
        atype = action.action_type
        obs_before = obs.model_copy()

        if self.debug:
            logger.debug("[BEFORE] step=%d action=%s stage=%d tasks=%d disease=%d",
                         state.step_count, atype.name, obs.stage,
                         obs.tasks_done, obs.disease_active)

        # ----------------------------------------------------------------
        # 1. Validate action
        # ----------------------------------------------------------------
        invalid_reason = self._validate_action(action, obs)
        if invalid_reason:
            penalty = self._invalid_action_penalty(atype)
            state.step_count += 1
            state.stage_step_count += 1
            state.total_reward += penalty
            obs = self._apply_env_dynamics(obs)
            obs = self._update_disease(obs, state)
            state.obs = obs
            done = state.step_count >= MAX_STEPS
            truncated = done
            info = self._build_info(state, obs, penalty, invalid_reason, None)
            return obs.model_copy(), round(penalty, 4), done, truncated, info

        # ----------------------------------------------------------------
        # 2. Compute over-fertilization amount BEFORE applying
        # ----------------------------------------------------------------
        overfert_this_step = 0.0
        if atype == ActionType.APPLY_FERTILIZER:
            overfert_this_step = (
                (action.n_delta or 0.0)
                + (action.p_delta or 0.0)
                + (action.k_delta or 0.0)
            )

        # ----------------------------------------------------------------
        # 3. Apply action
        # ----------------------------------------------------------------
        task_importance: str | None = None
        all_high_tasks_done = False
        all_tasks_done = False

        if atype == ActionType.SELECT_CROP:
            obs = self._action_select_crop(action, obs, state)

        elif atype == ActionType.APPLY_FERTILIZER:
            obs = self._action_apply_fertilizer(action, obs)
            # Delayed consequence: over-fertilization → disease spike next step
            if overfert_this_step > OVERFERT_STEP_LIMIT:
                state.overfert_pending = True

        elif atype == ActionType.IRRIGATE:
            obs = self._action_irrigate(action, obs)

        elif atype == ActionType.AMEND_PH:
            obs = self._action_amend_ph(action, obs)

        elif atype == ActionType.COMPLETE_TASK:
            obs, task_importance = self._action_complete_task(action, obs)
            state.total_tasks_completed += 1 if task_importance else 0

        elif atype == ActionType.ADVANCE_STAGE:
            obs, all_high_tasks_done, all_tasks_done = self._action_advance_stage(obs, state)

        elif atype == ActionType.APPLY_TREATMENT:
            obs = self._action_apply_treatment(obs)

        elif atype == ActionType.WAIT:
            pass

        # ----------------------------------------------------------------
        # 4. Track consecutive waits
        # ----------------------------------------------------------------
        if atype == ActionType.WAIT:
            state.consecutive_waits += 1
        else:
            state.consecutive_waits = 0

        # ----------------------------------------------------------------
        # 5. Environmental dynamics
        # ----------------------------------------------------------------
        obs = self._apply_env_dynamics(obs)

        # ----------------------------------------------------------------
        # 6. Disease event (over-fertilization spikes probability)
        # ----------------------------------------------------------------
        obs = self._update_disease(obs, state)

        # ----------------------------------------------------------------
        # 7. Track disease untreated steps
        # ----------------------------------------------------------------
        if obs.disease_active == 1 and atype != ActionType.APPLY_TREATMENT:
            state.disease_untreated_steps += 1
        elif obs.disease_active == 0:
            state.disease_untreated_steps = 0

        # ----------------------------------------------------------------
        # 8. Check nutrient collapse (irreversible flag)
        # ----------------------------------------------------------------
        if not state.nutrient_collapsed and obs.crop_index > 0:
            state.nutrient_collapsed = self._check_nutrient_collapse(obs)

        # ----------------------------------------------------------------
        # 9. Compute step reward
        # ----------------------------------------------------------------
        reward, breakdown = step_reward(
            action=action,
            obs_before=obs_before,
            obs_after=obs,
            crop_confidence=state.crop_confidence,
            task_importance=task_importance,
            all_high_tasks_done=all_high_tasks_done,
            all_tasks_done=all_tasks_done,
            disease_untreated_steps=state.disease_untreated_steps,
            stage_step_count=state.stage_step_count,
            consecutive_waits=state.consecutive_waits,
            nutrient_collapsed=state.nutrient_collapsed,
            overfert_this_step=overfert_this_step,
        )

        if self.debug and atype == ActionType.SELECT_CROP:
            print(f"[DEBUG] Reward assigned={reward:.4f}")

        # ----------------------------------------------------------------
        # 9b. Climate shock event — runs AFTER reward is initialized
        # ----------------------------------------------------------------
        event_type = None
        if self._rng.random() < 0.15:
            if self._rng.random() < 0.5:
                # --- Drought ---
                event_type = "drought"
                new_rain = max(0.0, obs.rainfall - 20.0)
                obs = obs.model_copy(update={"rainfall": round(new_rain, 2)})
                reward -= 0.2
                if obs.rainfall < 20.0:
                    state.yield_potential_multiplier *= 0.7
            else:
                # --- Disease outbreak ---
                event_type = "disease_outbreak"
                state.disease_risk = min(1.0, state.disease_risk + 0.4)
                reward -= 0.2
                if state.disease_risk > 0.8:
                    state.yield_potential_multiplier *= 0.6
                    obs = obs.model_copy(update={"disease_active": 1})

        # ----------------------------------------------------------------
        # 10. Stage step counter (reset on ADVANCE_STAGE)
        # ----------------------------------------------------------------
        if atype == ActionType.ADVANCE_STAGE:
            state.stage_step_count = 0
        else:
            state.stage_step_count += 1

        # Hard cap: force advance if agent is stuck in a stage too long
        if state.stage_step_count >= STAGE_BUDGET_HARD_CAP and obs.stage < 4:
            obs, _, _ = self._action_advance_stage(obs, state)
            reward += -1.0   # forced advance penalty
            state.stage_step_count = 0

        # ----------------------------------------------------------------
        # 11. Done / failure conditions
        # ----------------------------------------------------------------
        state.step_count += 1
        state.total_reward += reward
        state.obs = obs

        done = False
        truncated = False
        failure_reason: str | None = None

        # Normal harvest
        if obs.stage == 4 and atype == ActionType.ADVANCE_STAGE:
            done = True
            state.yield_at_harvest = self._compute_final_yield(obs, state)
            reward += terminal_reward(
                crop_index=obs.crop_index,
                yield_ton_ha=state.yield_at_harvest,
                total_tasks=self._total_tasks(),
                tasks_completed=state.total_tasks_completed,
                disease_untreated_steps=state.disease_untreated_steps,
                max_steps=MAX_STEPS,
                nutrient_collapsed=state.nutrient_collapsed,
                skipped_high_task_stages=state.skipped_high_task_stages,
            )

        # Failure: severe disease (lowered threshold)
        if state.disease_untreated_steps >= DISEASE_FAILURE_STEPS:
            done = True
            failure_reason = "severe_disease"
            reward += -4.0
            state.yield_at_harvest = self._compute_final_yield(obs, state) * 0.1

        # Failure: nutrient collapse reached terminal stage
        if state.nutrient_collapsed and obs.stage >= 3:
            done = True
            failure_reason = "nutrient_collapse"
            reward += -3.0
            state.yield_at_harvest = self._compute_final_yield(obs, state) * 0.15

        # Failure: no crop selected by stage 2
        if obs.stage >= 2 and obs.crop_index == 0:
            done = True
            failure_reason = "no_crop_selected"
            reward += -3.0

        # Max steps
        if state.step_count >= MAX_STEPS:
            truncated = True
            if state.yield_at_harvest == 0.0:
                state.yield_at_harvest = self._compute_final_yield(obs, state)

        state.done = done or truncated
        info = self._build_info(state, obs, reward, failure_reason, event_type)
        info["truncated"] = truncated
        info["reward_breakdown"] = breakdown

        if self.debug:
            logger.debug("[AFTER] reward=%.4f done=%s breakdown=%s",
                         reward, done or truncated, breakdown)

        return obs.model_copy(), round(reward, 4), done or truncated, truncated, info

    # ------------------------------------------------------------------
    # score()
    # ------------------------------------------------------------------
    def score(self) -> float:
        from agrosarthi_rl_env.reward import compute_score
        if self._state is None:
            return 0.0
        state = self._state
        return compute_score(
            crop_index=state.obs.crop_index,
            yield_ton_ha=state.yield_at_harvest,
            tasks_completed=state.total_tasks_completed,
            total_tasks=self._total_tasks(),
            disease_untreated_steps=state.disease_untreated_steps,
            max_steps=MAX_STEPS,
            total_reward=state.total_reward,
            steps_taken=state.step_count,
            nutrient_collapsed=state.nutrient_collapsed,
            skipped_high_task_stages=state.skipped_high_task_stages,
        )

    # ------------------------------------------------------------------
    # Action validation
    # ------------------------------------------------------------------

    def _validate_action(self, action: Action, obs: Observation) -> str | None:
        atype = action.action_type

        if atype == ActionType.SELECT_CROP:
            if obs.stage != 0:
                return f"SELECT_CROP only valid in stage 0, current={obs.stage}"
            if action.crop_index is None or action.crop_index == 0:
                return "SELECT_CROP requires crop_index in [1, 21]"

        if atype == ActionType.COMPLETE_TASK:
            if action.task_index is None:
                return "COMPLETE_TASK requires task_index"
            if action.task_index >= len(STAGE_TASKS[obs.stage]):
                return (f"task_index={action.task_index} out of range for "
                        f"stage={obs.stage} (max={len(STAGE_TASKS[obs.stage])-1})")

        if atype == ActionType.ADVANCE_STAGE and obs.stage == 4:
            return "Already at harvest stage"

        return None

    def _invalid_action_penalty(self, atype: ActionType) -> float:
        return {
            ActionType.SELECT_CROP:   -0.5,
            ActionType.COMPLETE_TASK: -0.3,
            ActionType.ADVANCE_STAGE: -0.4,
        }.get(atype, -0.2)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _action_select_crop(self, action: Action, obs: Observation,
                             state: EpisodeState) -> Observation:
        obs = obs.model_copy(update={"crop_index": action.crop_index})
        state.crop_confidence = score_crop(
            action.crop_index,
            obs.N, obs.P, obs.K, obs.ph, obs.temperature, obs.rainfall
        )
        if self.debug:
            selected_crop = CROP_LIST[action.crop_index]
            print(f"[DEBUG] Crop={selected_crop}")
            print(f"[DEBUG] Suitability={state.crop_confidence:.4f}")
            print(f"[DEBUG] Soil: N={obs.N}, P={obs.P}, K={obs.K}, pH={obs.ph}")
        return obs

    def _action_apply_fertilizer(self, action: Action, obs: Observation) -> Observation:
        n = min(obs.N + (action.n_delta or 0.0), 200.0)
        p = min(obs.P + (action.p_delta or 0.0), 200.0)
        k = min(obs.K + (action.k_delta or 0.0), 200.0)
        return obs.model_copy(update={"N": round(n, 2), "P": round(p, 2), "K": round(k, 2)})

    def _action_irrigate(self, action: Action, obs: Observation) -> Observation:
        new_rain = min(obs.rainfall + (action.irrigation_mm or 0.0), 500.0)
        return obs.model_copy(update={"rainfall": round(new_rain, 2)})

    def _action_amend_ph(self, action: Action, obs: Observation) -> Observation:
        new_ph = max(0.0, min(14.0, round(obs.ph + (action.ph_delta or 0.0), 2)))
        return obs.model_copy(update={"ph": new_ph})

    def _action_complete_task(self, action: Action,
                               obs: Observation) -> tuple[Observation, str | None]:
        stage_tasks = STAGE_TASKS[obs.stage]
        task_idx = action.task_index
        if task_idx is None or task_idx >= len(stage_tasks):
            return obs, None
        importance = stage_tasks[task_idx]["importance"]
        new_tasks_done = min(obs.tasks_done + 1, len(stage_tasks))
        return obs.model_copy(update={"tasks_done": new_tasks_done}), importance

    def _action_advance_stage(self, obs: Observation,
                               state: EpisodeState) -> tuple[Observation, bool, bool]:
        stage_tasks = STAGE_TASKS[obs.stage]
        high_count = sum(1 for t in stage_tasks if t["importance"] == "high")
        all_high_done = obs.tasks_done >= high_count
        all_done = obs.tasks_done >= len(stage_tasks)

        if not all_high_done:
            state.skipped_high_task_stages += 1

        if obs.stage < 4:
            obs = obs.model_copy(update={"stage": obs.stage + 1, "tasks_done": 0})

        return obs, all_high_done, all_done

    def _action_apply_treatment(self, obs: Observation) -> Observation:
        return obs.model_copy(update={"disease_active": 0})

    # ------------------------------------------------------------------
    # Environmental dynamics
    # ------------------------------------------------------------------

    def _apply_env_dynamics(self, obs: Observation) -> Observation:
        new_temp, new_rain = self.weather_sim.step(obs.temperature, obs.rainfall, obs.stage)
        new_N = max(0.0, obs.N - NUTRIENT_DECAY["N"])
        new_P = max(0.0, obs.P - NUTRIENT_DECAY["P"])
        new_K = max(0.0, obs.K - NUTRIENT_DECAY["K"])
        return obs.model_copy(update={
            "temperature": new_temp, "rainfall": new_rain,
            "N": round(new_N, 2), "P": round(new_P, 2), "K": round(new_K, 2),
        })

    def _update_disease(self, obs: Observation, state: EpisodeState) -> Observation:
        if obs.disease_active == 1:
            return obs

        base_prob = DISEASE_BASE_PROB.get(obs.stage, 0.05)

        # Excess irrigation doubles probability
        if obs.rainfall > 300.0:
            base_prob *= 2.0

        # Over-fertilization from last step: spike probability to 0.6
        if state.overfert_pending:
            base_prob = max(base_prob, 0.6)
            state.overfert_pending = False

        if self._rng.random() < base_prob:
            return obs.model_copy(update={"disease_active": 1})
        return obs

    def _check_nutrient_collapse(self, obs: Observation) -> bool:
        """Returns True if any nutrient is in the collapse zone."""
        crop_name = CROP_LIST[obs.crop_index]
        optima = CROP_OPTIMA.get(crop_name, {})
        if not optima:
            return False
        return (
            obs.N < optima["N"][0] * NUTRIENT_COLLAPSE_FRAC or
            obs.P < optima["P"][0] * NUTRIENT_COLLAPSE_FRAC or
            obs.K < optima["K"][0] * NUTRIENT_COLLAPSE_FRAC
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _total_tasks(self) -> int:
        return sum(len(s) for s in STAGE_TASKS)

    def _build_info(self, state: EpisodeState, obs: Observation,
                    reward: float, failure_reason: str | None,
                    event_type: str | None = None) -> dict:
        return {
            "step":                      state.step_count,
            "stage":                     obs.stage,
            "crop":                      CROP_LIST[obs.crop_index],
            "crop_confidence":           state.crop_confidence,
            "crop_suitability":          state.crop_confidence,
            "crop_selected":             CROP_LIST[obs.crop_index],
            "selected_crop":             CROP_LIST[obs.crop_index],
            "reward":                    round(reward, 4),
            "disease_untreated_steps":   state.disease_untreated_steps,
            "yield_at_harvest":          state.yield_at_harvest,
            "total_reward":              state.total_reward,
            "nutrient_collapsed":        state.nutrient_collapsed,
            "skipped_high_task_stages":  state.skipped_high_task_stages,
            "failure_reason":            failure_reason,
            "event":                     event_type,
            "disease_risk":              round(state.disease_risk, 3),
            "yield_potential_multiplier": round(state.yield_potential_multiplier, 3),
        }

    def _compute_final_yield(self, obs: Observation, state: EpisodeState) -> float:
        total = self._total_tasks()
        task_ratio = state.total_tasks_completed / max(total, 1)
        disease_penalty = min(state.disease_untreated_steps * 0.05, 0.6)
        collapse_penalty = 0.7 if state.nutrient_collapsed else 0.0

        raw = estimate_yield(
            crop_index=obs.crop_index,
            area_hectare=1.0,
            suitability_score=state.crop_confidence,
            tasks_done_ratio=task_ratio,
            disease_penalty=disease_penalty,
        )
        # Apply climate shock degradation (drought / disease outbreak events)
        return round(raw * (1.0 - collapse_penalty) * state.yield_potential_multiplier, 3)
