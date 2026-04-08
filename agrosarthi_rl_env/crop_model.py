# =============================================================================
# agrosarthi_rl_env/crop_model.py
# Rule-based crop suitability scorer — replaces Keras .h5 inference
# Used when the original model files are not available, or as a fast sim layer
# =============================================================================
import math
from agrosarthi_rl_env.constants import CROP_LIST, CROP_OPTIMA


def _range_score(value: float, low: float, high: float) -> float:
    """
    Returns 1.0 if value is within [low, high].
    Decays linearly to 0.0 at 2x the distance outside the range.
    """
    if low <= value <= high:
        return 1.0
    margin = (high - low) * 0.5 + 1e-6
    if value < low:
        return max(0.0, 1.0 - (low - value) / margin)
    return max(0.0, 1.0 - (value - high) / margin)


def score_crop(
    crop_index: int,
    N: float, P: float, K: float,
    ph: float, temperature: float, rainfall: float
) -> float:
    """
    Returns a suitability score in [0.0, 1.0] for the given crop
    under the provided soil/climate conditions.

    Weights: N(0.15) P(0.15) K(0.15) ph(0.20) temp(0.20) rain(0.15)
    """
    if crop_index == 0:
        return 0.0  # "none" crop

    crop_name = CROP_LIST[crop_index]
    optima = CROP_OPTIMA.get(crop_name)
    if optima is None:
        return 0.0  # unknown crop → no suitability → always penalised on selection

    weights = {"N": 0.15, "P": 0.15, "K": 0.15, "ph": 0.20, "temp": 0.20, "rain": 0.15}

    scores = {
        "N":    _range_score(N,           *optima["N"]),
        "P":    _range_score(P,           *optima["P"]),
        "K":    _range_score(K,           *optima["K"]),
        "ph":   _range_score(ph,          *optima["ph"]),
        "temp": _range_score(temperature, *optima["temp"]),
        "rain": _range_score(rainfall,    *optima["rainfall"]),
    }

    return round(sum(weights[k] * scores[k] for k in weights), 4)


def top_crops(
    N: float, P: float, K: float,
    ph: float, temperature: float, rainfall: float,
    top_n: int = 3
) -> list[tuple[int, float]]:
    """
    Returns top_n (crop_index, score) tuples sorted by score descending.
    Excludes crop_index=0 ("none").
    """
    scored = [
        (i, score_crop(i, N, P, K, ph, temperature, rainfall))
        for i in range(1, len(CROP_LIST))
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def estimate_yield(
    crop_index: int,
    area_hectare: float,
    suitability_score: float,
    tasks_done_ratio: float,
    disease_penalty: float
) -> float:
    """
    Rule-based yield estimator — replaces sklearn .pkl model for mid-episode use.
    Formula:
        yield = baseline * area * suitability * task_ratio * (1 - disease_penalty)

    disease_penalty: 0.0 (healthy) to 0.4 (severe untreated disease)
    tasks_done_ratio: 0.0 to 1.0
    """
    from agrosarthi_rl_env.constants import BASELINE_YIELD, CROP_LIST

    if crop_index == 0:
        return 0.0

    crop_name = CROP_LIST[crop_index]
    baseline = BASELINE_YIELD.get(crop_name, 2.0)

    raw = baseline * area_hectare * suitability_score * tasks_done_ratio
    penalized = raw * (1.0 - min(disease_penalty, 0.4))
    return round(max(0.0, penalized), 3)
