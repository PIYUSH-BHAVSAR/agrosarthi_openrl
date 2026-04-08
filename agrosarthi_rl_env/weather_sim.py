# =============================================================================
# agrosarthi_rl_env/weather_sim.py
# Deterministic (seeded) weather simulator — replaces live WeatherAPI calls
# =============================================================================
import random
from agrosarthi_rl_env.constants import WEATHER_SIM


class WeatherSimulator:
    """
    Simulates temperature and rainfall changes per step.
    Seeded for reproducibility. Replaces WeatherService.get_forecast().
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def step(self, current_temp: float, current_rain: float, stage: int) -> tuple[float, float]:
        """
        Returns (new_temperature, new_rainfall) after one step.
        Uses stage-specific mean/std from WEATHER_SIM.
        """
        cfg = WEATHER_SIM[stage]

        temp_mean, temp_std = cfg["temp_delta"]
        rain_mean, rain_std = cfg["rain_delta"]

        new_temp = current_temp + self.rng.gauss(temp_mean, temp_std)
        new_rain = current_rain + self.rng.gauss(rain_mean, rain_std)

        # Clamp to physical bounds
        new_temp = max(5.0, min(50.0, new_temp))
        new_rain = max(0.0, min(500.0, new_rain))

        return round(new_temp, 2), round(new_rain, 2)

    def reset(self, seed: int):
        """Re-seed for a new episode."""
        self.rng = random.Random(seed)
