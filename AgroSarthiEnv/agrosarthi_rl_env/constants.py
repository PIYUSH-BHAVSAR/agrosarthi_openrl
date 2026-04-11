# =============================================================================
# agrosarthi_rl_env/constants.py
# All static domain knowledge — no LLM, no API calls
# =============================================================================

# Ordered list of supported crops (index 0 = "none / not selected")
CROP_LIST = [
    "none",       # 0 — placeholder before selection
    "rice",       # 1
    "wheat",      # 2
    "maize",      # 3
    "chickpea",   # 4
    "kidneybeans",# 5
    "pigeonpeas", # 6
    "mothbeans",  # 7
    "mungbean",   # 8
    "blackgram",  # 9
    "lentil",     # 10
    "pomegranate",# 11
    "banana",     # 12
    "mango",      # 13
    "grapes",     # 14
    "watermelon", # 15
    "muskmelon",  # 16
    "apple",      # 17
    "orange",     # 18
    "papaya",     # 19
    "coconut",    # 20
    "cotton",     # 21
]

# Optimal soil/climate ranges per crop
# Format: { crop_name: { N:(min,max), P:(min,max), K:(min,max), ph:(min,max),
#                        temp:(min,max), rainfall:(min,max) } }
CROP_OPTIMA: dict = {
    "rice":        {"N":(60,120), "P":(30,60),  "K":(30,60),  "ph":(5.5,7.0), "temp":(20,35), "rainfall":(150,300)},
    "wheat":       {"N":(60,120), "P":(30,60),  "K":(30,60),  "ph":(6.0,7.5), "temp":(10,25), "rainfall":(50,150)},
    "maize":       {"N":(60,100), "P":(30,60),  "K":(20,50),  "ph":(5.5,7.5), "temp":(18,35), "rainfall":(50,200)},
    "chickpea":    {"N":(20,60),  "P":(40,80),  "K":(20,40),  "ph":(6.0,8.0), "temp":(15,30), "rainfall":(30,100)},
    "kidneybeans": {"N":(20,40),  "P":(60,100), "K":(20,40),  "ph":(6.0,7.5), "temp":(15,30), "rainfall":(50,150)},
    "pigeonpeas":  {"N":(20,40),  "P":(40,80),  "K":(20,40),  "ph":(5.5,7.0), "temp":(20,35), "rainfall":(60,150)},
    "mothbeans":   {"N":(20,40),  "P":(40,60),  "K":(20,40),  "ph":(6.0,8.0), "temp":(25,40), "rainfall":(20,80)},
    "mungbean":    {"N":(20,40),  "P":(40,60),  "K":(20,40),  "ph":(6.0,7.5), "temp":(25,35), "rainfall":(50,100)},
    "blackgram":   {"N":(20,40),  "P":(40,60),  "K":(20,40),  "ph":(6.0,7.5), "temp":(25,35), "rainfall":(50,100)},
    "lentil":      {"N":(20,40),  "P":(40,80),  "K":(20,40),  "ph":(6.0,8.0), "temp":(15,25), "rainfall":(30,100)},
    "pomegranate": {"N":(20,40),  "P":(10,30),  "K":(20,40),  "ph":(5.5,7.5), "temp":(25,40), "rainfall":(50,150)},
    "banana":      {"N":(80,120), "P":(20,40),  "K":(40,80),  "ph":(5.5,7.0), "temp":(25,35), "rainfall":(100,200)},
    "mango":       {"N":(20,40),  "P":(10,30),  "K":(20,40),  "ph":(5.5,7.5), "temp":(24,35), "rainfall":(50,150)},
    "grapes":      {"N":(20,40),  "P":(10,30),  "K":(20,40),  "ph":(5.5,7.0), "temp":(15,35), "rainfall":(50,100)},
    "watermelon":  {"N":(80,120), "P":(40,60),  "K":(40,80),  "ph":(6.0,7.0), "temp":(25,40), "rainfall":(40,100)},
    "muskmelon":   {"N":(80,120), "P":(40,60),  "K":(40,80),  "ph":(6.0,7.5), "temp":(25,40), "rainfall":(40,100)},
    "apple":       {"N":(20,40),  "P":(10,30),  "K":(20,40),  "ph":(5.5,6.5), "temp":(5,20),  "rainfall":(100,200)},
    "orange":      {"N":(20,40),  "P":(10,30),  "K":(20,40),  "ph":(5.5,7.0), "temp":(15,35), "rainfall":(75,150)},
    "papaya":      {"N":(40,80),  "P":(20,40),  "K":(40,80),  "ph":(6.0,7.0), "temp":(25,35), "rainfall":(100,200)},
    "coconut":     {"N":(20,40),  "P":(10,20),  "K":(40,80),  "ph":(5.5,8.0), "temp":(25,35), "rainfall":(100,300)},
    "cotton":      {"N":(60,120), "P":(30,60),  "K":(30,60),  "ph":(6.0,8.0), "temp":(20,35), "rainfall":(50,150)},
}

# Tasks per cultivation stage (5 stages × up to 5 tasks)
# importance: "high" | "medium" | "low"
STAGE_TASKS: list = [
    # Stage 0: Land Preparation
    [
        {"title": "Clear field",       "importance": "high"},
        {"title": "Soil testing",      "importance": "high"},
        {"title": "Tillage",           "importance": "medium"},
        {"title": "Drainage setup",    "importance": "medium"},
        {"title": "Bed formation",     "importance": "low"},
    ],
    # Stage 1: Sowing
    [
        {"title": "Seed selection",    "importance": "high"},
        {"title": "Seed treatment",    "importance": "high"},
        {"title": "Planting",          "importance": "high"},
        {"title": "Initial watering",  "importance": "high"},
        {"title": "Mulching",          "importance": "low"},
    ],
    # Stage 2: Vegetative Growth
    [
        {"title": "Fertilizer application", "importance": "high"},
        {"title": "Weed control",           "importance": "medium"},
        {"title": "Irrigation management",  "importance": "high"},
        {"title": "Thinning",               "importance": "medium"},
        {"title": "Soil aeration",          "importance": "low"},
    ],
    # Stage 3: Flowering / Fruiting
    [
        {"title": "Monitor plant health",   "importance": "high"},
        {"title": "Pest management",        "importance": "high"},
        {"title": "Micronutrient spray",    "importance": "medium"},
        {"title": "Support structures",     "importance": "medium"},
        {"title": "Pollination support",    "importance": "low"},
    ],
    # Stage 4: Harvest
    [
        {"title": "Maturity assessment",    "importance": "high"},
        {"title": "Harvesting",             "importance": "high"},
        {"title": "Post-harvest handling",  "importance": "medium"},
        {"title": "Yield recording",        "importance": "medium"},
        {"title": "Field cleanup",          "importance": "low"},
    ],
]

# Baseline yield (ton/ha) per crop — used for terminal reward comparison
BASELINE_YIELD: dict = {
    "rice": 3.5, "wheat": 3.0, "maize": 4.0, "chickpea": 1.2,
    "kidneybeans": 1.5, "pigeonpeas": 1.0, "mothbeans": 0.8,
    "mungbean": 1.0, "blackgram": 0.9, "lentil": 1.1,
    "pomegranate": 8.0, "banana": 20.0, "mango": 6.0, "grapes": 10.0,
    "watermelon": 25.0, "muskmelon": 15.0, "apple": 10.0, "orange": 12.0,
    "papaya": 30.0, "coconut": 5.0, "cotton": 2.0,
}

# Disease probability per stage (base rate, modified by irrigation excess)
DISEASE_BASE_PROB: dict = {
    0: 0.02,  # Land Prep
    1: 0.05,  # Sowing
    2: 0.10,  # Vegetative
    3: 0.15,  # Flowering
    4: 0.05,  # Harvest
}

# Weather simulation: per-stage temperature and rainfall deltas (mean, std)
WEATHER_SIM: dict = {
    0: {"temp_delta": (0.0, 1.0),  "rain_delta": (5.0,  10.0)},
    1: {"temp_delta": (0.5, 1.5),  "rain_delta": (10.0, 15.0)},
    2: {"temp_delta": (1.0, 2.0),  "rain_delta": (20.0, 20.0)},
    3: {"temp_delta": (0.5, 1.5),  "rain_delta": (15.0, 15.0)},
    4: {"temp_delta": (-0.5, 1.0), "rain_delta": (5.0,  10.0)},
}

# Nutrient decay per step (natural leaching / uptake)
NUTRIENT_DECAY = {"N": 2.0, "P": 0.5, "K": 1.0}

# Max steps per episode
MAX_STEPS = 60
