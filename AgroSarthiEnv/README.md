---
title: AgroSarthi Env
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
---

# AgroSarthiEnv

OpenEnv-compatible agricultural RL environment simulating a full crop growing season.

The agent manages soil nutrients, irrigation, disease response, and task execution across 5 cultivation stages to maximise yield.

## Tasks

- **easy** — Crop selection based on soil and weather conditions
- **medium** — Farm operations: fertilizer, irrigation, disease prevention
- **hard** — Full season optimization: maximize yield under constraints

## API Endpoints

- `POST /reset` — reset environment
- `POST /step` — execute action
- `GET /grade/easy` — easy task score
- `GET /grade/medium` — medium task score
- `GET /grade/hard` — hard task score
- `GET /health` — health check
- `GET /docs` — API documentation

## Quick Start

```python
from AgroSarthiEnv import AgroSarthiEnvEnv, AgroSarthiEnvAction

env = AgroSarthiEnvEnv(base_url="https://pylord-agrosarthi-env.hf.space")
result = env.reset()
result = env.step(AgroSarthiEnvAction(action_type="SELECT_CROP", crop_index=1))
```
