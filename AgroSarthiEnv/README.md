---
title: AgroSarthi Env
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
secrets:
  - API_BASE_URL
  - API_KEY
  - HF_TOKEN
  - MODEL_NAME
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
