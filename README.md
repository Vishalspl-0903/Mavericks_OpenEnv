---
title: medical-triage-env
sdk: docker
app_port: 8000
colorFrom: blue
colorTo: green
---

# Medical Triage Environment

An OpenEnv benchmark for emergency department triage using the Emergency Severity Index (ESI 1-5).

## Overview
The agent receives structured patient presentations and must either classify urgency or request a clarifying question when additional history is needed. The benchmark emphasizes triage prioritization, clinical reasoning, and safe escalation.

## Tasks
| Task ID | Difficulty | Scenario | Correct ESI |
|---|---|---|---|
| classic-mi | Easy | STEMI with hypotension, diaphoresis, and chest pain | 1 |
| meningitis-suspect | Medium | Suspected meningococcal meningitis with petechiae | 1 |
| masked-sepsis | Hard | Elderly urosepsis masked by beta-blockade and CKD | 2 |

## Reward Summary
- ESI accuracy: 50%
- Reasoning quality: 30%
- Action appropriateness: 20%
- Undertriage penalty: applied for dangerous low-acuity assignments
- Urgency bonus: correct ESI on early steps
- Step penalty: small penalty per additional step

## Setup
```bash
pip install -r requirements.txt
docker build -t medical-triage-env .
docker run -p 8000:8000 medical-triage-env
openenv validate
python inference.py
```

## Environment Variables
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` or `API_KEY`

Use `.env` only for local development. Do not commit secrets.

## Baseline Scores
| Task                | Difficulty | Score |
|---------------------|------------|-------|
| classic-mi          | Easy       | 0.57 |
| meningitis-suspect  | Medium     | 0.57 |
| masked-sepsis       | Hard       | 0.76 |
