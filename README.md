---
title: workflowenv
sdk: docker
---

# WorkflowEnv

OpenEnv-compliant RL environment for workflow automation tasks.

An AI agent receives a natural-language workflow description and must produce
a valid structured automation plan (trigger → steps → edges → error handling).

## Tasks
- **Easy**: Google Sheets row added → Gmail notification (2-step linear)
- **Medium**: GitHub issue → conditional Jira + Slack or label (branching)
- **Hard**: Daily cron → Postgres → Python → BigQuery + Slack + PagerDuty error branch

## Run locally
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Inference
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
python inference.py
```

## API
- `POST /reset` — start episode `{"task_id": "easy"}`
- `POST /step`  — submit workflow plan
- `GET  /state` — current episode state
- `GET  /health` — health check

## Author
Darshan Linge Gowda
