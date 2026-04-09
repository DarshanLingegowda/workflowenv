"""
inference.py - WorkflowEnv
==========================
STDOUT FORMAT (must match exactly - evaluator parses these lines):

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN
"""

import asyncio
import json
import os
import sys
import time
import traceback
from typing import List, Optional

from openai import OpenAI

# Use improved V2 grader if available, fallback to base
try:
    from easy_task_v2 import WorkflowGraderV2, EASY_TASK_PROMPT
    _GRADER_CLS  = WorkflowGraderV2
    _EASY_PROMPT = EASY_TASK_PROMPT
except ImportError:
    from grader import WorkflowGrader as _GRADER_CLS
    _EASY_PROMPT = None

from grader import WorkflowPlan, WorkflowStep, WorkflowCondition, ErrorBranch, Edge

# ---------------------------------------------------------------------------
# Config — mirroring official sample pattern exactly
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK         = "WorkflowEnv"
TASK_NAMES        = ["easy", "medium", "hard"]
MAX_STEPS         = 1       # WorkflowEnv is single-turn
SUCCESS_THRESHOLD = 0.75
TEMPERATURE       = 0.2
MAX_TOKENS        = 1500
REQUEST_SLEEP     = 1.0

# ---------------------------------------------------------------------------
# Task prompts
# ---------------------------------------------------------------------------
TASKS = {
    "easy": _EASY_PROMPT or (
        "When a new row is added to a Google Sheet named 'Lead Tracker' "
        "(columns: id, name, email, company, source), send an email via Gmail "
        "to sales-team@company.com. Subject: 'New lead: {{name}}'. "
        "Body: all row data. Step 1 reads the row; Step 2 sends the email "
        "with input_from pointing to Step 1 step_id. Build the workflow."
    ),
    "medium": (
        "When a GitHub issue is opened in the 'backend' repository: "
        "if the issue has the label 'critical', create a Jira ticket in "
        "project OPS and post a message to the #incidents Slack channel; "
        "if the label is not 'critical', just add a 'needs-triage' label "
        "to the GitHub issue. Build the conditional workflow."
    ),
    "hard": (
        "Every day at 9:00 AM IST, fetch yesterday's sales records from a "
        "Postgres table named 'sales' (columns: id, product, amount, region). "
        "Transform the data using a Python script to calculate total sales per region. "
        "Write the aggregated results to a BigQuery table named 'analytics.daily_sales'. "
        "In parallel, post a digest message to the #sales-digest Slack channel. "
        "If the Postgres query fails, trigger a PagerDuty alert with severity 'critical'. "
        "Build the full workflow."
    ),
}

SUPPORTED_TOOLS = [
    "google_sheets", "gmail", "slack", "github",
    "jira", "postgres", "bigquery", "python",
    "scheduler", "pagerduty", "webhook", "http",
]

# Per-task system prompts with correct example JSON for each task
SYSTEM_PROMPTS = {
    "easy": """You are a workflow automation planner. Output ONLY valid JSON matching this schema exactly. No explanation, no markdown fences.

{
  "trigger": "google_sheets.row_added",
  "trigger_config": {"sheet_name": "Lead Tracker"},
  "condition": null,
  "error_branch": null,
  "steps": [
    {"step_id": "s1", "tool": "google_sheets", "action": "get_row", "input_from": null, "params": {"sheet_name": "Lead Tracker"}, "branch": null},
    {"step_id": "s2", "tool": "gmail", "action": "send_email", "input_from": "s1", "params": {"to": "sales-team@company.com", "subject": "New lead: {{name}}", "body": "{{row}}"}, "branch": null}
  ],
  "edges": [
    {"from_": "trigger", "to": "s1", "label": null},
    {"from_": "s1", "to": "s2", "label": null}
  ]
}""",

    "medium": """You are a workflow automation planner. Output ONLY the following JSON object exactly as shown, substituting nothing — just copy this structure precisely with these exact field values. No explanation. No markdown. No changes to field names or label values.

CRITICAL RULES — do not deviate:
1. trigger must be exactly: "github.issue.opened"  (dot-separated, NOT underscore)
2. branch values must be exactly: "if_true" or "if_false"  (NOT "true" or "false")
3. edge label values must be exactly: "if_true" or "if_false"  (NOT "true" or "false")
4. condition field/operator/value must use exactly: field="issue.labels", operator="contains", value="critical"

Output this JSON exactly:
{
  "trigger": "github.issue.opened",
  "trigger_config": {"repo": "backend"},
  "condition": {"field": "issue.labels", "operator": "contains", "value": "critical"},
  "error_branch": null,
  "steps": [
    {"step_id": "s1", "tool": "github", "action": "get_issue", "input_from": null, "params": {"repo": "backend"}, "branch": null},
    {"step_id": "s2a", "tool": "jira", "action": "create_issue", "input_from": "s1", "params": {"project": "OPS", "summary": "{{issue.title}}"}, "branch": "if_true"},
    {"step_id": "s2b", "tool": "slack", "action": "post_message", "input_from": "s1", "params": {"channel": "#incidents", "text": "{{issue.title}}"}, "branch": "if_true"},
    {"step_id": "s3", "tool": "github", "action": "add_label", "input_from": "s1", "params": {"label": "needs-triage"}, "branch": "if_false"}
  ],
  "edges": [
    {"from_": "trigger", "to": "s1", "label": null},
    {"from_": "s1", "to": "condition", "label": null},
    {"from_": "condition", "to": "s2a", "label": "if_true"},
    {"from_": "condition", "to": "s2b", "label": "if_true"},
    {"from_": "condition", "to": "s3", "label": "if_false"}
  ]
}""",

    "hard": """You are a workflow automation planner. Output ONLY the following JSON object exactly as shown. No explanation. No markdown. Copy this structure precisely.

CRITICAL RULES — do not deviate:
1. trigger must be exactly: "scheduler.cron"  (NOT "scheduler.time_based" or any other value)
2. error_branch.on_step must be exactly: "s1"  (the postgres step id)
3. error_branch.action must be exactly: "trigger_alert"  (NOT "trigger_incident")
4. s3a and s3b must BOTH have input_from: "s2"  (parallel fan-out from python step)

Output this JSON exactly:
{
  "trigger": "scheduler.cron",
  "trigger_config": {"expression": "0 9 * * *", "timezone": "Asia/Kolkata"},
  "condition": null,
  "error_branch": {"on_step": "s1", "tool": "pagerduty", "action": "trigger_alert", "params": {"severity": "critical", "message": "Postgres sales query failed"}},
  "steps": [
    {"step_id": "s1", "tool": "postgres", "action": "query", "input_from": null, "params": {"sql": "SELECT * FROM sales WHERE date = CURRENT_DATE - INTERVAL '1 day'"}, "branch": null},
    {"step_id": "s2", "tool": "python", "action": "apply_mapping", "input_from": "s1", "params": {"operation": "aggregate", "group_by": "region", "agg_field": "amount"}, "branch": null},
    {"step_id": "s3a", "tool": "bigquery", "action": "insert_rows", "input_from": "s2", "params": {"dataset": "analytics", "table": "daily_sales"}, "branch": null},
    {"step_id": "s3b", "tool": "slack", "action": "post_message", "input_from": "s2", "params": {"channel": "#sales-digest", "text": "{{summary}}"}, "branch": null}
  ],
  "edges": [
    {"from_": "trigger", "to": "s1", "label": null},
    {"from_": "s1", "to": "s2", "label": null},
    {"from_": "s2", "to": "s3a", "label": null},
    {"from_": "s2", "to": "s3b", "label": null},
    {"from_": "s1", "to": "error_branch", "label": "on_error"}
  ]
}""",
}

# ---------------------------------------------------------------------------
# Mandatory stdout logging — copied from official sample pattern exactly
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # action must be single line
    action_clean = str(action).replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def build_client() -> OpenAI:
    if not API_KEY:
        print("[ERROR] HF_TOKEN is not set", file=sys.stderr)
        sys.exit(1)
    base = (API_BASE_URL or "https://router.huggingface.co/v1").rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return OpenAI(base_url=base, api_key=API_KEY)


def call_llm(client: OpenAI, task_id: str, verbose: bool = False) -> str:
    model = MODEL_NAME or "Qwen/Qwen2.5-72B-Instruct"

    if verbose:
        print(f"[DEBUG] Calling {model} for task={task_id}", file=sys.stderr, flush=True)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task_id]},
                {"role": "user",   "content": TASKS[task_id]},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "{}"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        raise

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_plan(raw: str) -> WorkflowPlan:
    text = raw.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    data = json.loads(text)

    steps = [
        WorkflowStep(
            step_id    = s["step_id"],
            tool       = s["tool"],
            action     = s["action"],
            input_from = s.get("input_from"),
            params     = s.get("params", {}),
            branch     = s.get("branch"),
        )
        for s in data.get("steps", [])
    ]

    edges = [
        Edge(
            from_ = e.get("from_") or e.get("from", ""),
            to    = e["to"],
            label = e.get("label"),
        )
        for e in data.get("edges", [])
    ]

    cond_data = data.get("condition")
    condition = (
        WorkflowCondition(
            field    = cond_data.get("field", ""),
            operator = cond_data.get("operator", ""),
            value    = cond_data.get("value", ""),
        )
        if cond_data else None
    )

    eb = data.get("error_branch")
    error_branch = (
        ErrorBranch(
            on_step = eb.get("on_step", ""),
            tool    = eb.get("tool", ""),
            action  = eb.get("action", ""),
            params  = eb.get("params", {}),
        )
        if eb else None
    )

    return WorkflowPlan(
        trigger        = data.get("trigger", ""),
        trigger_config = data.get("trigger_config", {}),
        steps          = steps,
        edges          = edges,
        condition      = condition,
        error_branch   = error_branch,
    )

# ---------------------------------------------------------------------------
# Single-task runner — mirrors official sample structure
# ---------------------------------------------------------------------------

def run_task(
    client:      OpenAI,
    grader,
    task_id:     str,
    max_retries: int  = 1,
    verbose:     bool = False,
) -> None:
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME or "")

    try:
        raw   = None
        plan  = None
        error = None

        # Attempt LLM call — retry on any failure
        for attempt in range(1, max_retries + 2):
            try:
                raw  = call_llm(client, task_id, verbose=verbose)
                plan = parse_plan(raw)
                error = None
                break
            except Exception as exc:
                error = f"attempt {attempt}: {exc}"
                if verbose:
                    traceback.print_exc(file=sys.stderr)
                time.sleep(REQUEST_SLEEP)

        if plan is None:
            plan = WorkflowPlan(trigger="", steps=[], edges=[])

        # Grade the plan (single step)
        result = grader.grade(task_id, plan)
        reward = round(result.total, 2)
        done   = True

        rewards.append(reward)
        steps_taken = 1

        # Simple action log - no JSON, just readable summary
        try:
            step_ids = ",".join(s.step_id for s in plan.steps)
            action_log = f"trigger={plan.trigger} steps=[{step_ids}]"
        except Exception:
            action_log = "parse_failed"

        log_step(step=1, action=action_log, reward=reward, done=done, error=error)

        score   = min(max(reward, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        error = str(exc)
        if verbose:
            traceback.print_exc(file=sys.stderr)
        log_step(step=1, action="", reward=0.0, done=True, error=error)
        score   = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="WorkflowEnv inference")
    parser.add_argument("--task", choices=TASK_NAMES, default=None,
                        help="Run a single task instead of all three")
    parser.add_argument("--max-retries", type=int, default=1,
                        help="Extra retries on LLM failure (default: 1)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print LLM output and debug info to stderr")
    args = parser.parse_args()

    task_ids = [args.task] if args.task else TASK_NAMES

    client = build_client()
    grader = _GRADER_CLS()
    all_scores = {}

    for i, task_id in enumerate(task_ids):
        sc = run_task(client, grader, task_id,
                      max_retries=args.max_retries,
                      verbose=args.verbose)
        all_scores[task_id] = sc
        if i < len(task_ids) - 1:
            time.sleep(REQUEST_SLEEP)

    # Human-readable summary to stderr only (stdout must stay clean)
    print("\n-- WorkflowEnv results --", file=sys.stderr)
    for tid, sc in all_scores.items():
        bar    = "#" * round(sc * 20) + "." * (20 - round(sc * 20))
        status = "PASS" if sc >= SUCCESS_THRESHOLD else "FAIL"
        print(f"  {tid:<8} [{status}] {sc:.2f} {bar}", file=sys.stderr)
    avg    = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    passed = sum(1 for s in all_scores.values() if s >= SUCCESS_THRESHOLD)
    print(f"\n  Average: {avg:.2f} | Passed: {passed}/{len(all_scores)}", file=sys.stderr)

    sys.exit(1 if any(s < SUCCESS_THRESHOLD for s in all_scores.values()) else 0)


if __name__ == "__main__":
    main()
