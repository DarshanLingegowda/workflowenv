"""
==========================

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]uccess=<true|false> steps=<n> rewards=<r1,r2,...>

Rules (from sample inference script):
  - One [START] line at episode begin.
  - One [STEP] line per step, immediately after env.step() returns.
  - One [END] line always emitted (even on exception).
  - reward and rewards formatted to 2 decimal places.
  - done and success are lowercase: true or false.
  - error is raw error string, or null if none.
  - All fields on a single line, no newlines within a line.

Required env vars:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Usage:
    python inference.py
    python inference.py --task easy
    python inference.py --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Optional

from openai import OpenAI

# Use improved V2 grader if available, fallback to base
try:
    from easy_task_v2 import WorkflowGraderV2, EASY_TASK_PROMPT
    _GRADER_CLS  = WorkflowGraderV2
    _EASY_PROMPT = EASY_TASK_PROMPT
except ImportError:
    from grader import WorkflowGrader as _GRADER_CLS  # type: ignore
    _EASY_PROMPT = None

from grader import (
    WorkflowPlan,
    WorkflowStep,
    WorkflowCondition,
    ErrorBranch,
    Edge,
)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME   = os.getenv("MODEL_NAME",   "<your-active-model>")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

MAX_TOKENS        = 1500
TEMPERATURE       = 0.2
REQUEST_SLEEP     = 1.0
SUCCESS_THRESHOLD = 0.75

BENCHMARK  = "WorkflowEnv"
TASK_NAMES = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Task prompts
# ---------------------------------------------------------------------------

TASKS: dict[str, str] = {
    "easy": _EASY_PROMPT or (
        "When a new row is added to a Google Sheet named 'Lead Tracker' "
        "(columns: id, name, email, company, source), send an email via Gmail "
        "to sales-team@company.com. Subject: 'New lead: {{name}}'. "
        "Body: all row data. Step 1 reads the row; Step 2 sends the email "
        "with input_from pointing to Step 1's step_id. Build the workflow."
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
        "In parallel, post a digest message to the #sales-digest Slack channel "
        "summarising the totals. "
        "If the Postgres query fails for any reason, trigger a PagerDuty alert "
        "with severity 'critical'. Build the full workflow."
    ),
}

SUPPORTED_TOOLS = [
    "google_sheets", "gmail", "slack", "github",
    "jira", "postgres", "bigquery", "python",
    "scheduler", "pagerduty", "webhook", "http",
]
SYSTEM_PROMPT = f"""You are a workflow automation expert.

Return ONLY valid JSON.

{{
  "trigger": "google_sheets.row_added",
  "trigger_config": {{
    "sheet_name": "Lead Tracker"
  }},
  "condition": null,
  "error_branch": null,
  "steps": [
    {{
      "step_id": "s1",
      "tool": "google_sheets",
      "action": "get_row",
      "input_from": null,
      "params": {{
        "sheet_name": "Lead Tracker"
      }},
      "branch": null
    }},
    {{
      "step_id": "s2",
      "tool": "gmail",
      "action": "send_email",
      "input_from": "s1",
      "params": {{
        "to": "sales-team@company.com",
        "subject": "New lead: {{name}}",
        "body": "{{row}}"
      }},
      "branch": null
    }}
  ],
  "edges": [
    {{
      "from_": "trigger",
      "to": "s1",
      "label": null
    }},
    {{
      "from_": "s1",
      "to": "s2",
      "label": null
    }}
  ]
}}

Supported tools: {', '.join(SUPPORTED_TOOLS)}
"""
# ---------------------------------------------------------------------------
# Mandatory structured stdout logging
# Plain text, NOT JSON. Do not change field names or order.
# ---------------------------------------------------------------------------

def log_start(*, task: str, env: str, model: str) -> None:
    """[START] task=<name> env=<benchmark> model=<model>"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    *,
    step:   int,
    action: str,
    reward: float,
    done:   bool,
    error:  Optional[str],
) -> None:
    """[STEP] step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>"""
    done_str   = "true" if done else "false"
    error_str  = error if error is not None else "null"
    action_str = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP]  step={step} action={action_str} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(
    *,
    success: bool,
    steps: int,
    rewards: list[float],
) -> None:
    """[END] success=<true|false> steps=<n> rewards=<r1,r2,...>"""
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def build_client() -> OpenAI:
    missing = [name for name, val in [
        ("API_BASE_URL", API_BASE_URL),
        ("MODEL_NAME",   MODEL_NAME),
        ("HF_TOKEN",     HF_TOKEN),
    ] if not val or val.startswith("<")]
    if missing:
        print(
            f"[ERROR] Missing or unconfigured env vars: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def call_llm(client: OpenAI, prompt: str, verbose: bool = False) -> str:
    if verbose:
        print(f"[DEBUG] Calling model={MODEL_NAME} ...", file=sys.stderr, flush=True)

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=False,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    if verbose:
        print(f"[DEBUG] LLM raw ({len(raw)} chars):\n{raw}\n",
              file=sys.stderr, flush=True)
    return raw


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def parse_plan(raw: str) -> WorkflowPlan:
    text = raw.strip()
    # Strip accidental markdown fences
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
# Single-task runner
# ---------------------------------------------------------------------------

def run_task(
    client: OpenAI,
    grader,
    task_id: str,
    max_retries: int = 1,
    verbose: bool = False,
) -> float:
    """
    Run one WorkflowEnv task (single-turn: agent submits one plan).
    Returns score in [0.0, 1.0].
    """
    rewards: list[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False
    error_msg: Optional[str] = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        raw = None
        plan = None

        for attempt in range(1, max_retries + 2):
            try:
                raw = call_llm(client, TASKS[task_id], verbose=verbose)
                plan = parse_plan(raw)
                break
            except Exception as exc:
                error_msg = f"LLM error attempt {attempt}: {exc}"
                if verbose:
                    traceback.print_exc(file=sys.stderr)
                time.sleep(REQUEST_SLEEP)

        if plan is None:
            plan = WorkflowPlan(trigger="", steps=[], edges=[])

        result = grader.grade(task_id, plan)
        reward = round(result.total, 2)
        done = True

        rewards.append(reward)
        steps_taken = 1

        try:
            action_log = json.dumps(
                {
                    "trigger": plan.trigger,
                    "steps": [s.step_id for s in plan.steps],
                }
            )
        except Exception:
            action_log = raw[:200] if raw else "{}"

        log_step(
            step=1,
            action=action_log,
            reward=reward,
            done=done,
            error=error_msg,
        )

        score = min(max(reward, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        error_msg = str(exc)
        if verbose:
            traceback.print_exc(file=sys.stderr)
        log_step(step=1, action="", reward=0.00, done=True, error=error_msg)
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return score

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="WorkflowEnv inference")
    parser.add_argument(
        "--task", choices=TASK_NAMES, default=None,
        help="Run a single task instead of all three",
    )
    parser.add_argument(
        "--max-retries", type=int, default=1,
        help="Extra LLM retries on JSON parse failure (default: 1)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print LLM output and debug info to stderr",
    )
    args = parser.parse_args()

    task_ids = [args.task] if args.task else TASK_NAMES

    client = build_client()
    grader = _GRADER_CLS()

    all_scores: dict[str, float] = {}

    for i, task_id in enumerate(task_ids):
        score = run_task(
            client, grader, task_id,
            max_retries=args.max_retries,
            verbose=args.verbose,
        )
        all_scores[task_id] = score
        if i < len(task_ids) - 1:
            time.sleep(REQUEST_SLEEP)

    for tid, sc in all_scores.items():
        status = "PASS" if sc >= SUCCESS_THRESHOLD else "FAIL"
        print(f"  {tid:<8}  [{status}]  {sc:.2f}  ", file=sys.stderr)
        avg = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
        passed = sum(1 for s in all_scores.values() if s >= SUCCESS_THRESHOLD)
        print(f"\n  Average : {avg:.2f}  |  Passed: {passed}/{len(all_scores)}",
          file=sys.stderr)

    sys.exit(1 if any(s < SUCCESS_THRESHOLD for s in all_scores.values()) else 0)


if __name__ == "__main__":
    main()
