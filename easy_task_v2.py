"""
-----------------------------------------------------------------
Improvements over v1:
     subject template, and body template so ground truth is unambiguous.
     Adds: action_correct, field_mapping, data_flow.
     Partial credit on every criterion.
     {{lead_name}}, etc. rather than demanding an exact template string.
     Sheets and send_email / send for Gmail.
     Sheets step, not be null or point to trigger.
  7. Standalone: paste over EasyGrader + update TASKS["easy"] in
     inference.py as shown at the bottom of this file.

Run self-tests:
    python easy_task_v2.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re


# (copy-paste these if running standalone, or just import from grader)
try:
    from grader import (
        WorkflowStep, WorkflowCondition, ErrorBranch,
        Edge, WorkflowPlan, GradeResult, BaseGrader, WorkflowGrader,
    )
except ImportError:
    raise SystemExit("Put this file next to grader.py and re-run.")



EASY_TRIGGER          = "google_sheets.row_added"
EASY_SHEET_NAME       = "Lead Tracker"
EASY_COLUMNS          = ["id", "name", "email", "company", "source"]

SHEETS_VALID_ACTIONS  = {"get_row", "read_row", "fetch_row", "get_rows"}
GMAIL_VALID_ACTIONS   = {"send_email", "send", "compose_email", "send_message"}

# Any of these patterns in subject/body params counts as a name reference
NAME_FIELD_PATTERNS   = [
    r"\{\{.*\bname\b.*\}\}",
    r"\{\{.*\.A\b.*\}\}",
    r"\{\{.*lead.*\}\}",
    r"\{\{.*row\b.*\}\}",
    r"\{\{.*data\b.*\}\}",
]

# Any of these patterns in body counts as a "all row data" reference
ROW_DATA_PATTERNS     = [
    r"\{\{.*row\b.*\}\}",
    r"\{\{.*data\b.*\}\}",
    r"\{\{.*\*.*\}\}",
    r"\{\{.*fields\b.*\}\}",
    r"\{\{.*all\b.*\}\}",
    r"\{\{.*record\b.*\}\}",
]



EASY_TASK_PROMPT = """
You are building an automation workflow. Here are the exact requirements:

TRIGGER
  - Tool   : google_sheets
  - Event  : row_added
  - Sheet  : "Lead Tracker"
  - Columns in the sheet: id, name, email, company, source

STEPS (in order)

RULES
  - Step 2 must receive its data from Step 1 (set input_from to Step 1's step_id).
  - Do not add extra steps.
  - condition and error_branch are both null for this task.

Build the workflow JSON now.
""".strip()



class EasyGraderV2(BaseGrader):
    """
    7-criterion grader for the easy task.

    Criterion                weight   what it checks
    trigger_correct          0.20     trigger == google_sheets.row_added
    trigger_config           0.10     sheet_name present in trigger_config
    steps_present            0.10     both google_sheets + gmail in steps
    action_correct           0.15     valid read action on sheets step,
                                      valid send action on gmail step
    data_flow                0.20     gmail.input_from == sheets step_id
    field_mapping_subject    0.10     subject param references lead name
    field_mapping_body       0.15     body param references row/all data
    Total                    1.00
    """

    def grade(self, plan: WorkflowPlan) -> GradeResult:
        scores: dict[str, float] = {}
        notes:  list[str]        = []

        sheets_steps = self._steps_by_tool(plan, "google_sheets")
        gmail_steps  = self._steps_by_tool(plan, "gmail")
        s_step = sheets_steps[0] if sheets_steps else None
        g_step = gmail_steps[0]  if gmail_steps  else None

        if plan.trigger == EASY_TRIGGER:
            scores["trigger_correct"] = 0.20
        else:
            scores["trigger_correct"] = 0.0
            notes.append(
                f"Trigger: expected '{EASY_TRIGGER}', got '{plan.trigger}'"
            )

        cfg = plan.trigger_config or {}
        sheet_val = cfg.get("sheet_name", cfg.get("sheet", cfg.get("spreadsheet", "")))
        if sheet_val and EASY_SHEET_NAME.lower() in str(sheet_val).lower():
            scores["trigger_config"] = 0.10
        elif sheet_val:
            scores["trigger_config"] = 0.05   # sheet_name key present but wrong value
            notes.append(
                f"trigger_config.sheet_name: expected '{EASY_SHEET_NAME}', got '{sheet_val}'"
            )
        else:
            scores["trigger_config"] = 0.0
            notes.append("trigger_config missing sheet_name (or sheet / spreadsheet) key")

        tool_score = 0.0
        if s_step:
            tool_score += 0.05
        else:
            notes.append("No google_sheets step found")
        if g_step:
            tool_score += 0.05
        else:
            notes.append("No gmail step found")
        scores["steps_present"] = self._clamp(tool_score)

        action_score = 0.0
        if s_step:
            if s_step.action in SHEETS_VALID_ACTIONS:
                action_score += 0.075
            else:
                notes.append(
                    f"Sheets action '{s_step.action}' not in {SHEETS_VALID_ACTIONS}"
                )
        if g_step:
            if g_step.action in GMAIL_VALID_ACTIONS:
                action_score += 0.075
            else:
                notes.append(
                    f"Gmail action '{g_step.action}' not in {GMAIL_VALID_ACTIONS}"
                )
        scores["action_correct"] = self._clamp(action_score)

        if g_step and s_step:
            if g_step.input_from == s_step.step_id:
                scores["data_flow"] = 0.20
            elif g_step.input_from is not None:
                # input_from is set but points to wrong step
                scores["data_flow"] = 0.08
                notes.append(
                    f"Gmail input_from='{g_step.input_from}' but expected '{s_step.step_id}'"
                )
            else:
                scores["data_flow"] = 0.0
        elif not g_step:
            scores["data_flow"] = 0.0
        else:
            # g_step exists but no sheets step to chain from
            scores["data_flow"] = 0.0

        if g_step:
            subject = str(g_step.params.get("subject", ""))
            if any(re.search(p, subject, re.IGNORECASE) for p in NAME_FIELD_PATTERNS):
                scores["field_mapping_subject"] = 0.10
            elif subject:
                # subject exists but no dynamic reference
                scores["field_mapping_subject"] = 0.04
                notes.append("see grader feedback")

            else:
                scores["field_mapping_subject"] = 0.0
                notes.append("Gmail params missing 'subject' key")
        else:
            scores["field_mapping_subject"] = 0.0

        if g_step:
            body = str(g_step.params.get("body", g_step.params.get("message", "")))
            if any(re.search(p, body, re.IGNORECASE) for p in ROW_DATA_PATTERNS):
                scores["field_mapping_body"] = 0.15
            elif body:
                scores["field_mapping_body"] = 0.05
                notes.append("see grader feedback")

            else:
                scores["field_mapping_body"] = 0.0
                notes.append("Gmail params missing 'body' (or 'message') key")
        else:
            scores["field_mapping_body"] = 0.0

        total = self._clamp(sum(scores.values()))
        return GradeResult(total=total, breakdown=scores, notes=notes)



class WorkflowGraderV2(WorkflowGrader):
    """Same as WorkflowGrader but uses EasyGraderV2 for the easy task."""
    _graders = {
        "easy":   EasyGraderV2(),
        **{k: v for k, v in WorkflowGrader._graders.items() if k != "easy"},
    }



def _make_perfect() -> WorkflowPlan:
    return WorkflowPlan(
        trigger="google_sheets.row_added",
        trigger_config={"sheet_name": "Lead Tracker"},
        steps=[
            WorkflowStep(
                step_id="s1", tool="google_sheets", action="get_row",
                params={"sheet_name": "Lead Tracker", "fields": ["*"]},
            ),
            WorkflowStep(
                step_id="s2", tool="gmail", action="send_email",
                input_from="s1",
                params={
                    "to":      "sales-team@company.com",
                    "subject": "New lead: {{s1.row.name}}",
                    "body":    "{{s1.row}}",
                },
            ),
        ],
        edges=[Edge("trigger", "s1"), Edge("s1", "s2")],
    )


def _run_tests():
    g = EasyGraderV2()
    passed = failed = 0

    def check(name: str, plan: WorkflowPlan, min_score: float, max_score: float):
        nonlocal passed, failed
        r = g.grade(plan)
        ok = min_score <= r.total <= max_score
        sym = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
            for note in r.notes:
                pass  # notes available in r.notes
        else:
            passed += 1


    # Perfect plan
    check("perfect plan",                  _make_perfect(),          1.00, 1.00)

    # Wrong trigger
    p = _make_perfect(); p.trigger = "webhook.receive_post"
    check("wrong trigger",                 p,                        0.00, 0.80)

    # Missing sheet_name in config
    p = _make_perfect(); p.trigger_config = {}
    check("missing trigger_config",        p,                        0.80, 0.95)

    # Wrong sheet name
    p = _make_perfect(); p.trigger_config = {"sheet_name": "Sales"}
    check("wrong sheet name (partial)",    p,                        0.85, 0.98)

    p = _make_perfect(); p.steps = [s for s in p.steps if s.tool != "google_sheets"]
    check("missing sheets step",           p,                        0.00, 0.75)

    # No Gmail step
    p = _make_perfect(); p.steps = [s for s in p.steps if s.tool != "gmail"]
    check("missing gmail step",            p,                        0.00, 0.50)

    # Wrong Sheets action
    p = _make_perfect(); p.steps[0].action = "create_row"
    check("wrong sheets action",           p,                        0.80, 0.95)

    # Wrong Gmail action
    p = _make_perfect(); p.steps[1].action = "delete_email"
    check("wrong gmail action",            p,                        0.80, 0.95)

    # Gmail input_from is None (broken data flow)
    p = _make_perfect(); p.steps[1].input_from = None
    check("broken data flow (None)",       p,                        0.60, 0.85)

    # Gmail input_from points to wrong step
    p = _make_perfect(); p.steps[1].input_from = "trigger"
    check("data flow points to trigger",   p,                        0.65, 0.88)

    # Static subject (no template)
    p = _make_perfect(); p.steps[1].params["subject"] = "New lead arrived"
    check("static subject",                p,                        0.85, 0.98)

    # Subject uses {{row.A}} (alternative template syntax)
    p = _make_perfect(); p.steps[1].params["subject"] = "New lead: {{row.A}}"
    check("subject uses {{row.A}}",        p,                        1.00, 1.00)

    # Static body
    p = _make_perfect(); p.steps[1].params["body"] = "A new lead was added."
    check("static body",                   p,                        0.80, 0.95)

    # Body uses {{data}} instead of {{row}}
    p = _make_perfect(); p.steps[1].params["body"] = "Details: {{data}}"
    check("body uses {{data}}",            p,                        1.00, 1.00)

    # Completely empty plan
    check("empty plan",                    WorkflowPlan(trigger="", steps=[], edges=[]),
                                                                     0.00, 0.05)

    # Extra steps don't hurt
    p = _make_perfect()
    p.steps.append(WorkflowStep("s3", "slack", "post_message", input_from="s2"))
    p.edges.append(Edge("s2", "s3"))
    check("extra slack step (no penalty)", p,                        1.00, 1.00)

    print(f"  {passed} passed  |  {failed} failed\n")
    return failed == 0



INFERENCE_PATCH = '''

# 1. Import the v2 grader at the top:
from easy_task_v2 import EASY_TASK_PROMPT, WorkflowGraderV2

# 2. Replace the easy task prompt in TASKS:
TASKS = {
    "easy": {
    },
    "medium": { ... },
    "hard":   { ... },
}

# 3. Replace grader instantiation in run_task / main:
'''


if __name__ == "__main__":
    ok = _run_tests()

    print("inference.py patch instructions:")
    print(INFERENCE_PATCH)

    raise SystemExit(0 if ok else 1)
