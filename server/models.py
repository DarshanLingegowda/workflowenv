"""
server/models.py — WorkflowEnv typed Pydantic models
------------------------------------------------------
Defines the Action, Observation, and State types required by OpenEnv spec.
All models use Pydantic for automatic validation and serialisation.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action — what the agent submits
# ---------------------------------------------------------------------------

class WorkflowStep(BaseModel):
    step_id:    str
    tool:       str
    action:     str
    input_from: Optional[str]       = None
    params:     dict                = Field(default_factory=dict)
    branch:     Optional[str]       = None   # "if_true" | "if_false" | None


class WorkflowCondition(BaseModel):
    field:    str
    operator: str   # "contains" | "equals" | "gt" | "lt"
    value:    str


class WorkflowErrorBranch(BaseModel):
    on_step: str
    tool:    str
    action:  str
    params:  dict = Field(default_factory=dict)


class WorkflowEdge(BaseModel):
    from_:  str = Field(..., alias="from_")
    to:     str
    label:  Optional[str] = None

    class Config:
        populate_by_name = True


class WorkflowAction(BaseModel):
    """The complete workflow plan submitted by the agent in one step()."""
    trigger:        str
    trigger_config: dict                          = Field(default_factory=dict)
    condition:      Optional[WorkflowCondition]   = None
    error_branch:   Optional[WorkflowErrorBranch] = None
    steps:          List[WorkflowStep]             = Field(default_factory=list)
    edges:          List[WorkflowEdge]             = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Observation — what the agent receives back
# ---------------------------------------------------------------------------

class WorkflowObservation(BaseModel):
    """Returned by reset() and step()."""
    task_description: str
    task_id:          str
    step_count:       int
    last_score:       float = 0.0
    feedback:         str   = ""
    done:             bool  = False


# ---------------------------------------------------------------------------
# State — internal episode state returned by state()
# ---------------------------------------------------------------------------

class WorkflowState(BaseModel):
    """Returned by state()."""
    episode_id: str = ""
    task_id:    str = "easy"
    step_count: int = 0


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: dict[str, dict] = {
    "easy": {
        "prompt": (
            "When a new row is added to a Google Sheet named 'Lead Tracker' "
            "(columns: id, name, email, company, source), send an email via Gmail "
            "to sales-team@company.com. Subject: 'New lead: {{name}}'. "
            "Body: all row data. Step 1 reads the row; Step 2 sends the email "
            "with input_from pointing to Step 1's step_id. Build the workflow."
        ),
    },
    "medium": {
        "prompt": (
            "When a GitHub issue is opened in the 'backend' repository: "
            "if the issue has the label 'critical', create a Jira ticket in "
            "project OPS and post a message to the #incidents Slack channel; "
            "if the label is not 'critical', just add a 'needs-triage' label "
            "to the GitHub issue. Build the conditional workflow."
        ),
    },
    "hard": {
        "prompt": (
            "Every day at 9:00 AM IST, fetch yesterday's sales records from a "
            "Postgres table named 'sales' (columns: id, product, amount, region). "
            "Transform the data using a Python script to calculate total sales per region. "
            "Write the aggregated results to a BigQuery table named 'analytics.daily_sales'. "
            "In parallel, post a digest message to the #sales-digest Slack channel "
            "summarising the totals. "
            "If the Postgres query fails for any reason, trigger a PagerDuty alert "
            "with severity 'critical'. Build the full workflow."
        ),
    },
}
