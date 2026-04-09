"""
server/grader_bridge.py
-----------------------
Converts Pydantic WorkflowAction (from models.py) into the dataclass
WorkflowPlan (used by grader.py), then calls the appropriate grader.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from grader import (
    WorkflowPlan,
    WorkflowStep        as GraderStep,
    WorkflowCondition   as GraderCondition,
    ErrorBranch         as GraderErrorBranch,
    Edge                as GraderEdge,
    GradeResult,
)

try:
    from easy_task_v2 import WorkflowGraderV2
    _grader = WorkflowGraderV2()
except ImportError:
    from grader import WorkflowGrader
    _grader = WorkflowGrader()

from .models import WorkflowAction


def grade_action(task_id: str, action: WorkflowAction) -> GradeResult:

    steps = [
        GraderStep(
            step_id    = s.step_id,
            tool       = s.tool,
            action     = s.action,
            input_from = s.input_from,
            params     = s.params,
            branch     = s.branch,
        )
        for s in action.steps
    ]

    edges = [
        GraderEdge(
            from_ = e.from_,
            to    = e.to,
            label = e.label,
        )
        for e in action.edges
    ]

    condition = None
    if action.condition:
        condition = GraderCondition(
            field    = action.condition.field,
            operator = action.condition.operator,
            value    = action.condition.value,
        )

    error_branch = None
    if action.error_branch:
        error_branch = GraderErrorBranch(
            on_step = action.error_branch.on_step,
            tool    = action.error_branch.tool,
            action  = action.error_branch.action,
            params  = action.error_branch.params,
        )

    plan = WorkflowPlan(
        trigger        = action.trigger,
        trigger_config = action.trigger_config,
        steps          = steps,
        edges          = edges,
        condition      = condition,
        error_branch   = error_branch,
    )

    return _grader.grade(task_id, plan)
