"""
WorkflowEnv Grader
------------------
Scores agent-produced workflow plans against ground truth.
Each task returns a float in [0.0, 1.0] with partial credit.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class WorkflowStep:
    step_id: str
    tool: str
    action: str
    input_from: Optional[str] = None
    params: dict = field(default_factory=dict)
    branch: Optional[str] = None   # "if_true" | "if_false" | None


@dataclass
class WorkflowCondition:
    field: str
    operator: str   # "contains" | "equals" | "gt" | "lt"
    value: str


@dataclass
class ErrorBranch:
    on_step: str
    tool: str
    action: str
    params: dict = field(default_factory=dict)


@dataclass
class Edge:
    from_: str
    to: str
    label: Optional[str] = None   # "if_true" | "if_false" | "on_error" | None


@dataclass
class WorkflowPlan:
    """The agent's submitted workflow plan."""
    trigger: str
    trigger_config: dict = field(default_factory=dict)
    steps: list[WorkflowStep] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    condition: Optional[WorkflowCondition] = None
    error_branch: Optional[ErrorBranch] = None


@dataclass
class GradeResult:
    total: float
    breakdown: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def __repr__(self):
        lines = [f"Score: {self.total:.2f}  [{bar}]", "Breakdown:"]
        for k, v in self.breakdown.items():
            lines.append(f"  {tick} {k:<35} {v:.2f}")
        if self.notes:
            lines.append("Notes:")
            for n in self.notes:
                lines.append(f"  - {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Base grader
# ---------------------------------------------------------------------------

class BaseGrader:
    """Shared helpers used by all task graders."""

    def _tools_in_plan(self, plan: WorkflowPlan) -> set[str]:
        tools = {s.tool for s in plan.steps}
        if plan.error_branch:
            tools.add(plan.error_branch.tool)
        return tools

    def _steps_by_tool(self, plan: WorkflowPlan, tool: str) -> list[WorkflowStep]:
        return [s for s in plan.steps if s.tool == tool]

    def _steps_on_branch(self, plan: WorkflowPlan, branch: str) -> list[WorkflowStep]:
        return [s for s in plan.steps if s.branch == branch]

    def _has_edge(self, plan: WorkflowPlan, from_: str, to: str,
                  label: Optional[str] = None) -> bool:
        for e in plan.edges:
            frm = e.from_ if hasattr(e, "from_") else e.get("from_", e.get("from"))
            if frm == from_ and e.to == to:
                if label is None or e.label == label:
                    return True
        return False

    def _upstream_of(self, plan: WorkflowPlan, step_id: str) -> Optional[str]:
        """Return the step_id that feeds into step_id, or None."""
        step = next((s for s in plan.steps if s.step_id == step_id), None)
        return step.input_from if step else None

    def _clamp(self, v: float) -> float:
        return round(min(max(v, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Task graders
# ---------------------------------------------------------------------------

class EasyGrader(BaseGrader):
    """
    """

    GROUND_TRUTH_TRIGGER = "google_sheets.row_added"
    REQUIRED_TOOLS = {"google_sheets", "gmail"}

    def grade(self, plan: WorkflowPlan) -> GradeResult:
        scores: dict[str, float] = {}
        notes: list[str] = []

        # 1. Trigger correct (0.25)
        if plan.trigger == self.GROUND_TRUTH_TRIGGER:
            scores["trigger_correct"] = 0.25
        else:
            scores["trigger_correct"] = 0.0
            notes.append(f"Expected trigger '{self.GROUND_TRUTH_TRIGGER}', got '{plan.trigger}'")

        # 2. Required tools present (0.25)
        found = self._tools_in_plan(plan) & self.REQUIRED_TOOLS
        scores["steps_present"] = self._clamp(
            0.25 * len(found) / len(self.REQUIRED_TOOLS)
        )
        if found != self.REQUIRED_TOOLS:
            notes.append(f"Missing tools: {self.REQUIRED_TOOLS - found}")

        # 3. Tool assignment correct: s1=google_sheets, s2=gmail (0.25)
        sheets_steps = self._steps_by_tool(plan, "google_sheets")
        gmail_steps  = self._steps_by_tool(plan, "gmail")
        tool_score = 0.0
        if sheets_steps:
            tool_score += 0.125
        if gmail_steps:
            tool_score += 0.125
            # Gmail step should depend on sheets step
            gmail_step = gmail_steps[0]
            if sheets_steps and gmail_step.input_from == sheets_steps[0].step_id:
                tool_score += 0.0   # already counted above; data-flow checked in edges
        scores["tools_correct"] = self._clamp(tool_score)

        if len(plan.steps) >= 2:
            s1_id = sheets_steps[0].step_id if sheets_steps else None
            s2_id = gmail_steps[0].step_id  if gmail_steps  else None
            edge_score = 0.0
            if s1_id and self._has_edge(plan, "trigger", s1_id):
                edge_score += 0.125
            if s1_id and s2_id and self._has_edge(plan, s1_id, s2_id):
                edge_score += 0.125
            scores["edges_correct"] = self._clamp(edge_score)
        else:
            scores["edges_correct"] = 0.0

        total = self._clamp(sum(scores.values()))
        return GradeResult(total=total, breakdown=scores, notes=notes)


class MediumGrader(BaseGrader):
    """
          if_true: Jira create + Slack post  |  if_false: GitHub add_label
    Max score: 1.0  (5 criteria with adjusted weights)
    """

    GROUND_TRUTH_TRIGGER = "github.issue.opened"

    def grade(self, plan: WorkflowPlan) -> GradeResult:
        scores: dict[str, float] = {}
        notes: list[str] = []

        # 1. Trigger correct (0.20)
        scores["trigger_correct"] = (
            0.20 if plan.trigger == self.GROUND_TRUTH_TRIGGER else 0.0
        )
        if not scores["trigger_correct"]:
            notes.append(f"Expected 'github.issue.opened', got '{plan.trigger}'")

        # 2. Condition present and references 'critical' (0.20)
        cond = plan.condition
        if cond and "critical" in str(cond.value).lower():
            scores["condition_present"] = 0.20
        elif cond:
            scores["condition_present"] = 0.10
            notes.append("Condition present but does not reference 'critical'")
        else:
            scores["condition_present"] = 0.0

        # 3. if_true branch: Jira AND Slack (0.25)
        true_branch = self._steps_on_branch(plan, "if_true")
        true_tools  = {s.tool for s in true_branch}
        has_jira  = "jira"  in true_tools
        has_slack = "slack" in true_tools
        branch_true_score = 0.0
        if has_jira:
            branch_true_score += 0.125
        if has_slack:
            branch_true_score += 0.125
        if not has_jira:
            notes.append("jira.create_issue missing from if_true branch")
        if not has_slack:
            notes.append("slack.post_message missing from if_true branch")
        scores["if_true_branch_complete"] = self._clamp(branch_true_score)

        # 4. if_false branch: GitHub add_label (0.20)
        false_branch = self._steps_on_branch(plan, "if_false")
        false_tools  = {s.tool for s in false_branch}
        if "github" in false_tools:
            gh_step = next(s for s in false_branch if s.tool == "github")
            scores["if_false_branch_complete"] = (
                0.20 if "label" in gh_step.action else 0.10
            )
        else:
            scores["if_false_branch_complete"] = 0.0
            notes.append("github.add_label missing from if_false branch")

        # 5. Edges: both branches have edges from condition node (0.15)
        edge_score = 0.0
        true_steps  = [s.step_id for s in true_branch]
        false_steps = [s.step_id for s in false_branch]
        true_edges  = [e for e in plan.edges if e.label == "if_true"  and e.to in true_steps]
        false_edges = [e for e in plan.edges if e.label == "if_false" and e.to in false_steps]
        if true_edges:
            edge_score += 0.075
        if false_edges:
            edge_score += 0.075
        scores["edges_correct"] = self._clamp(edge_score)

        total = self._clamp(sum(scores.values()))
        return GradeResult(total=total, breakdown=scores, notes=notes)


class HardGrader(BaseGrader):
    """
          BigQuery insert (parallel) + Slack digest
    Max score: 1.0  (6 criteria + 0.05 bonus)
    """

    GROUND_TRUTH_TRIGGER = "scheduler.cron"
    REQUIRED_TOOLS = {"postgres", "python", "bigquery", "slack"}

    def grade(self, plan: WorkflowPlan) -> GradeResult:
        scores: dict[str, float] = {}
        notes: list[str] = []

        # 1. Trigger: scheduler.cron with some expression (0.15)
        if plan.trigger == self.GROUND_TRUTH_TRIGGER:
            has_expr = bool(plan.trigger_config.get("expression"))
            scores["trigger_correct"] = 0.15 if has_expr else 0.10
            if not has_expr:
                notes.append("Cron expression missing from trigger_config")
        else:
            scores["trigger_correct"] = 0.0
            notes.append(f"Expected 'scheduler.cron', got '{plan.trigger}'")

        # 2. All 4 required tools present (0.20)
        found = self._tools_in_plan(plan) & self.REQUIRED_TOOLS
        scores["steps_present"] = self._clamp(
            0.20 * len(found) / len(self.REQUIRED_TOOLS)
        )
        if found != self.REQUIRED_TOOLS:
            notes.append(f"Missing tools: {self.REQUIRED_TOOLS - found}")

        pg_steps  = self._steps_by_tool(plan, "postgres")
        py_steps  = self._steps_by_tool(plan, "python")
        bq_steps  = self._steps_by_tool(plan, "bigquery")
        sl_steps  = self._steps_by_tool(plan, "slack")

        order_score = 0.0
        if pg_steps and py_steps:
            if py_steps[0].input_from == pg_steps[0].step_id:
                order_score += 0.10
        if py_steps and bq_steps:
            if bq_steps[0].input_from == py_steps[0].step_id:
                order_score += 0.05
        if py_steps and sl_steps:
            if sl_steps[0].input_from == py_steps[0].step_id:
                order_score += 0.05
        scores["step_order_correct"] = self._clamp(order_score)
        if order_score < 0.20:
            notes.append("Step ordering or input_from chain is incorrect")

        # 4. Parallel fan-out: BQ and Slack both depend on Python, not each other (0.15)
        fan_score = 0.0
        if bq_steps and sl_steps and py_steps:
            py_id = py_steps[0].step_id
            bq_depends_on_py = bq_steps[0].input_from == py_id
            sl_depends_on_py = sl_steps[0].input_from == py_id
            bq_not_sl = sl_steps[0].step_id not in (bq_steps[0].input_from or "")
            sl_not_bq = bq_steps[0].step_id not in (sl_steps[0].input_from or "")
            if bq_depends_on_py and sl_depends_on_py:
                fan_score += 0.10
            if bq_not_sl and sl_not_bq:
                fan_score += 0.05
        scores["parallel_fan_out"] = self._clamp(fan_score)

        # 5. Error branch: PagerDuty alert on Postgres failure (0.20)
        eb = plan.error_branch
        if eb:
            eb_score = 0.0
            if eb.tool == "pagerduty":
                eb_score += 0.10
            if eb.action in ("trigger_alert", "create_incident"):
                eb_score += 0.05
            if pg_steps and eb.on_step == pg_steps[0].step_id:
                eb_score += 0.05
            scores["error_branch_present"] = self._clamp(eb_score)
        else:
            scores["error_branch_present"] = 0.0

        # 6. Edges: all 5 edges including on_error (0.10)
        expected_edge_count = 6
        edge_score = self._clamp(
            0.10 * min(len(plan.edges), expected_edge_count) / expected_edge_count
        )
        scores["edges_correct"] = edge_score

        total = self._clamp(sum(scores.values()))

        # Bonus: PagerDuty severity == "critical" (+0.05, capped at 1.0)
        bonus_notes = []
        if eb and eb.params.get("severity") == "critical":
            total = self._clamp(total + 0.05)
            bonus_notes.append("Bonus +0.05: PagerDuty severity correctly set to 'critical'")

        return GradeResult(
            total=total,
            breakdown=scores,
            notes=notes + bonus_notes,
        )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class WorkflowGrader:

    _graders = {
        "easy":   EasyGrader(),
        "medium": MediumGrader(),
        "hard":   HardGrader(),
    }

    def grade(self, task_id: str, plan: WorkflowPlan) -> GradeResult:
        grader = self._graders.get(task_id)
        if not grader:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose: easy | medium | hard")
        return grader.grade(plan)

    def grade_all(self, plans: dict[str, WorkflowPlan]) -> dict[str, GradeResult]:
        return {tid: self.grade(tid, plan) for tid, plan in plans.items()}
