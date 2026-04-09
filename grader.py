"""
WorkflowEnv Grader
------------------
Scores agent-produced workflow plans against ground truth.
Each task returns a float in [0.0, 1.0] with partial credit.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkflowStep:
    step_id: str
    tool: str
    action: str
    input_from: Optional[str] = None
    params: dict = field(default_factory=dict)
    branch: Optional[str] = None


@dataclass
class WorkflowCondition:
    field: str
    operator: str
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
    label: Optional[str] = None


@dataclass
class WorkflowPlan:
    trigger: str
    trigger_config: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    condition: Optional[WorkflowCondition] = None
    error_branch: Optional[ErrorBranch] = None


@dataclass
class GradeResult:
    total: float
    breakdown: dict
    notes: list = field(default_factory=list)

    def __repr__(self):
        filled = round(self.total * 20)
        bar = "X" * filled + "." * (20 - filled)
        lines = [f"Score: {self.total:.2f} [{bar}]", "Breakdown:"]
        for k, v in self.breakdown.items():
            tick = "v" if v > 0 else "x"
            lines.append(f"  {tick} {k:<35} {v:.2f}")
        if self.notes:
            lines.append("Notes:")
            for n in self.notes:
                lines.append(f"  - {n}")
        return "\n".join(lines)


class BaseGrader:

    def _tools_in_plan(self, plan):
        tools = {s.tool for s in plan.steps}
        if plan.error_branch:
            tools.add(plan.error_branch.tool)
        return tools

    def _steps_by_tool(self, plan, tool):
        return [s for s in plan.steps if s.tool == tool]

    def _steps_on_branch(self, plan, branch):
        # Accept both "if_true"/"if_false" and "true"/"false" from LLMs
        alt = branch.replace("if_", "") if branch.startswith("if_") else "if_" + branch
        return [s for s in plan.steps if s.branch in (branch, alt)]

    def _has_edge(self, plan, from_, to, label=None):
        for e in plan.edges:
            frm = e.from_ if hasattr(e, "from_") else e.get("from_", e.get("from", ""))
            if frm == from_ and e.to == to:
                if label is None or e.label == label:
                    return True
        return False

    def _clamp(self, v):
        return round(min(max(v, 0.0), 1.0), 4)


class EasyGrader(BaseGrader):
    GROUND_TRUTH_TRIGGER = "google_sheets.row_added"
    REQUIRED_TOOLS = {"google_sheets", "gmail"}

    def grade(self, plan):
        scores = {}
        notes = []

        if plan.trigger == self.GROUND_TRUTH_TRIGGER:
            scores["trigger_correct"] = 0.25
        else:
            scores["trigger_correct"] = 0.0
            notes.append(f"Expected '{self.GROUND_TRUTH_TRIGGER}', got '{plan.trigger}'")

        found = self._tools_in_plan(plan) & self.REQUIRED_TOOLS
        scores["steps_present"] = self._clamp(0.25 * len(found) / len(self.REQUIRED_TOOLS))
        if found != self.REQUIRED_TOOLS:
            notes.append(f"Missing tools: {self.REQUIRED_TOOLS - found}")

        sheets_steps = self._steps_by_tool(plan, "google_sheets")
        gmail_steps  = self._steps_by_tool(plan, "gmail")
        tool_score = 0.0
        if sheets_steps:
            tool_score += 0.125
        if gmail_steps:
            tool_score += 0.125
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
            notes.append("Fewer than 2 steps")

        total = self._clamp(sum(scores.values()))
        return GradeResult(total=total, breakdown=scores, notes=notes)


class MediumGrader(BaseGrader):
    GROUND_TRUTH_TRIGGER = "github.issue.opened"

    def grade(self, plan):
        scores = {}
        notes = []

        # Accept common LLM variants of the trigger name
        trigger_ok = plan.trigger in (
            "github.issue.opened",
            "github.issue_opened",
            "github.issues.opened",
        )
        scores["trigger_correct"] = 0.20 if trigger_ok else 0.0
        if not scores["trigger_correct"]:
            notes.append(f"Expected 'github.issue.opened', got '{plan.trigger}'")

        cond = plan.condition
        if cond and "critical" in str(cond.value).lower():
            scores["condition_present"] = 0.20
        elif cond:
            scores["condition_present"] = 0.10
            notes.append("Condition present but does not reference 'critical'")
        else:
            scores["condition_present"] = 0.0
            notes.append("No condition node found")

        true_branch = self._steps_on_branch(plan, "if_true")
        true_tools  = {s.tool for s in true_branch}
        has_jira  = "jira"  in true_tools
        has_slack = "slack" in true_tools
        branch_score = 0.0
        if has_jira:
            branch_score += 0.125
        if has_slack:
            branch_score += 0.125
        if not has_jira:
            notes.append("jira missing from if_true branch")
        if not has_slack:
            notes.append("slack missing from if_true branch")
        scores["if_true_branch_complete"] = self._clamp(branch_score)

        false_branch = self._steps_on_branch(plan, "if_false")
        false_tools  = {s.tool for s in false_branch}
        if "github" in false_tools:
            gh = next(s for s in false_branch if s.tool == "github")
            scores["if_false_branch_complete"] = 0.20 if "label" in gh.action else 0.10
        else:
            scores["if_false_branch_complete"] = 0.0
            notes.append("github.add_label missing from if_false branch")

        true_steps  = [s.step_id for s in true_branch]
        false_steps = [s.step_id for s in false_branch]
        true_edges  = [e for e in plan.edges if e.label in ("if_true",  "true")  and e.to in true_steps]
        false_edges = [e for e in plan.edges if e.label in ("if_false", "false") and e.to in false_steps]
        edge_score = 0.0
        if true_edges:
            edge_score += 0.075
        if false_edges:
            edge_score += 0.075
        scores["edges_correct"] = self._clamp(edge_score)

        total = self._clamp(sum(scores.values()))
        return GradeResult(total=total, breakdown=scores, notes=notes)


class HardGrader(BaseGrader):
    GROUND_TRUTH_TRIGGER = "scheduler.cron"
    REQUIRED_TOOLS = {"postgres", "python", "bigquery", "slack"}

    def grade(self, plan):
        scores = {}
        notes = []

        # Accept common LLM variants
        trigger_ok = plan.trigger in (
            "scheduler.cron",
            "scheduler.time_based",
            "scheduler.daily",
            "cron",
        )
        if trigger_ok:
            has_expr = bool(
                plan.trigger_config.get("expression") or
                plan.trigger_config.get("time") or
                plan.trigger_config.get("schedule")
            )
            scores["trigger_correct"] = 0.15 if has_expr else 0.10
            if not has_expr:
                notes.append("Cron expression missing from trigger_config")
        else:
            scores["trigger_correct"] = 0.0
            notes.append(f"Expected 'scheduler.cron', got '{plan.trigger}'")

        found = self._tools_in_plan(plan) & self.REQUIRED_TOOLS
        scores["steps_present"] = self._clamp(0.20 * len(found) / len(self.REQUIRED_TOOLS))
        if found != self.REQUIRED_TOOLS:
            notes.append(f"Missing tools: {self.REQUIRED_TOOLS - found}")

        pg_steps = self._steps_by_tool(plan, "postgres")
        py_steps = self._steps_by_tool(plan, "python")
        bq_steps = self._steps_by_tool(plan, "bigquery")
        sl_steps = self._steps_by_tool(plan, "slack")

        order_score = 0.0
        if pg_steps and py_steps and py_steps[0].input_from == pg_steps[0].step_id:
            order_score += 0.10
        if py_steps and bq_steps and bq_steps[0].input_from == py_steps[0].step_id:
            order_score += 0.05
        if py_steps and sl_steps and sl_steps[0].input_from == py_steps[0].step_id:
            order_score += 0.05
        scores["step_order_correct"] = self._clamp(order_score)
        if order_score < 0.20:
            notes.append("Step ordering or input_from chain is incorrect")

        fan_score = 0.0
        if bq_steps and sl_steps and py_steps:
            py_id = py_steps[0].step_id
            if bq_steps[0].input_from == py_id and sl_steps[0].input_from == py_id:
                fan_score += 0.10
            bq_not_sl = sl_steps[0].step_id not in (bq_steps[0].input_from or "")
            sl_not_bq = bq_steps[0].step_id not in (sl_steps[0].input_from or "")
            if bq_not_sl and sl_not_bq:
                fan_score += 0.05
        scores["parallel_fan_out"] = self._clamp(fan_score)

        eb = plan.error_branch
        if eb:
            eb_score = 0.0
            if eb.tool == "pagerduty":
                eb_score += 0.10
            if eb.action in ("trigger_alert", "create_incident",
                              "trigger_incident", "create_alert", "alert"):
                eb_score += 0.05
            if pg_steps and (
                eb.on_step == pg_steps[0].step_id or
                eb.on_step in ("s1", "postgres", "pg")
            ):
                eb_score += 0.05
            scores["error_branch_present"] = self._clamp(eb_score)
        else:
            scores["error_branch_present"] = 0.0
            notes.append("No error_branch defined")

        expected_edge_count = 5
        scores["edges_correct"] = self._clamp(
            0.10 * min(len(plan.edges), expected_edge_count) / expected_edge_count
        )

        total = self._clamp(sum(scores.values()))
        bonus_notes = []
        if eb and eb.params.get("severity") == "critical":
            total = self._clamp(total + 0.05)
            bonus_notes.append("Bonus +0.05: PagerDuty severity=critical")

        return GradeResult(total=total, breakdown=scores, notes=notes + bonus_notes)


class WorkflowGrader:
    _graders = {
        "easy":   EasyGrader(),
        "medium": MediumGrader(),
        "hard":   HardGrader(),
    }

    def grade(self, task_id, plan):
        grader = self._graders.get(task_id)
        if not grader:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose: easy | medium | hard")
        return grader.grade(plan)

    def grade_all(self, plans):
        return {tid: self.grade(tid, plan) for tid, plan in plans.items()}  
