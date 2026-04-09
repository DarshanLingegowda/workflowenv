"""
Unit tests for WorkflowEnv grader.
Run with:  python -m pytest test_grader.py -v
"""

import pytest
from grader import (
    WorkflowGrader, WorkflowPlan, WorkflowStep,
    WorkflowCondition, ErrorBranch, Edge,
)

grader = WorkflowGrader()


# ---------------------------------------------------------------------------
# Helpers to build plans quickly
# ---------------------------------------------------------------------------

def easy_perfect() -> WorkflowPlan:
    return WorkflowPlan(
        trigger="google_sheets.row_added",
        trigger_config={"sheet_name": "Lead Tracker"},
        steps=[
            WorkflowStep("s1", "google_sheets", "get_row"),
            WorkflowStep("s2", "gmail", "send_email", input_from="s1"),
        ],
        edges=[
            Edge("trigger", "s1"),
            Edge("s1", "s2"),
        ],
    )


def medium_perfect() -> WorkflowPlan:
    return WorkflowPlan(
        trigger="github.issue.opened",
        trigger_config={"repo": "backend"},
        condition=WorkflowCondition("issue.labels", "contains", "critical"),
        steps=[
            WorkflowStep("s1",  "github", "get_issue"),
            WorkflowStep("s2a", "jira",   "create_issue",  input_from="s1", branch="if_true"),
            WorkflowStep("s2b", "slack",  "post_message",  input_from="s1", branch="if_true"),
            WorkflowStep("s3",  "github", "add_label",     input_from="s1", branch="if_false"),
        ],
        edges=[
            Edge("trigger",   "s1"),
            Edge("s1",        "condition"),
            Edge("condition", "s2a", label="if_true"),
            Edge("condition", "s2b", label="if_true"),
            Edge("condition", "s3",  label="if_false"),
        ],
    )


def hard_perfect() -> WorkflowPlan:
    return WorkflowPlan(
        trigger="scheduler.cron",
        trigger_config={"expression": "0 9 * * *", "timezone": "Asia/Kolkata"},
        steps=[
            WorkflowStep("s1",  "postgres", "query"),
            WorkflowStep("s2",  "python",   "apply_mapping", input_from="s1"),
            WorkflowStep("s3a", "bigquery", "insert_rows",   input_from="s2"),
            WorkflowStep("s3b", "slack",    "post_message",  input_from="s2"),
        ],
        edges=[
            Edge("trigger", "s1"),
            Edge("s1",      "s2"),
            Edge("s2",      "s3a"),
            Edge("s2",      "s3b"),
            Edge("s1",      "error_branch", label="on_error"),
        ],
        error_branch=ErrorBranch(
            on_step="s1",
            tool="pagerduty",
            action="trigger_alert",
            params={"severity": "critical", "message": "Postgres failed"},
        ),
    )


# ---------------------------------------------------------------------------
# Easy task tests
# ---------------------------------------------------------------------------

class TestEasyGrader:

    def test_perfect_plan_scores_1(self):
        result = grader.grade("easy", easy_perfect())
        assert result.total == 1.0

    def test_wrong_trigger_loses_025(self):
        plan = easy_perfect()
        plan.trigger = "webhook.receive_post"
        result = grader.grade("easy", plan)
        assert result.breakdown["trigger_correct"] == 0.0
        assert result.total <= 0.75

    def test_missing_gmail_loses_steps_credit(self):
        plan = easy_perfect()
        plan.steps = [s for s in plan.steps if s.tool != "gmail"]
        result = grader.grade("easy", plan)
        assert result.breakdown["steps_present"] < 0.25

    def test_empty_plan_scores_0(self):
        plan = WorkflowPlan(trigger="", steps=[], edges=[])
        result = grader.grade("easy", plan)
        assert result.total == 0.0

    def test_no_edges_loses_edge_credit(self):
        plan = easy_perfect()
        plan.edges = []
        result = grader.grade("easy", plan)
        assert result.breakdown["edges_correct"] == 0.0

    def test_extra_steps_dont_hurt(self):
        plan = easy_perfect()
        plan.steps.append(WorkflowStep("s3", "http", "post", input_from="s2"))
        plan.edges.append(Edge("s2", "s3"))
        result = grader.grade("easy", plan)
        assert result.total == 1.0

    def test_partial_plan_gets_partial_score(self):
        plan = WorkflowPlan(
            trigger="google_sheets.row_added",
            steps=[WorkflowStep("s1", "google_sheets", "get_row")],
            edges=[Edge("trigger", "s1")],
        )
        result = grader.grade("easy", plan)
        assert 0.0 < result.total < 1.0


# ---------------------------------------------------------------------------
# Medium task tests
# ---------------------------------------------------------------------------

class TestMediumGrader:

    def test_perfect_plan_scores_1(self):
        result = grader.grade("medium", medium_perfect())
        assert result.total == 1.0

    def test_wrong_trigger_loses_020(self):
        plan = medium_perfect()
        plan.trigger = "github.push"
        result = grader.grade("medium", plan)
        assert result.breakdown["trigger_correct"] == 0.0

    def test_missing_condition_loses_condition_credit(self):
        plan = medium_perfect()
        plan.condition = None
        result = grader.grade("medium", plan)
        assert result.breakdown["condition_present"] == 0.0

    def test_condition_without_critical_gets_partial(self):
        plan = medium_perfect()
        plan.condition = WorkflowCondition("issue.labels", "contains", "bug")
        result = grader.grade("medium", plan)
        assert result.breakdown["condition_present"] == 0.10

    def test_missing_jira_from_true_branch_loses_credit(self):
        plan = medium_perfect()
        plan.steps = [s for s in plan.steps if s.tool != "jira"]
        result = grader.grade("medium", plan)
        assert result.breakdown["if_true_branch_complete"] < 0.25

    def test_missing_false_branch_loses_credit(self):
        plan = medium_perfect()
        plan.steps = [s for s in plan.steps if s.branch != "if_false"]
        result = grader.grade("medium", plan)
        assert result.breakdown["if_false_branch_complete"] == 0.0

    def test_no_branch_labels_loses_edge_and_branch_credit(self):
        plan = medium_perfect()
        for s in plan.steps:
            s.branch = None
        result = grader.grade("medium", plan)
        assert result.breakdown["if_true_branch_complete"] == 0.0
        assert result.breakdown["if_false_branch_complete"] == 0.0

    def test_true_branch_only_gets_partial(self):
        plan = medium_perfect()
        plan.steps = [s for s in plan.steps if s.branch != "if_false"]
        result = grader.grade("medium", plan)
        assert 0.0 < result.total < 1.0


# ---------------------------------------------------------------------------
# Hard task tests
# ---------------------------------------------------------------------------

class TestHardGrader:

    def test_perfect_plan_scores_1(self):
        result = grader.grade("hard", hard_perfect())
        assert result.total == 1.0

    def test_wrong_trigger_loses_credit(self):
        plan = hard_perfect()
        plan.trigger = "webhook.receive_post"
        result = grader.grade("hard", plan)
        assert result.breakdown["trigger_correct"] == 0.0

    def test_cron_without_expression_gets_partial_trigger(self):
        plan = hard_perfect()
        plan.trigger_config = {}
        result = grader.grade("hard", plan)
        assert result.breakdown["trigger_correct"] == 0.10

    def test_missing_bigquery_loses_steps_credit(self):
        plan = hard_perfect()
        plan.steps = [s for s in plan.steps if s.tool != "bigquery"]
        result = grader.grade("hard", plan)
        assert result.breakdown["steps_present"] < 0.20

    def test_wrong_step_order_loses_order_credit(self):
        plan = hard_perfect()
        # Break the chain: python no longer reads from postgres
        for s in plan.steps:
            if s.tool == "python":
                s.input_from = None
        result = grader.grade("hard", plan)
        assert result.breakdown["step_order_correct"] < 0.20

    def test_serial_instead_of_parallel_loses_fan_credit(self):
        plan = hard_perfect()
        # Make slack depend on bigquery instead of python
        for s in plan.steps:
            if s.tool == "slack":
                s.input_from = "s3a"
        result = grader.grade("hard", plan)
        assert result.breakdown["parallel_fan_out"] < 0.15

    def test_missing_error_branch_loses_020(self):
        plan = hard_perfect()
        plan.error_branch = None
        result = grader.grade("hard", plan)
        assert result.breakdown["error_branch_present"] == 0.0
        assert result.total <= 0.80

    def test_error_branch_wrong_tool_gets_partial(self):
        plan = hard_perfect()
        plan.error_branch = ErrorBranch(
            on_step="s1", tool="slack", action="post_message"
        )
        result = grader.grade("hard", plan)
        assert 0.0 < result.breakdown["error_branch_present"] < 0.20

    def test_bonus_for_critical_severity(self):
        plan = hard_perfect()
        plan.error_branch.params["severity"] = "critical"
        result = grader.grade("hard", plan)
        assert result.total == 1.0   # 0.95 base + 0.05 bonus, capped at 1.0

    def test_no_bonus_without_critical_severity(self):
        plan = hard_perfect()
        plan.error_branch.params["severity"] = "warning"
        base = grader.grade("hard", plan)
        plan.error_branch.params["severity"] = "critical"
        with_bonus = grader.grade("hard", plan)
        assert with_bonus.total >= base.total


# ---------------------------------------------------------------------------
# Dispatcher tests
# ---------------------------------------------------------------------------

class TestWorkflowGrader:

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task_id"):
            grader.grade("ultra_hard", easy_perfect())

    def test_grade_all_returns_all_three(self):
        results = grader.grade_all({
            "easy":   easy_perfect(),
            "medium": medium_perfect(),
            "hard":   hard_perfect(),
        })
        assert set(results.keys()) == {"easy", "medium", "hard"}
        for r in results.values():
            assert r.total == 1.0

    def test_score_always_in_range(self):
        plans = [easy_perfect(), medium_perfect(), hard_perfect()]
        task_ids = ["easy", "medium", "hard"]
        for tid, plan in zip(task_ids, plans):
            result = grader.grade(tid, plan)
            assert 0.0 <= result.total <= 1.0

    def test_grade_result_repr_contains_score(self):
        result = grader.grade("easy", easy_perfect())
        assert "1.00" in repr(result)
