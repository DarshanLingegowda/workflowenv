"""
-------------------------------------
Full OpenEnv-compliant environment implementing:
"""

from __future__ import annotations

import uuid
from typing import Any

from .models import (
    WorkflowAction,
    WorkflowObservation,
    WorkflowState,
    TASKS,
)


class WorkflowEnvironment:
    """
    Real-world task: the agent receives a natural-language workflow
    description and must produce a valid structured automation plan.

    Each episode = one task. The agent has up to MAX_STEPS attempts
    """

    MAX_STEPS = 1

    def __init__(self) -> None:
        self._state: WorkflowState = WorkflowState(
            episode_id="", task_id="easy", step_count=0,
        )

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "easy") -> WorkflowObservation:
        """Start a new episode for the given task."""
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Choose: {list(TASKS)}"
            )
        self._state = WorkflowState(
            episode_id=str(uuid.uuid4()),
            task_id=task_id,
            step_count=0,
        )
        task = TASKS[task_id]
        return WorkflowObservation(
            task_description=task["prompt"],
            task_id=task_id,
            step_count=0,
            last_score=0.0,
            feedback="Episode started. Submit your workflow plan.",
            done=False,
        )

    def step(self, action: WorkflowAction) -> tuple[WorkflowObservation, float, bool, dict]:
        """
        Evaluate the agent's workflow plan.
        Returns (observation, reward, done, info).
        """
        from .grader_bridge import grade_action

        self._state.step_count += 1
        done = self._state.step_count >= self.MAX_STEPS

        result = grade_action(self._state.task_id, action)
        reward = result.total

        obs = WorkflowObservation(
            task_description=TASKS[self._state.task_id]["prompt"],
            task_id=self._state.task_id,
            step_count=self._state.step_count,
            last_score=round(reward, 4),
            feedback=repr(result),
            done=done,
        )

        info: dict[str, Any] = {
            "episode_id":  self._state.episode_id,
            "breakdown":   result.breakdown,
            "notes":       result.notes,
        }

        return obs, reward, done, info

    def state(self) -> WorkflowState:
        """Return current episode state (step count, task, episode ID)."""
        return self._state

    def close(self) -> None:
        pass
