# Minimal models for OpenEnv validation

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class WorkflowAction(BaseModel):
    trigger: str
    trigger_config: Dict[str, Any] = {}
    condition: Optional[Dict[str, Any]] = None
    error_branch: Optional[Dict[str, Any]] = None
    steps: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []


class WorkflowObservation(BaseModel):
    task_description: str
    task_id: str
    step_count: int
    last_score: float
    feedback: str
    done: bool


class WorkflowState(BaseModel):
    episode_id: str = ""
    task_id: str = "easy"
    step_count: int = 0
