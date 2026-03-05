"""Result dataclasses for NTILC orchestrator runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from models.software_layer import DispatchResult


@dataclass
class RetrievalCandidate:
    cluster_id: int
    score: float
    tool_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": int(self.cluster_id),
            "score": float(self.score),
            "tool_name": self.tool_name,
        }


@dataclass
class OrchestratorStepResult:
    candidate: RetrievalCandidate
    generated_text: str
    command: str
    dispatch_arguments: Dict[str, Any]
    dispatch_result: DispatchResult
    dispatch_block: str
    response_block: str
    action_id: Optional[int] = None
    action_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "candidate": self.candidate.to_dict(),
            "generated_text": self.generated_text,
            "command": self.command,
            "dispatch_arguments": self.dispatch_arguments,
            "dispatch_result": self.dispatch_result.to_dict(),
            "dispatch_block": self.dispatch_block,
            "response_block": self.response_block,
        }
        payload["action_id"] = self.action_id
        payload["action_text"] = self.action_text
        return payload


@dataclass
class OrchestratorRunResult:
    request: str
    plan_block: str
    candidates: List[RetrievalCandidate] = field(default_factory=list)
    steps: List[OrchestratorStepResult] = field(default_factory=list)
    atomic_actions: List[Dict[str, Any]] = field(default_factory=list)
    action_failures: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        if self.atomic_actions:
            planned_ids = {
                int(action.get("id"))
                for action in self.atomic_actions
                if isinstance(action, Mapping) and action.get("id") is not None
            }
            successful_ids = {
                int(step.action_id)
                for step in self.steps
                if step.action_id is not None and step.dispatch_result.ok
            }
            if planned_ids:
                return not self.action_failures and planned_ids.issubset(successful_ids)
        return any(step.dispatch_result.ok for step in self.steps)

    @property
    def final_step(self) -> Optional[OrchestratorStepResult]:
        return self.steps[-1] if self.steps else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": self.request,
            "plan_block": self.plan_block,
            "success": self.success,
            "candidates": [c.to_dict() for c in self.candidates],
            "steps": [s.to_dict() for s in self.steps],
            "atomic_actions": list(self.atomic_actions),
            "action_failures": list(self.action_failures),
        }
