"""Agent package exports."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "NTILCOrchestratorAgent": ("orchestrator.agent.runtime", "NTILCOrchestratorAgent"),
    "QwenOrchestratorModel": ("orchestrator.generation.model", "QwenOrchestratorModel"),
    "OrchestratorStepResult": ("orchestrator.results.types", "OrchestratorStepResult"),
    "OrchestratorRunResult": ("orchestrator.results.types", "OrchestratorRunResult"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'orchestrator.agent' has no attribute '{name}'")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
