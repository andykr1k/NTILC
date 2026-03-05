"""Generation component exports."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "QwenOrchestratorModel": ("orchestrator.generation.model", "QwenOrchestratorModel"),
    "build_plan_prompt": ("orchestrator.generation.prompting", "build_plan_prompt"),
    "build_dispatch_prompt": ("orchestrator.generation.prompting", "build_dispatch_prompt"),
    "build_command_prompt": ("orchestrator.generation.prompting", "build_command_prompt"),
    "build_full_from_tail": ("orchestrator.generation.prompting", "build_full_from_tail"),
    "safe_first_line": ("orchestrator.generation.prompting", "safe_first_line"),
    "normalize_command_for_tool": ("orchestrator.generation.prompting", "normalize_command_for_tool"),
    "extract_plan_block": ("orchestrator.generation.prompting", "extract_plan_block"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'orchestrator.generation' has no attribute '{name}'")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
