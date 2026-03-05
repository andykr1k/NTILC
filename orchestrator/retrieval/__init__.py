"""Cluster retrieval orchestration exports."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["ClusterBasedToolSystem"]


def __getattr__(name: str) -> Any:
    if name != "ClusterBasedToolSystem":
        raise AttributeError(f"module 'orchestrator.retrieval' has no attribute '{name}'")
    module = importlib.import_module("orchestrator.retrieval.system")
    value = getattr(module, name)
    globals()[name] = value
    return value
