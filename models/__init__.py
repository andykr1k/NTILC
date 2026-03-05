"""NTILC models package with lazy exports."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ToolIntentEmbedder": ("models.intent_embedder", "ToolIntentEmbedder"),
    "ProjectionHead": ("models.projection_head", "ProjectionHead"),
    "ClusterRetrieval": ("models.cluster_retrieval", "ClusterRetrieval"),
    "QueryEncoder": ("models.query_encoder", "QueryEncoder"),
    "ClusterToolMapper": ("models.software_layer", "ClusterToolMapper"),
    "ToolSpec": ("models.software_layer", "ToolSpec"),
    "ToolArgumentSchema": ("models.software_layer", "ToolArgumentSchema"),
    "DispatchCall": ("models.software_layer", "DispatchCall"),
    "DispatchResult": ("models.software_layer", "DispatchResult"),
    "ToolDispatcher": ("models.software_layer", "ToolDispatcher"),
    "build_shell_tool_callable": ("models.software_layer", "build_shell_tool_callable"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'models' has no attribute '{name}'")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
