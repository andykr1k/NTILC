"""
Runtime helpers for direct tool execution without text parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class DispatchResult:
    ok: bool
    tool: str
    arguments: Dict[str, Any]
    result: Any = None
    errors: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "arguments": self.arguments,
            "result": self.result,
            "errors": self.errors or [],
        }


class ToolRegistry:
    """
    Direct callable registry for tool execution.
    This bypasses string parsing and dispatches strongly-typed kwargs directly.
    """

    def __init__(self):
        self._tools: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        if not callable(fn):
            raise TypeError(f"Tool {name} must be callable")
        self._tools[name] = fn

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def dispatch(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> DispatchResult:
        arguments = arguments or {}
        fn = self._tools.get(name)
        if fn is None:
            return DispatchResult(
                ok=False,
                tool=name,
                arguments=arguments,
                result=None,
                errors=[f"Tool not registered: {name}"],
            )
        try:
            result = fn(**arguments)
            return DispatchResult(ok=True, tool=name, arguments=arguments, result=result, errors=[])
        except Exception as exc:
            return DispatchResult(
                ok=False,
                tool=name,
                arguments=arguments,
                result=None,
                errors=[f"Execution failed: {exc}"],
            )

    def dispatch_chain(self, calls: List[Dict[str, Any]]) -> List[DispatchResult]:
        results: List[DispatchResult] = []
        for call in calls:
            tool_name = str(call.get("tool", "")).strip()
            args = call.get("arguments", {})
            if not tool_name:
                results.append(
                    DispatchResult(
                        ok=False,
                        tool="",
                        arguments={},
                        result=None,
                        errors=["Missing tool name"],
                    )
                )
                continue
            if not isinstance(args, dict):
                args = {}
            results.append(self.dispatch(tool_name, args))
        return results
