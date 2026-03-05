"""
Software layer for cluster-to-tool mapping and validated dispatch.

This module provides:
1) Cluster ID -> tool-name mapping
2) Tool registration with schemas and safety rules
3) Dispatcher that validates and executes tool callables
"""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


TypeSpec = Union[type, Tuple[type, ...]]


def _readable_type_name(expected: TypeSpec) -> str:
    if isinstance(expected, tuple):
        return " | ".join(t.__name__ for t in expected)
    return expected.__name__


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


@dataclass
class ToolArgumentSchema:
    """
    Basic argument schema for a tool callable.
    """

    required: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)
    types: Dict[str, TypeSpec] = field(default_factory=dict)
    allow_unknown: bool = True

    def validate(self, arguments: Mapping[str, Any]) -> List[str]:
        errors: List[str] = []
        if not isinstance(arguments, Mapping):
            return ["Arguments must be a mapping/dict."]

        missing = [name for name in self.required if name not in arguments]
        if missing:
            errors.append(f"Missing required arguments: {', '.join(missing)}")

        if not self.allow_unknown:
            allowed = set(self.required) | set(self.optional) | set(self.types.keys())
            unknown = [name for name in arguments.keys() if name not in allowed]
            if unknown:
                errors.append(f"Unknown arguments: {', '.join(unknown)}")

        for key, expected_type in self.types.items():
            if key not in arguments:
                continue
            value = arguments[key]
            if value is None:
                continue
            if not isinstance(value, expected_type):
                errors.append(
                    f"Argument `{key}` has type {type(value).__name__}, "
                    f"expected {_readable_type_name(expected_type)}."
                )

        return errors


@dataclass
class ToolSpec:
    """
    Registered tool metadata and callable.
    """

    name: str
    executor: Optional[Callable[..., Any]] = None
    argument_schema: ToolArgumentSchema = field(default_factory=ToolArgumentSchema)
    safety_rules: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class DispatchCall:
    """
    Tool call request for batch dispatch.
    """

    tool: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    cluster_id: Optional[int] = None


@dataclass
class DispatchResult:
    """
    Dispatch outcome.
    """

    ok: bool
    tool: str
    arguments: Dict[str, Any]
    cluster_id: Optional[int] = None
    result: Any = None
    errors: List[str] = field(default_factory=list)
    executed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "cluster_id": self.cluster_id,
            "arguments": self.arguments,
            "result": self.result,
            "errors": self.errors,
            "executed": self.executed,
        }


class ClusterToolMapper:
    """
    Maintains cluster/tool mapping plus tool registry.
    """

    def __init__(self, cluster_to_tool: Optional[Mapping[int, str]] = None):
        self._cluster_to_tool: Dict[int, str] = {}
        self._tool_specs: Dict[str, ToolSpec] = {}

        if cluster_to_tool:
            for cluster_id, tool_name in cluster_to_tool.items():
                self.set_cluster_mapping(cluster_id, tool_name)

    @classmethod
    def from_tool_names(cls, tool_names: Sequence[str]) -> "ClusterToolMapper":
        mapping = {idx: str(name).strip() for idx, name in enumerate(tool_names)}
        return cls(cluster_to_tool=mapping)

    @classmethod
    def from_retrieval_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        map_location: str = "cpu",
    ) -> "ClusterToolMapper":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("`torch` is required to load retrieval checkpoints.") from exc

        payload = torch.load(str(checkpoint_path), map_location=map_location, weights_only=False)
        tool_names = payload.get("tool_names")
        if not isinstance(tool_names, list) or not tool_names:
            tool_to_idx = payload.get("tool_to_idx", {})
            if isinstance(tool_to_idx, dict) and tool_to_idx:
                ordered = sorted(
                    ((int(idx), str(tool)) for tool, idx in tool_to_idx.items()),
                    key=lambda x: x[0],
                )
                tool_names = [tool for _, tool in ordered]
            else:
                raise ValueError(
                    f"Checkpoint at {checkpoint_path} has no usable `tool_names` or `tool_to_idx`."
                )
        return cls.from_tool_names(tool_names)

    def set_cluster_mapping(self, cluster_id: int, tool_name: str) -> None:
        cid = int(cluster_id)
        name = str(tool_name).strip()
        if not name:
            raise ValueError("tool_name must be non-empty.")
        self._cluster_to_tool[cid] = name
        if name not in self._tool_specs:
            self._tool_specs[name] = ToolSpec(name=name)

    def cluster_to_tool(self, cluster_id: int) -> str:
        cid = int(cluster_id)
        if cid not in self._cluster_to_tool:
            raise KeyError(f"No tool mapping found for cluster_id={cid}.")
        return self._cluster_to_tool[cid]

    def register_tool(
        self,
        tool_name: str,
        executor: Optional[Callable[..., Any]] = None,
        argument_schema: Optional[ToolArgumentSchema] = None,
        safety_rules: Optional[Sequence[str]] = None,
        description: Optional[str] = None,
    ) -> ToolSpec:
        name = str(tool_name).strip()
        if not name:
            raise ValueError("tool_name must be non-empty.")

        spec = self._tool_specs.get(name, ToolSpec(name=name))
        if executor is not None:
            if not callable(executor):
                raise TypeError(f"Executor for `{name}` must be callable.")
            spec.executor = executor
        if argument_schema is not None:
            spec.argument_schema = argument_schema
        if safety_rules is not None:
            spec.safety_rules = [str(rule).strip() for rule in safety_rules if str(rule).strip()]
        if description is not None:
            spec.description = str(description).strip()

        self._tool_specs[name] = spec
        return spec

    def register_shell_tool(
        self,
        tool_name: str,
        timeout_seconds: int = 20,
        cwd: Optional[Union[str, Path]] = None,
        extra_safety_rules: Optional[Sequence[str]] = None,
    ) -> ToolSpec:
        rules = ["non_empty:command"]
        if extra_safety_rules:
            rules.extend(str(rule).strip() for rule in extra_safety_rules if str(rule).strip())

        schema = ToolArgumentSchema(
            required=["command"],
            optional=["query"],
            types={"command": str, "query": str},
            allow_unknown=True,
        )
        executor = build_shell_tool_callable(
            tool_name=tool_name,
            timeout_seconds=timeout_seconds,
            cwd=cwd,
        )
        return self.register_tool(
            tool_name=tool_name,
            executor=executor,
            argument_schema=schema,
            safety_rules=rules,
            description="Shell tool executor",
        )

    def register_shell_tools_for_all_clusters(
        self,
        timeout_seconds: int = 20,
        cwd: Optional[Union[str, Path]] = None,
    ) -> None:
        for tool_name in sorted(set(self._cluster_to_tool.values())):
            self.register_shell_tool(tool_name=tool_name, timeout_seconds=timeout_seconds, cwd=cwd)

    def get_tool_spec(self, tool_name: str) -> Optional[ToolSpec]:
        return self._tool_specs.get(str(tool_name).strip())

    def known_tools(self) -> List[str]:
        return sorted(self._tool_specs.keys())

    def clusters(self) -> List[int]:
        return sorted(self._cluster_to_tool.keys())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_to_tool": dict(sorted(self._cluster_to_tool.items(), key=lambda kv: kv[0])),
            "tools": {
                name: {
                    "description": spec.description,
                    "required_args": list(spec.argument_schema.required),
                    "optional_args": list(spec.argument_schema.optional),
                    "type_args": {
                        key: _readable_type_name(value)
                        for key, value in spec.argument_schema.types.items()
                    },
                    "allow_unknown_args": spec.argument_schema.allow_unknown,
                    "safety_rules": list(spec.safety_rules),
                    "has_executor": spec.executor is not None,
                }
                for name, spec in sorted(self._tool_specs.items(), key=lambda kv: kv[0])
            },
        }


class ToolDispatcher:
    """
    Validated dispatcher for registered tools.
    """

    def __init__(
        self,
        mapper: ClusterToolMapper,
        fail_on_nonzero_exit: bool = True,
    ):
        self.mapper = mapper
        self.fail_on_nonzero_exit = fail_on_nonzero_exit

    def dispatch(
        self,
        tool: str,
        arguments: Optional[Dict[str, Any]] = None,
        execute: bool = True,
        granted_permissions: Optional[Iterable[str]] = None,
        cluster_id: Optional[int] = None,
    ) -> DispatchResult:
        tool_name = str(tool).strip()
        args = arguments or {}
        if not tool_name:
            return DispatchResult(
                ok=False,
                tool=tool_name,
                arguments=args,
                cluster_id=cluster_id,
                errors=["Missing tool name."],
            )

        spec = self.mapper.get_tool_spec(tool_name)
        if spec is None:
            return DispatchResult(
                ok=False,
                tool=tool_name,
                arguments=args,
                cluster_id=cluster_id,
                errors=[f"Tool not registered: {tool_name}"],
            )

        validation_errors = []
        validation_errors.extend(spec.argument_schema.validate(args))
        validation_errors.extend(
            self._validate_safety_rules(
                tool_name=tool_name,
                arguments=args,
                safety_rules=spec.safety_rules,
                granted_permissions=granted_permissions,
            )
        )
        if validation_errors:
            return DispatchResult(
                ok=False,
                tool=tool_name,
                arguments=args,
                cluster_id=cluster_id,
                errors=validation_errors,
            )

        if not execute:
            return DispatchResult(
                ok=True,
                tool=tool_name,
                arguments=args,
                cluster_id=cluster_id,
                executed=False,
                result=None,
                errors=[],
            )

        if spec.executor is None:
            return DispatchResult(
                ok=False,
                tool=tool_name,
                arguments=args,
                cluster_id=cluster_id,
                executed=False,
                errors=[f"Tool has no executor: {tool_name}"],
            )

        try:
            result = spec.executor(**args)
        except Exception as exc:
            return DispatchResult(
                ok=False,
                tool=tool_name,
                arguments=args,
                cluster_id=cluster_id,
                executed=True,
                errors=[f"Execution failed: {exc}"],
            )

        if self.fail_on_nonzero_exit and isinstance(result, Mapping):
            return_code = result.get("returncode")
            if isinstance(return_code, int) and return_code != 0:
                stderr = str(result.get("stderr", "")).strip()
                msg = f"Tool exited with non-zero code: {return_code}"
                if stderr:
                    msg = f"{msg} | stderr={stderr}"
                return DispatchResult(
                    ok=False,
                    tool=tool_name,
                    arguments=args,
                    cluster_id=cluster_id,
                    executed=True,
                    result=result,
                    errors=[msg],
                )

        return DispatchResult(
            ok=True,
            tool=tool_name,
            arguments=args,
            cluster_id=cluster_id,
            executed=True,
            result=result,
            errors=[],
        )

    def dispatch_cluster(
        self,
        cluster_id: int,
        arguments: Optional[Dict[str, Any]] = None,
        execute: bool = True,
        granted_permissions: Optional[Iterable[str]] = None,
    ) -> DispatchResult:
        try:
            tool_name = self.mapper.cluster_to_tool(cluster_id)
        except KeyError as exc:
            return DispatchResult(
                ok=False,
                tool="",
                arguments=arguments or {},
                cluster_id=int(cluster_id),
                errors=[str(exc)],
            )
        return self.dispatch(
            tool=tool_name,
            arguments=arguments or {},
            execute=execute,
            granted_permissions=granted_permissions,
            cluster_id=int(cluster_id),
        )

    def dispatch_chain(
        self,
        calls: Sequence[Union[DispatchCall, Mapping[str, Any]]],
        execute: bool = True,
        granted_permissions: Optional[Iterable[str]] = None,
    ) -> List[DispatchResult]:
        results: List[DispatchResult] = []
        for raw_call in calls:
            call = self._normalize_call(raw_call)
            if call.cluster_id is not None:
                result = self.dispatch_cluster(
                    cluster_id=call.cluster_id,
                    arguments=call.arguments,
                    execute=execute,
                    granted_permissions=granted_permissions,
                )
            else:
                result = self.dispatch(
                    tool=call.tool,
                    arguments=call.arguments,
                    execute=execute,
                    granted_permissions=granted_permissions,
                    cluster_id=None,
                )
            results.append(result)
        return results

    def _normalize_call(self, raw_call: Union[DispatchCall, Mapping[str, Any]]) -> DispatchCall:
        if isinstance(raw_call, DispatchCall):
            return raw_call
        if not isinstance(raw_call, Mapping):
            return DispatchCall(tool="", arguments={}, cluster_id=None)

        cluster_id: Optional[int] = None
        if raw_call.get("cluster_id") is not None:
            try:
                cluster_id = int(raw_call["cluster_id"])
            except (TypeError, ValueError):
                cluster_id = None

        tool = str(raw_call.get("tool", "")).strip()
        arguments = raw_call.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        return DispatchCall(tool=tool, arguments=arguments, cluster_id=cluster_id)

    def _validate_safety_rules(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        safety_rules: Sequence[str],
        granted_permissions: Optional[Iterable[str]],
    ) -> List[str]:
        errors: List[str] = []
        permission_set = {str(p).strip() for p in (granted_permissions or []) if str(p).strip()}

        for raw_rule in safety_rules:
            rule = str(raw_rule).strip()
            if not rule:
                continue

            if rule.startswith("requires_permission:"):
                permission = rule.split(":", 1)[1].strip()
                if permission and permission not in permission_set:
                    errors.append(
                        f"Tool `{tool_name}` requires permission `{permission}`."
                    )
                continue

            if rule.startswith("forbid_arg:"):
                arg_name = rule.split(":", 1)[1].strip()
                if arg_name and arg_name in arguments:
                    errors.append(
                        f"Argument `{arg_name}` is forbidden by safety policy for `{tool_name}`."
                    )
                continue

            if rule.startswith("forbid_value:"):
                payload = rule.split(":", 1)[1].strip()
                if "=" not in payload:
                    errors.append(f"Invalid safety rule format: `{rule}`")
                    continue
                key, expected_value = payload.split("=", 1)
                key = key.strip()
                expected_value = expected_value.strip()
                if key in arguments and str(arguments.get(key)) == expected_value:
                    errors.append(
                        f"Argument `{key}` has forbidden value `{expected_value}` for `{tool_name}`."
                    )
                continue

            if rule.startswith("non_empty:"):
                arg_name = rule.split(":", 1)[1].strip()
                if arg_name and _is_empty_value(arguments.get(arg_name)):
                    errors.append(
                        f"Argument `{arg_name}` must be non-empty for `{tool_name}`."
                    )
                continue

            if rule.startswith("regex:"):
                payload = rule.split(":", 1)[1]
                if ":" not in payload:
                    errors.append(f"Invalid safety rule format: `{rule}`")
                    continue
                arg_name, pattern = payload.split(":", 1)
                arg_name = arg_name.strip()
                pattern = pattern.strip()
                value = str(arguments.get(arg_name, ""))
                if not re.match(pattern, value):
                    errors.append(
                        f"Argument `{arg_name}` does not satisfy pattern `{pattern}` for `{tool_name}`."
                    )
                continue

            errors.append(f"Unknown safety rule: `{rule}`")

        return errors


def build_shell_tool_callable(
    tool_name: str,
    timeout_seconds: int = 20,
    cwd: Optional[Union[str, Path]] = None,
) -> Callable[..., Dict[str, Any]]:
    """
    Build an executor that runs a shell tool with subprocess (no shell=True).

    Expected arguments:
    - command (required in default schema): full command string that starts with tool_name
    """

    normalized_tool = str(tool_name).strip()
    if not normalized_tool:
        raise ValueError("tool_name must be non-empty.")

    run_cwd = str(Path(cwd)) if cwd is not None else None

    def _executor(
        command: str,
        **_: Any,
    ) -> Dict[str, Any]:
        command_text = str(command).strip()
        if not command_text:
            raise ValueError("`command` must be non-empty.")

        cmd_tokens = shlex.split(command_text)
        if not cmd_tokens:
            raise ValueError("Command is empty after tokenization.")
        if cmd_tokens[0] != normalized_tool:
            raise ValueError(
                f"Command tool mismatch: expected `{normalized_tool}`, got `{cmd_tokens[0]}`."
            )

        try:
            completed = subprocess.run(
                cmd_tokens,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=run_cwd,
                check=False,
            )
            return {
                "command": " ".join(shlex.quote(tok) for tok in cmd_tokens),
                "returncode": int(completed.returncode),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return {
                "command": " ".join(shlex.quote(tok) for tok in cmd_tokens),
                "returncode": 124,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": True,
            }

    return _executor

