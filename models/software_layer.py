"""
Software Layer for NTILC: Maps cluster IDs to tools.

This decouples the model from execution. All tool execution logic,
versioning, safety rules, and permissions are handled here.
The model only retrieves the correct cluster ID.
"""

import json
import inspect
import re
import types
from typing import Callable, Dict, List, Optional, Any, Set, Union, get_args, get_origin
from collections import defaultdict

from models.tool_schemas import TOOL_SCHEMAS


class ClusterToToolMapper:
    """
    Maps cluster IDs to one or more tools.
    
    Maintains:
    - Cluster → Tool mappings
    - Versioning
    - Safety rules
    - Permissions
    - Tool indices and arguments
    """
    
    def __init__(
        self,
        cluster_to_tools: Optional[Dict[int, List[str]]] = None,
        tool_versions: Optional[Dict[str, str]] = None,
        safety_rules: Optional[Dict[str, List[str]]] = None
    ):
        """
        Args:
            cluster_to_tools: Mapping from cluster ID to list of tool names
            tool_versions: Mapping from tool name to version string
            safety_rules: Mapping from tool name to list of safety constraints
        """
        self.cluster_to_tools = cluster_to_tools or {}
        self.tool_versions = tool_versions or {}
        self.safety_rules = safety_rules or {}
        self._cluster_cache: Dict[int, Dict[str, Any]] = {}
        self.tool_callables: Dict[str, Callable[..., Any]] = {}
        self.cluster_callables: Dict[int, Callable[..., Any]] = {}
        
        # Build reverse mapping: tool -> clusters
        self.tool_to_clusters = defaultdict(list)
        for cluster_id, tools in self.cluster_to_tools.items():
            for tool in tools:
                self.tool_to_clusters[tool].append(cluster_id)
        self.build_cache()
    
    def add_cluster_mapping(self, cluster_id: int, tools: List[str]):
        """Add or update cluster to tool mapping."""
        self.cluster_to_tools[cluster_id] = tools
        
        # Update reverse mapping
        for tool in tools:
            if cluster_id not in self.tool_to_clusters[tool]:
                self.tool_to_clusters[tool].append(cluster_id)
        self._update_cache_for_cluster(cluster_id)

    def _update_cache_for_cluster(self, cluster_id: int):
        """Update cached info for a single cluster."""
        tools = self.cluster_to_tools.get(cluster_id, [])
        if not tools:
            self._cluster_cache.pop(cluster_id, None)
            return

        tool_name = tools[0]
        tool_schema = TOOL_SCHEMAS.get(tool_name)
        if not tool_schema:
            self._cluster_cache.pop(cluster_id, None)
            return

        required_args = [
            name for name, info in tool_schema["parameters"].items()
            if info.get("required", False)
        ]
        optional_args = [
            name for name, info in tool_schema["parameters"].items()
            if not info.get("required", False)
        ]
        defaults = {
            name: info["default"]
            for name, info in tool_schema["parameters"].items()
            if "default" in info
        }

        self._cluster_cache[cluster_id] = {
            "tool": tool_name,
            "schema": tool_schema,
            "required_args": required_args,
            "optional_args": optional_args,
            "defaults": defaults,
        }

    def build_cache(self):
        """Precompute cluster -> tool metadata for fast resolution."""
        self._cluster_cache = {}
        for cluster_id in self.cluster_to_tools.keys():
            self._update_cache_for_cluster(cluster_id)
    
    def get_tools_for_cluster(self, cluster_id: int) -> List[str]:
        """Get tools associated with a cluster ID."""
        return self.cluster_to_tools.get(cluster_id, [])
    
    def get_clusters_for_tool(self, tool_name: str) -> List[int]:
        """Get clusters associated with a tool."""
        return self.tool_to_clusters.get(tool_name, [])

    def register_tool_callable(self, tool_name: str, fn: Callable[..., Any]):
        """
        Register a callable implementation for a tool name.
        """
        if tool_name not in TOOL_SCHEMAS:
            raise ValueError(f"Cannot register callable for unknown tool: {tool_name}")
        if not callable(fn):
            raise TypeError(f"Registered implementation for {tool_name} must be callable")
        self.tool_callables[tool_name] = fn

    def register_cluster_callable(self, cluster_id: int, fn: Callable[..., Any]):
        """
        Register a callable implementation for a specific cluster ID.
        Takes priority over tool-level callable registrations.
        """
        if cluster_id not in self.cluster_to_tools:
            raise ValueError(f"Cannot register callable for unknown cluster: {cluster_id}")
        if not callable(fn):
            raise TypeError(f"Registered implementation for cluster {cluster_id} must be callable")
        self.cluster_callables[cluster_id] = fn

    def get_callable_for_cluster(
        self,
        cluster_id: int,
        tool_name: Optional[str] = None
    ) -> Optional[Callable[..., Any]]:
        """
        Resolve callable for cluster:
        1) explicit cluster callable
        2) tool-level callable for resolved tool
        """
        if cluster_id in self.cluster_callables:
            return self.cluster_callables[cluster_id]

        if tool_name is None:
            tools = self.cluster_to_tools.get(cluster_id, [])
            tool_name = tools[0] if tools else None
        if tool_name is None:
            return None

        return self.tool_callables.get(tool_name)
    
    def resolve_cluster(
        self,
        cluster_id: int,
        similarity_score: float = 1.0,
        threshold: float = 0.5
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve cluster ID to tool with safety checks.
        
        Args:
            cluster_id: Cluster ID from retrieval
            similarity_score: Confidence score
            threshold: Minimum similarity threshold
        
        Returns:
            Dictionary with tool information or None if below threshold
        """
        if similarity_score < threshold:
            return None
        
        if cluster_id not in self.cluster_to_tools:
            return None
        
        tools = self.cluster_to_tools[cluster_id]
        
        # For now, return first tool (can be extended for multi-tool plans)
        if len(tools) == 0:
            return None
        
        tool_name = tools[0]
        
        # Get tool schema
        tool_schema = TOOL_SCHEMAS.get(tool_name)
        if not tool_schema:
            return None
        
        return {
            "tool": tool_name,
            "version": self.tool_versions.get(tool_name, "1.0.0"),
            "schema": tool_schema,
            "safety_rules": self.safety_rules.get(tool_name, []),
            "confidence": similarity_score,
            "cluster_id": cluster_id
        }

    def resolve_cluster_fast(
        self,
        cluster_id: int,
        similarity_score: float = 1.0,
        threshold: float = 0.5
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve cluster using cached metadata for speed.
        """
        if similarity_score < threshold:
            return None

        cached = self._cluster_cache.get(cluster_id)
        if not cached:
            return None

        tool_name = cached["tool"]
        return {
            "tool": tool_name,
            "version": self.tool_versions.get(tool_name, "1.0.0"),
            "schema": cached["schema"],
            "safety_rules": self.safety_rules.get(tool_name, []),
            "confidence": similarity_score,
            "cluster_id": cluster_id,
            "required_args": cached["required_args"],
            "optional_args": cached["optional_args"],
            "defaults": cached["defaults"],
        }
    
    def resolve_multiple_clusters(
        self,
        cluster_ids: List[int],
        similarities: List[float],
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Resolve multiple cluster IDs (for multi-tool scenarios).
        
        Args:
            cluster_ids: List of cluster IDs
            similarities: List of similarity scores
            threshold: Minimum similarity threshold
        
        Returns:
            List of resolved tool dictionaries
        """
        results = []
        for cluster_id, similarity in zip(cluster_ids, similarities):
            resolved = self.resolve_cluster(cluster_id, similarity, threshold)
            if resolved:
                results.append(resolved)
        
        return results
    
    def validate_tool_execution(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_permissions: Optional[Set[str]] = None,
        callable_obj: Optional[Callable[..., Any]] = None,
        strict: bool = True
    ) -> tuple[bool, List[str]]:
        """
        Validate tool execution with safety rules and permissions.
        
        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments
            user_permissions: Set of user permissions
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check tool exists
        if tool_name not in TOOL_SCHEMAS:
            errors.append(f"Unknown tool: {tool_name}")
            return False, errors
        
        # Check permissions
        if user_permissions is not None:
            # For now, all tools are allowed (can add permission checks)
            pass
        
        # Check safety rules
        safety_rules = self.safety_rules.get(tool_name, [])
        for rule in safety_rules:
            # Apply safety rule (can be extended)
            pass

        # Validate arguments against schema
        schema = TOOL_SCHEMAS[tool_name]
        normalized_args = self._apply_schema_defaults(schema, arguments)
        required_args = [
            name for name, info in schema["parameters"].items()
            if info.get("required", False)
        ]

        for arg in required_args:
            if arg not in normalized_args:
                errors.append(f"Missing required argument: {arg}")
            elif strict and normalized_args[arg] is None:
                errors.append(f"Required argument {arg} cannot be None")

        for arg_name, arg_value in normalized_args.items():
            if arg_name not in schema["parameters"]:
                errors.append(f"Unknown argument: {arg_name}")
                continue

            param_info = schema["parameters"][arg_name]
            expected_type = param_info.get("type", "str")
            if strict and not self._is_valid_schema_type(arg_value, expected_type, param_info):
                errors.append(
                    f"Argument {arg_name} should be {expected_type}; got {type(arg_value).__name__}"
                )

            options = param_info.get("options")
            if options and arg_value not in options:
                errors.append(f"Argument {arg_name} must be one of {options}")

        if callable_obj is not None and strict:
            errors.extend(self._validate_callable_signature(callable_obj, normalized_args))
        
        is_valid = len(errors) == 0
        return is_valid, errors

    def dispatch_cluster(
        self,
        cluster_id: int,
        arguments: Dict[str, Any],
        similarity_score: float = 1.0,
        threshold: float = 0.5,
        user_permissions: Optional[Set[str]] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Resolve cluster -> callable, validate args, then execute.
        """
        resolved = self.resolve_cluster_fast(
            cluster_id=cluster_id,
            similarity_score=similarity_score,
            threshold=threshold,
        )
        if not resolved:
            return {
                "ok": False,
                "cluster_id": cluster_id,
                "tool": None,
                "result": None,
                "errors": [f"Cluster {cluster_id} could not be resolved"],
            }

        tool_name = resolved["tool"]
        callable_obj = self.get_callable_for_cluster(cluster_id=cluster_id, tool_name=tool_name)
        if callable_obj is None:
            return {
                "ok": False,
                "cluster_id": cluster_id,
                "tool": tool_name,
                "result": None,
                "errors": [f"No callable registered for cluster {cluster_id} / tool {tool_name}"],
            }

        normalized_args = self._apply_schema_defaults(TOOL_SCHEMAS[tool_name], arguments)
        is_valid, errors = self.validate_tool_execution(
            tool_name=tool_name,
            arguments=normalized_args,
            user_permissions=user_permissions,
            callable_obj=callable_obj,
            strict=strict,
        )
        if not is_valid:
            return {
                "ok": False,
                "cluster_id": cluster_id,
                "tool": tool_name,
                "result": None,
                "errors": errors,
            }

        try:
            result = callable_obj(**normalized_args)
        except Exception as exc:
            return {
                "ok": False,
                "cluster_id": cluster_id,
                "tool": tool_name,
                "result": None,
                "errors": [f"Callable execution failed: {exc}"],
            }

        return {
            "ok": True,
            "cluster_id": cluster_id,
            "tool": tool_name,
            "result": result,
            "arguments": normalized_args,
            "errors": [],
        }

    @staticmethod
    def _apply_schema_defaults(schema: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(arguments)
        for arg_name, param_info in schema.get("parameters", {}).items():
            if arg_name not in normalized and "default" in param_info:
                normalized[arg_name] = param_info["default"]
        return normalized

    @staticmethod
    def _is_valid_email(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value) is not None

    def _is_valid_schema_type(
        self,
        value: Any,
        expected_type: str,
        param_info: Dict[str, Any]
    ) -> bool:
        if expected_type == "any":
            return True
        if expected_type == "str":
            return isinstance(value, str)
        if expected_type == "int":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected_type == "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected_type == "bool":
            return isinstance(value, bool)
        if expected_type == "enum":
            options = param_info.get("options", [])
            return (not options) or (value in options)
        if expected_type == "email":
            return self._is_valid_email(value)
        if expected_type == "List[email]":
            return isinstance(value, list) and all(self._is_valid_email(v) for v in value)
        if expected_type == "DateRange":
            return (
                isinstance(value, dict) and
                isinstance(value.get("from_date"), str) and
                isinstance(value.get("to_date"), str)
            )
        return True

    def _validate_callable_signature(
        self,
        callable_obj: Callable[..., Any],
        arguments: Dict[str, Any]
    ) -> List[str]:
        errors: List[str] = []
        signature = inspect.signature(callable_obj)

        try:
            signature.bind(**arguments)
        except TypeError as exc:
            errors.append(f"Callable signature mismatch: {exc}")
            return errors

        for param_name, param in signature.parameters.items():
            if param.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue

            if param.default is inspect._empty and param_name not in arguments:
                errors.append(f"Callable missing required argument: {param_name}")

        for arg_name, arg_value in arguments.items():
            param = signature.parameters.get(arg_name)
            if not param:
                continue
            annotation = param.annotation
            if annotation is inspect._empty:
                continue
            if not self._matches_annotation(arg_value, annotation):
                ann_name = getattr(annotation, "__name__", str(annotation))
                errors.append(
                    f"Callable argument {arg_name} should match annotation {ann_name}"
                )

        return errors

    def _matches_annotation(self, value: Any, annotation: Any) -> bool:
        if annotation is Any:
            return True

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is None:
            if annotation is int:
                return isinstance(value, int) and not isinstance(value, bool)
            if annotation is float:
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            if annotation is bool:
                return isinstance(value, bool)
            if annotation is str:
                return isinstance(value, str)
            try:
                return isinstance(value, annotation)
            except TypeError:
                return True

        if origin in (list, List):
            if not isinstance(value, list):
                return False
            if not args:
                return True
            return all(self._matches_annotation(v, args[0]) for v in value)

        if origin in (dict, Dict):
            if not isinstance(value, dict):
                return False
            if len(args) != 2:
                return True
            key_type, value_type = args
            return all(
                self._matches_annotation(k, key_type) and self._matches_annotation(v, value_type)
                for k, v in value.items()
            )

        if origin in (Union, types.UnionType):
            return any(self._matches_annotation(value, arg) for arg in args)

        return True
    
    def initialize_from_tools(self, tools: List[str] = None):
        """
        Initialize cluster mappings from tool list.
        
        For now, creates one cluster per tool (1:1 mapping).
        Can be extended to learn more sophisticated mappings.
        """
        if tools is None:
            tools = list(TOOL_SCHEMAS.keys())
        
        # Simple 1:1 mapping: cluster_id = tool_index
        for idx, tool_name in enumerate(tools):
            self.add_cluster_mapping(idx, [tool_name])
        self.build_cache()
    
    def save(self, filepath: str):
        """Save mappings to file."""
        data = {
            "cluster_to_tools": self.cluster_to_tools,
            "tool_versions": self.tool_versions,
            "safety_rules": self.safety_rules
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load mappings from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.cluster_to_tools = {int(k): v for k, v in data.get("cluster_to_tools", {}).items()}
        self.tool_versions = data.get("tool_versions", {})
        self.safety_rules = data.get("safety_rules", {})
        
        # Rebuild reverse mapping
        self.tool_to_clusters = defaultdict(list)
        for cluster_id, tools in self.cluster_to_tools.items():
            for tool in tools:
                self.tool_to_clusters[tool].append(cluster_id)
        self.build_cache()
