"""
Utility functions for parsing and validating tool calls.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from ablation.tool_schemas import TOOL_SCHEMAS


def parse_tool_call(tool_call_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse a tool call JSON string into structured format.

    Args:
        tool_call_str: JSON string representing tool call

    Returns:
        Parsed tool call dict or None if invalid
    """
    try:
        data = json.loads(tool_call_str)
        if not isinstance(data, dict):
            return None
        if "tool" not in data or "arguments" not in data:
            return None
        return data
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def extract_tool(tool_call: Dict[str, Any]) -> Optional[str]:
    """Extract tool name from parsed tool call."""
    return tool_call.get("tool") if tool_call else None


def extract_arguments(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Extract arguments from parsed tool call."""
    return tool_call.get("arguments", {}) if tool_call else {}


def validate_tool_call(tool_call: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a tool call against its schema.

    Args:
        tool_call: Parsed tool call dict

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    # Check tool exists
    tool_name = tool_call.get("tool")
    if not tool_name:
        errors.append("Missing 'tool' field")
        return False, errors

    if tool_name not in TOOL_SCHEMAS:
        errors.append(f"Unknown tool: {tool_name}")
        return False, errors

    schema = TOOL_SCHEMAS[tool_name]
    arguments = tool_call.get("arguments", {})

    # Check required arguments
    for param_name, param_info in schema["parameters"].items():
        if param_info.get("required", False):
            if param_name not in arguments:
                errors.append(f"Missing required argument: {param_name}")

    # Check argument types
    for arg_name, arg_value in arguments.items():
        if arg_name not in schema["parameters"]:
            errors.append(f"Unknown argument: {arg_name}")
            continue

        param_info = schema["parameters"][arg_name]
        expected_type = param_info.get("type", "str")

        # Type checking
        if expected_type == "int" and not isinstance(arg_value, int):
            errors.append(
                f"Argument {arg_name} should be int, got {type(arg_value).__name__}")
        elif expected_type == "float" and not isinstance(arg_value, (int, float)):
            errors.append(
                f"Argument {arg_name} should be float, got {type(arg_value).__name__}")
        elif expected_type == "bool" and not isinstance(arg_value, bool):
            errors.append(
                f"Argument {arg_name} should be bool, got {type(arg_value).__name__}")
        elif expected_type == "enum":
            options = param_info.get("options", [])
            if options and arg_value not in options:
                errors.append(
                    f"Argument {arg_name} must be one of {options}, got {arg_value}")
        elif expected_type == "List[email]":
            if not isinstance(arg_value, list):
                errors.append(
                    f"Argument {arg_name} should be a list, got {type(arg_value).__name__}")
            # Could add email validation here

    is_valid = len(errors) == 0
    return is_valid, errors


def get_tool_schema(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get schema for a tool."""
    return TOOL_SCHEMAS.get(tool_name)


def get_required_args(tool_name: str) -> List[str]:
    """Get list of required arguments for a tool."""
    schema = TOOL_SCHEMAS.get(tool_name)
    if not schema:
        return []

    return [
        name for name, info in schema["parameters"].items()
        if info.get("required", False)
    ]


def get_optional_args(tool_name: str) -> List[str]:
    """Get list of optional arguments for a tool."""
    schema = TOOL_SCHEMAS.get(tool_name)
    if not schema:
        return []

    return [
        name for name, info in schema["parameters"].items()
        if not info.get("required", False)
    ]


def get_arg_type(tool_name: str, arg_name: str) -> Optional[str]:
    """Get type of an argument for a tool."""
    schema = TOOL_SCHEMAS.get(tool_name)
    if not schema:
        return None

    param_info = schema["parameters"].get(arg_name)
    if not param_info:
        return None

    return param_info.get("type", "str")


def get_arg_constraints(tool_name: str, arg_name: str) -> Dict[str, Any]:
    """Get constraints for an argument."""
    schema = TOOL_SCHEMAS.get(tool_name)
    if not schema:
        return {}

    param_info = schema["parameters"].get(arg_name)
    if not param_info:
        return {}

    constraints = {}
    if "options" in param_info:
        constraints["enum"] = param_info["options"]
    if "default" in param_info:
        constraints["default"] = param_info["default"]

    return constraints


def repair_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attempt to repair a tool call by adding defaults and fixing types.

    Args:
        tool_call: Parsed tool call dict

    Returns:
        Repaired tool call dict
    """
    tool_name = tool_call.get("tool")
    if not tool_name or tool_name not in TOOL_SCHEMAS:
        return tool_call

    schema = TOOL_SCHEMAS[tool_name]
    arguments = tool_call.get("arguments", {}).copy()

    # Add default values for missing optional args
    for param_name, param_info in schema["parameters"].items():
        if param_name not in arguments:
            if "default" in param_info:
                arguments[param_name] = param_info["default"]

    # Fix type mismatches where possible
    for arg_name, arg_value in arguments.items():
        if arg_name not in schema["parameters"]:
            continue

        param_info = schema["parameters"][arg_name]
        expected_type = param_info.get("type", "str")

        # Try to convert types
        if expected_type == "int" and isinstance(arg_value, (float, str)):
            try:
                arguments[arg_name] = int(float(arg_value))
            except (ValueError, TypeError):
                pass
        elif expected_type == "float" and isinstance(arg_value, (int, str)):
            try:
                arguments[arg_name] = float(arg_value)
            except (ValueError, TypeError):
                pass
        elif expected_type == "bool" and isinstance(arg_value, str):
            arguments[arg_name] = arg_value.lower() in ("true", "1", "yes")

    return {
        "tool": tool_name,
        "arguments": arguments
    }
