"""Prompt and normalization helpers for planner and argument generation."""

from __future__ import annotations

import re
import shlex
from typing import Optional, Sequence

from orchestrator.protocol import (
    ACTION_END,
    ACTION_START,
    ARG_END,
    ARG_START,
    DISPATCH_END,
    DISPATCH_START,
    KEY_END,
    KEY_START,
    PLAN_END,
    PLAN_START,
    VALUE_END,
    VALUE_START,
)


def build_plan_prompt(request: str, max_actions: int) -> str:
    req = str(request).strip()
    limit = max(1, int(max_actions))
    return (
        "You are a Linux tool-use planner.\n"
        "Return only protocol tags and payload text.\n"
        "Use this exact schema:\n"
        f"{PLAN_START}\n"
        f"  {ACTION_START}atomic action text{ACTION_END}\n"
        f"{PLAN_END}\n\n"
        "Rules:\n"
        f"- Output at most {limit} actions.\n"
        "- Each action must be atomic and executable by one tool call.\n"
        "- Split combined tasks into multiple actions.\n"
        "- Preserve execution order.\n"
        "- Do not include commentary.\n\n"
        f"User request: {req}\n"
        "Plan:"
    )


def build_dispatch_prompt(
    query: str,
    tool: str,
    mode: str,
    current_action: Optional[str] = None,
    prior_step_summaries: Optional[Sequence[str]] = None,
) -> str:
    history_lines = [f"- {line}" for line in (prior_step_summaries or []) if str(line).strip()]
    history_block = "\n".join(history_lines) if history_lines else "- none"
    action_text = str(current_action).strip() if current_action else str(query).strip()

    command_instruction = (
        "Set key `command` to a full shell command starting with the selected tool."
        if mode != "tail"
        else "Set key `command` to command tail only (arguments/values), without the tool name."
    )

    return (
        "You map an atomic Linux action to structured dispatch arguments.\n"
        "Return only protocol tags and payload text using this exact schema:\n"
        f"{DISPATCH_START}"
        f"{ARG_START}{KEY_START}command{KEY_END}{VALUE_START}<payload>{VALUE_END}{ARG_END}"
        f"{ARG_START}{KEY_START}query{KEY_END}{VALUE_START}<payload>{VALUE_END}{ARG_END}"
        f"{DISPATCH_END}\n"
        "No extra text.\n\n"
        f"{command_instruction}\n"
        "Always include key `query` with the original user request.\n\n"
        f"Original user request: {query}\n"
        f"Current atomic action: {action_text}\n"
        f"Selected tool: {tool}\n"
        f"Prior step summaries:\n{history_block}\n"
        "Dispatch:"
    )


def build_command_prompt(
    query: str,
    tool: str,
    mode: str,
    current_action: Optional[str] = None,
    prior_step_summaries: Optional[Sequence[str]] = None,
) -> str:
    # Backward-compatible alias now routed to dispatch-argument prompting.
    return build_dispatch_prompt(
        query=query,
        tool=tool,
        mode=mode,
        current_action=current_action,
        prior_step_summaries=prior_step_summaries,
    )


def build_full_from_tail(tool: str, tail: str) -> str:
    tail_text = str(tail).strip()
    if not tail_text or tail_text == "<NO_ARGS>":
        return str(tool).strip()
    return f"{str(tool).strip()} {tail_text}".strip()


def safe_first_line(text: str) -> str:
    line = str(text).strip().splitlines()
    return line[0].strip() if line else ""


def normalize_command_for_tool(
    tool: str,
    generated_text: str,
    mode: str,
    enforce_selected_tool: bool = True,
) -> str:
    tool_name = str(tool).strip()
    cleaned = safe_first_line(generated_text)
    if mode == "tail":
        return build_full_from_tail(tool=tool_name, tail=cleaned)

    if not cleaned:
        return tool_name

    if not enforce_selected_tool:
        return cleaned

    try:
        tokens = shlex.split(cleaned)
    except ValueError:
        tokens = cleaned.split()

    if not tokens:
        return tool_name
    if tokens[0] == tool_name:
        return cleaned
    if len(tokens) == 1:
        return f"{tool_name} {tokens[0]}".strip()
    return f"{tool_name} {' '.join(tokens[1:])}".strip()


def extract_plan_block(text: str) -> str:
    raw = str(text or "").strip()
    match = re.search(r"<plan>.*?</plan>", raw, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0).strip()
    return raw
