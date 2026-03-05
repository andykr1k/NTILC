"""Protocol block builders used by the NTILC orchestrator runtime."""

from __future__ import annotations

from html import escape as xml_escape
from typing import Any, Mapping, Sequence

from models.software_layer import DispatchResult


def build_len_tag(tag: str, value: Any) -> str:
    text = str(value)
    return f"<{tag}>{xml_escape(text)}</{tag}>"


def build_plan_block(actions: Sequence[str]) -> str:
    lines = ["<plan>"]
    for action in actions:
        text = str(action).strip()
        if not text:
            continue
        lines.append(f"  <action>{xml_escape(text)}</action>")
    lines.append("</plan>")
    return "\n".join(lines)


def build_dispatch_block(tool: str, arguments: Mapping[str, Any]) -> str:
    lines = ["<dispatch>", f"  {build_len_tag('tool', tool)}"]
    for name, value in arguments.items():
        key = xml_escape(str(name))
        arg_text = xml_escape(str(value))
        lines.append(
            f"  <arg><key>{key}</key><value>{arg_text}</value></arg>"
        )
    lines.append("</dispatch>")
    return "\n".join(lines)


def build_response_block(tool: str, dispatch_result: DispatchResult, retry: bool) -> str:
    if dispatch_result.ok:
        text = "ok"
        if isinstance(dispatch_result.result, Mapping):
            returncode = dispatch_result.result.get("returncode")
            stdout = str(dispatch_result.result.get("stdout", "")).strip()
            if returncode is not None:
                text = f"returncode={returncode}"
            if stdout:
                text = f"{text}; stdout={stdout[:200]}"
    else:
        text = "; ".join(dispatch_result.errors) if dispatch_result.errors else "dispatch failed"

    lines = [
        "<response>",
        f"  {build_len_tag('tool', tool)}",
        f"  {build_len_tag('status', 'ok' if dispatch_result.ok else 'fail')}",
        f"  {build_len_tag('text', text)}",
        f"  {build_len_tag('retry', 'true' if retry else 'false')}",
        "</response>",
    ]
    return "\n".join(lines)
