"""
Validation and fallback splitting for atomic plan actions.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from .parser import PlanAction


_SPLIT_SENTENCE_RE = re.compile(r"\s*(?:;|\bthen\b|\band then\b)\s*", flags=re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")


def _starts_like_command(text: str) -> bool:
    verbs = {
        "make",
        "create",
        "move",
        "copy",
        "remove",
        "delete",
        "list",
        "read",
        "write",
        "find",
        "search",
        "run",
        "execute",
        "open",
        "send",
        "fetch",
        "download",
        "rename",
        "mkdir",
        "mv",
        "cp",
        "rm",
        "grep",
        "cat",
    }
    words = text.strip().split()
    return bool(words) and words[0].lower() in verbs


def _split_on_and_if_needed(text: str) -> List[str]:
    candidate = str(text).strip()
    if " and " not in candidate.lower():
        return [candidate] if candidate else []

    parts = re.split(r"\s+and\s+", candidate, flags=re.IGNORECASE)
    cleaned = [part.strip(" ,") for part in parts if part.strip(" ,")]
    if len(cleaned) < 2:
        return [candidate] if candidate else []

    command_like = sum(1 for chunk in cleaned if _starts_like_command(chunk))
    if command_like >= 2:
        return cleaned
    return [candidate]


def is_atomic_action(instruction: str) -> bool:
    text = str(instruction).strip()
    if not text:
        return False

    if ";" in text:
        return False
    if re.search(r"\bthen\b", text, flags=re.IGNORECASE):
        return False

    and_split = _split_on_and_if_needed(text)
    return len(and_split) == 1


def split_non_atomic_action(instruction: str) -> List[str]:
    text = str(instruction).strip()
    if not text:
        return []

    stage_one = [piece.strip(" ,") for piece in _SPLIT_SENTENCE_RE.split(text) if piece.strip(" ,")]
    if not stage_one:
        return []

    pieces: List[str] = []
    for piece in stage_one:
        pieces.extend(_split_on_and_if_needed(piece))

    return [piece for piece in pieces if piece]


def enforce_atomic_actions(actions: Sequence[PlanAction]) -> List[PlanAction]:
    flattened: List[PlanAction] = []
    next_id = 1
    for action in actions:
        instruction = str(action.instruction).strip()
        if not instruction:
            continue
        if is_atomic_action(instruction):
            flattened.append(PlanAction(id=next_id, instruction=instruction))
            next_id += 1
            continue

        split_actions = split_non_atomic_action(instruction)
        for chunk in split_actions:
            flattened.append(PlanAction(id=next_id, instruction=chunk))
            next_id += 1

    return flattened


def salvage_plan_actions(raw_plan_text: str, fallback_request: str) -> List[PlanAction]:
    text = str(raw_plan_text or "")

    stripped = re.sub(r"</?(?:plan|action|len:[^>]+|len)>", "\n", text, flags=re.IGNORECASE)
    lines = [
        _BULLET_RE.sub("", line).strip(" ,")
        for line in stripped.splitlines()
        if _BULLET_RE.sub("", line).strip(" ,")
    ]

    if not lines:
        lines = [str(fallback_request).strip()]

    actions: List[PlanAction] = []
    next_id = 1
    for line in lines:
        chunks = split_non_atomic_action(line)
        if not chunks:
            chunks = [line]
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            actions.append(PlanAction(id=next_id, instruction=chunk))
            next_id += 1

    if not actions:
        fallback = str(fallback_request).strip()
        actions = [PlanAction(id=1, instruction=fallback)] if fallback else []

    return actions


def actions_to_instruction_list(actions: Iterable[PlanAction]) -> List[str]:
    return [str(action.instruction).strip() for action in actions if str(action.instruction).strip()]
