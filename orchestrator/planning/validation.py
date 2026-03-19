"""
Validation and fallback splitting for atomic plan actions.
"""

from __future__ import annotations

import re
import shlex
from typing import Iterable, List, Sequence

from .parser import PlanAction


_SPLIT_SENTENCE_RE = re.compile(r"\s*(?:;|\bthen\b|\band then\b)\s*", flags=re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")
_SHELL_OPERATOR_RE = re.compile(r"(?:\|\||&&|[|><`])|\$\(")
_OPTION_RE = re.compile(r"^-{1,2}[A-Za-z0-9]")
_PATHLIKE_RE = re.compile(r"^(?:\.{1,2}/|/|~|[*?])")
_FILENAME_RE = re.compile(r".+\.[A-Za-z0-9]{1,8}$")

_UNAMBIGUOUS_COMMAND_HEADS = {
    "ls",
    "grep",
    "cat",
    "mkdir",
    "rm",
    "cp",
    "mv",
    "pwd",
    "touch",
    "echo",
    "sed",
    "awk",
    "chmod",
    "chown",
    "curl",
    "wget",
    "tar",
    "zip",
    "unzip",
    "ps",
    "kill",
    "du",
    "df",
    "wc",
    "xargs",
    "which",
    "whereis",
    "locate",
    "basename",
    "dirname",
    "less",
    "more",
    "tee",
    "tr",
    "realpath",
    "readlink",
}
_AMBIGUOUS_COMMAND_HEADS = {"find", "sort", "uniq", "cut", "head", "tail"}
_NATURAL_LANGUAGE_OBJECT_WORDS = {
    "file",
    "files",
    "directory",
    "directories",
    "folder",
    "folders",
    "repo",
    "repository",
    "text",
    "references",
    "reference",
    "logs",
    "log",
    "lines",
    "line",
    "processes",
    "process",
    "permissions",
    "usage",
    "matches",
    "content",
    "contents",
}


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


def _tokenize_shellish_text(text: str) -> List[str]:
    try:
        return shlex.split(str(text))
    except ValueError:
        return str(text).split()


def _looks_shellish_token(token: str) -> bool:
    value = str(token).strip()
    if not value:
        return False
    if _OPTION_RE.match(value):
        return True
    if _PATHLIKE_RE.match(value):
        return True
    if value in {".", ".."}:
        return True
    if "/" in value or "*" in value or "?" in value:
        return True
    if "=" in value:
        return True
    if _FILENAME_RE.match(value):
        return True
    return False


def looks_like_shell_command(instruction: str) -> bool:
    text = str(instruction).strip()
    if not text:
        return False
    if _SHELL_OPERATOR_RE.search(text):
        return True

    tokens = _tokenize_shellish_text(text)
    if not tokens:
        return False

    head = tokens[0].lower()
    if head in _UNAMBIGUOUS_COMMAND_HEADS:
        return True
    if head not in _AMBIGUOUS_COMMAND_HEADS:
        return False
    if len(tokens) == 1:
        return True
    if any(_looks_shellish_token(token) for token in tokens[1:]):
        return True

    nl_hints = sum(1 for token in tokens[1:] if token.lower() in _NATURAL_LANGUAGE_OBJECT_WORDS)
    return nl_hints == 0


def _describe_shell_command(head: str, tokens: Sequence[str]) -> str:
    lowered = str(head).lower()
    flags = {token for token in tokens[1:] if str(token).startswith("-")}
    recursive = any(flag in {"-r", "-R", "--recursive"} for flag in flags)

    mapping = {
        "ls": "list files",
        "grep": "search for matching text recursively" if recursive else "search for matching text",
        "find": "find matching files",
        "cat": "read file contents",
        "mkdir": "create a directory",
        "rm": "remove files",
        "cp": "copy files",
        "mv": "move or rename files",
        "pwd": "show the current directory",
        "touch": "create a file",
        "echo": "write text output",
        "head": "show the first lines of a file",
        "tail": "show the last lines of a file",
        "sed": "edit or transform text",
        "awk": "extract or transform text",
        "sort": "sort text lines",
        "uniq": "deduplicate repeated lines",
        "wc": "count lines words or bytes",
        "chmod": "change file permissions",
        "chown": "change file ownership",
        "curl": "download content",
        "wget": "download content",
        "tar": "archive or extract files",
        "zip": "create an archive",
        "unzip": "extract an archive",
        "ps": "list running processes",
        "kill": "stop a process",
        "du": "measure disk usage",
        "df": "show filesystem usage",
        "cut": "extract fields from text",
        "xargs": "build command arguments from input",
        "which": "locate an executable",
        "whereis": "locate a program",
        "locate": "find matching files",
        "basename": "extract a file name from a path",
        "dirname": "extract a directory path",
        "less": "view file contents",
        "more": "view file contents",
        "tee": "write output to a file and stdout",
        "tr": "translate or delete characters",
        "realpath": "resolve a path",
        "readlink": "resolve a symlink or path",
    }
    return mapping.get(lowered, "")


def coerce_action_to_natural_language(instruction: str, fallback_request: str = "") -> str:
    text = str(instruction).strip()
    if not text:
        return ""
    if not looks_like_shell_command(text):
        return text

    tokens = _tokenize_shellish_text(text)
    if not tokens:
        return str(fallback_request).strip()

    description = _describe_shell_command(tokens[0], tokens)
    if description:
        return description

    fallback = str(fallback_request).strip()
    if fallback:
        return fallback
    return "complete the requested shell task"


def normalize_actions_to_natural_language(
    actions: Sequence[PlanAction],
    fallback_request: str = "",
) -> List[PlanAction]:
    normalized: List[PlanAction] = []
    next_id = 1
    for action in actions:
        instruction = coerce_action_to_natural_language(
            instruction=str(action.instruction),
            fallback_request=fallback_request,
        )
        if not instruction:
            continue
        normalized.append(PlanAction(id=next_id, instruction=instruction))
        next_id += 1
    return normalized


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

    return normalize_actions_to_natural_language(actions, fallback_request=fallback_request)


def actions_to_instruction_list(actions: Iterable[PlanAction]) -> List[str]:
    return [str(action.instruction).strip() for action in actions if str(action.instruction).strip()]
