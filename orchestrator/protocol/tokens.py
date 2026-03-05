"""Token-level protocol helpers for plan/dispatch/response control tags."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PLAN_START = "<plan>"
PLAN_END = "</plan>"
ACTION_START = "<action>"
ACTION_END = "</action>"
DISPATCH_START = "<dispatch>"
DISPATCH_END = "</dispatch>"
ARG_START = "<arg>"
ARG_END = "</arg>"
KEY_START = "<key>"
KEY_END = "</key>"
VALUE_START = "<value>"
VALUE_END = "</value>"
RESPONSE_START = "<response>"
RESPONSE_END = "</response>"
TOOL_START = "<tool>"
TOOL_END = "</tool>"
STATUS_START = "<status>"
STATUS_END = "</status>"
TEXT_START = "<text>"
TEXT_END = "</text>"
RETRY_START = "<retry>"
RETRY_END = "</retry>"

PROTOCOL_SPECIAL_TOKENS: Tuple[str, ...] = (
    PLAN_START,
    PLAN_END,
    ACTION_START,
    ACTION_END,
    DISPATCH_START,
    DISPATCH_END,
    ARG_START,
    ARG_END,
    KEY_START,
    KEY_END,
    VALUE_START,
    VALUE_END,
    RESPONSE_START,
    RESPONSE_END,
    TOOL_START,
    TOOL_END,
    STATUS_START,
    STATUS_END,
    TEXT_START,
    TEXT_END,
    RETRY_START,
    RETRY_END,
)


@dataclass(frozen=True)
class ProtocolTokenIds:
    plan_start: int
    plan_end: int
    action_start: int
    action_end: int
    dispatch_start: int
    dispatch_end: int
    arg_start: int
    arg_end: int
    key_start: int
    key_end: int
    value_start: int
    value_end: int


def register_protocol_tokens(tokenizer: Any) -> int:
    existing = set(getattr(tokenizer, "additional_special_tokens", []) or [])
    missing = [tok for tok in PROTOCOL_SPECIAL_TOKENS if tok not in existing]
    if not missing:
        return 0
    return int(tokenizer.add_special_tokens({"additional_special_tokens": list(missing)}))


def _token_id(tokenizer: Any, token: str, strict: bool) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        if strict:
            raise ValueError(f"Protocol token not found in tokenizer: {token}")
        return -1

    unk = getattr(tokenizer, "unk_token_id", None)
    if strict and unk is not None and token_id == unk:
        raise ValueError(f"Protocol token resolved to unk_token_id: {token}")
    return int(token_id)


def resolve_protocol_token_ids(tokenizer: Any, strict: bool = False) -> ProtocolTokenIds:
    return ProtocolTokenIds(
        plan_start=_token_id(tokenizer, PLAN_START, strict=strict),
        plan_end=_token_id(tokenizer, PLAN_END, strict=strict),
        action_start=_token_id(tokenizer, ACTION_START, strict=strict),
        action_end=_token_id(tokenizer, ACTION_END, strict=strict),
        dispatch_start=_token_id(tokenizer, DISPATCH_START, strict=strict),
        dispatch_end=_token_id(tokenizer, DISPATCH_END, strict=strict),
        arg_start=_token_id(tokenizer, ARG_START, strict=strict),
        arg_end=_token_id(tokenizer, ARG_END, strict=strict),
        key_start=_token_id(tokenizer, KEY_START, strict=strict),
        key_end=_token_id(tokenizer, KEY_END, strict=strict),
        value_start=_token_id(tokenizer, VALUE_START, strict=strict),
        value_end=_token_id(tokenizer, VALUE_END, strict=strict),
    )


def spans_between_token_ids(token_ids: Sequence[int], start_id: int, end_id: int) -> List[List[int]]:
    if start_id < 0 or end_id < 0:
        return []

    ids = [int(i) for i in token_ids]
    spans: List[List[int]] = []
    idx = 0
    n = len(ids)

    while idx < n:
        while idx < n and ids[idx] != start_id:
            idx += 1
        if idx >= n:
            break
        payload_start = idx + 1
        idx = payload_start
        while idx < n and ids[idx] != end_id:
            idx += 1
        if idx >= n:
            break
        spans.append(ids[payload_start:idx])
        idx += 1

    return spans


def decode_payload_tokens(tokenizer: Any, token_ids: Iterable[int]) -> str:
    ids = [int(tok) for tok in token_ids]
    if not ids:
        return ""
    return str(tokenizer.decode(ids, skip_special_tokens=False)).strip()


def extract_plan_actions_from_ids(
    tokenizer: Any,
    generated_token_ids: Sequence[int],
    token_ids: ProtocolTokenIds,
) -> List[str]:
    actions: List[str] = []
    for payload in spans_between_token_ids(
        generated_token_ids,
        start_id=token_ids.action_start,
        end_id=token_ids.action_end,
    ):
        text = decode_payload_tokens(tokenizer, payload)
        if text:
            actions.append(text)
    return actions


def extract_dispatch_arguments_from_ids(
    tokenizer: Any,
    generated_token_ids: Sequence[int],
    token_ids: ProtocolTokenIds,
) -> Dict[str, str]:
    arguments: Dict[str, str] = {}
    arg_payloads = spans_between_token_ids(
        generated_token_ids,
        start_id=token_ids.arg_start,
        end_id=token_ids.arg_end,
    )
    for arg_payload in arg_payloads:
        key_spans = spans_between_token_ids(
            arg_payload,
            start_id=token_ids.key_start,
            end_id=token_ids.key_end,
        )
        value_spans = spans_between_token_ids(
            arg_payload,
            start_id=token_ids.value_start,
            end_id=token_ids.value_end,
        )
        if not key_spans or not value_spans:
            continue

        key = decode_payload_tokens(tokenizer, key_spans[0]).strip()
        value = decode_payload_tokens(tokenizer, value_spans[0]).strip()
        if key:
            arguments[key] = value

    return arguments
