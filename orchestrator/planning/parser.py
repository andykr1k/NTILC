"""
Plan block parsing for atomic action extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape as xml_unescape
from typing import List


_PLAN_BLOCK_RE = re.compile(r"<plan>(.*?)</plan>", flags=re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"<action>(.*?)</action>", flags=re.IGNORECASE | re.DOTALL)
_LEN_RE = re.compile(r"<len:\s*\d+\s*>(.*?)</len>", flags=re.IGNORECASE | re.DOTALL)


@dataclass
class PlanAction:
    id: int
    instruction: str

    def to_dict(self) -> dict:
        return {"id": int(self.id), "instruction": self.instruction}


def _strip_len_wrapper(text: str) -> str:
    match = _LEN_RE.search(text)
    if match:
        return xml_unescape(match.group(1)).strip()
    return xml_unescape(text).strip()


def parse_plan_block(plan_text: str) -> List[PlanAction]:
    text = str(plan_text or "")
    block_match = _PLAN_BLOCK_RE.search(text)
    if not block_match:
        raise ValueError("No <plan>...</plan> block found.")

    body = block_match.group(1)
    action_chunks = _ACTION_RE.findall(body)
    if not action_chunks:
        raise ValueError("No <action>...</action> entries found in plan block.")

    actions: List[PlanAction] = []
    for idx, chunk in enumerate(action_chunks, start=1):
        instruction = _strip_len_wrapper(chunk)
        if not instruction:
            continue
        actions.append(PlanAction(id=idx, instruction=instruction))

    if not actions:
        raise ValueError("Parsed plan block but all actions were empty.")

    return actions
