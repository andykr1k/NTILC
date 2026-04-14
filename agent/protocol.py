from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable


ASSISTANT_CONTROL_TAGS = ("search_tools", "select_tool")
CONTROLLER_CONTROL_TAGS = ("tool_candidates", "tool_spec", "dispatch", "response")
ALL_CONTROL_TAGS = ASSISTANT_CONTROL_TAGS + CONTROLLER_CONTROL_TAGS
JSON_BLOCK_TAGS = frozenset(CONTROLLER_CONTROL_TAGS)


@dataclass(frozen=True)
class ControlBlock:
    tag: str
    content: str
    raw_text: str
    start_index: int
    end_index: int


@dataclass(frozen=True)
class ParseIssue:
    message: str
    fragment: str


@dataclass(frozen=True)
class ParserSnapshot:
    visible_text: str
    block: ControlBlock | None
    issue: ParseIssue | None
    trimmed_output: str


def serialize_text_block(tag: str, content: str) -> str:
    return f"<{tag}>{content.strip()}</{tag}>"


def serialize_json_block(tag: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return f"<{tag}>{encoded}</{tag}>"


def decode_json_block(block: ControlBlock) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(block.content)
    except json.JSONDecodeError as exc:
        return None, f"Malformed JSON in <{block.tag}> block: {exc}"
    if not isinstance(payload, dict):
        return None, f"Expected JSON object inside <{block.tag}> block."
    return payload, None


class IncrementalControlBlockParser:
    def __init__(self, tags: Iterable[str]) -> None:
        self.tags = tuple(tags)
        self.buffer = ""
        self.visible_parts: list[str] = []
        self.block: ControlBlock | None = None
        self.issue: ParseIssue | None = None
        self.finished = False
        self._cursor = 0

    def feed(self, chunk: str) -> tuple[list[str], ControlBlock | None]:
        if self.finished or not chunk:
            return [], self.block

        self.buffer += chunk
        text_chunks, block = self._drain(final=False)
        if text_chunks:
            self.visible_parts.extend(text_chunks)
        return text_chunks, block

    def finalize(self) -> ParserSnapshot:
        text_chunks, _ = self._drain(final=True)
        if text_chunks:
            self.visible_parts.extend(text_chunks)

        if self.block is None and self.buffer:
            self.visible_parts.append(self.buffer)
            self.buffer = ""

        trimmed_output = "".join(self.visible_parts)
        if self.block is not None:
            trimmed_output += self.block.raw_text

        return ParserSnapshot(
            visible_text="".join(self.visible_parts),
            block=self.block,
            issue=self.issue,
            trimmed_output=trimmed_output,
        )

    def _drain(self, final: bool) -> tuple[list[str], ControlBlock | None]:
        emitted_text: list[str] = []

        while self.buffer and self.block is None and self.issue is None:
            start_index = self.buffer.find("<")
            if start_index == -1:
                emitted_text.append(self.buffer)
                self._cursor += len(self.buffer)
                self.buffer = ""
                break

            if start_index > 0:
                visible = self.buffer[:start_index]
                emitted_text.append(visible)
                self._cursor += len(visible)
                self.buffer = self.buffer[start_index:]

            matched_tag = self._match_opening_tag(self.buffer)
            if matched_tag is None:
                if final and self._looks_like_partial_opening(self.buffer):
                    self.issue = ParseIssue(
                        message="Incomplete control block opener.",
                        fragment=self.buffer,
                    )
                    self.buffer = ""
                    break
                if not final and self._looks_like_partial_opening(self.buffer):
                    break
                emitted_text.append(self.buffer[0])
                self._cursor += 1
                self.buffer = self.buffer[1:]
                continue

            open_tag = f"<{matched_tag}>"
            close_tag = f"</{matched_tag}>"
            close_index = self.buffer.find(close_tag, len(open_tag))
            if close_index == -1:
                if final:
                    self.issue = ParseIssue(
                        message=f"Incomplete <{matched_tag}> control block.",
                        fragment=self.buffer,
                    )
                    self.buffer = ""
                break

            raw_text = self.buffer[: close_index + len(close_tag)]
            content = raw_text[len(open_tag) : -len(close_tag)]
            start = self._cursor
            self.block = ControlBlock(
                tag=matched_tag,
                content=content.strip(),
                raw_text=raw_text,
                start_index=start,
                end_index=start + len(raw_text),
            )
            self._cursor += len(raw_text)
            self.buffer = ""
            self.finished = True
            break

        return emitted_text, self.block

    def _match_opening_tag(self, text: str) -> str | None:
        for tag in self.tags:
            if text.startswith(f"<{tag}>"):
                return tag
        return None

    def _looks_like_partial_opening(self, text: str) -> bool:
        if not text.startswith("<"):
            return False
        for tag in self.tags:
            opening = f"<{tag}>"
            if opening.startswith(text):
                return True
        return False


def parse_text_with_control_blocks(text: str, tags: Iterable[str]) -> ParserSnapshot:
    parser = IncrementalControlBlockParser(tags)
    parser.feed(text)
    return parser.finalize()
