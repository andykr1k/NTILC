from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import time
from typing import Any

import streamlit as st

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from agent.runtime import (
    AgentController,
    RuntimeConfig,
    build_agent_resources,
    truncate_text,
)


def is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime import exists
    except Exception:
        return False
    return bool(exists())


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 980px;
            padding-top: 1.25rem;
            padding-bottom: 6rem;
        }
        [data-testid="stChatMessageContent"] {
            max-width: 860px;
        }
        [data-testid="stSidebar"] {
            border-right: 0.5px solid rgba(0,0,0,0.08);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 0.85rem;
            padding-bottom: 0.75rem;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.35rem;
        }
        [data-testid="stSidebar"] form {
            border: 0 !important;
            padding: 0 !important;
            background: transparent !important;
        }
        [data-testid="stSidebar"] .stButton button,
        [data-testid="stSidebar"] .stFormSubmitButton button {
            min-height: 2rem;
            padding-top: 0.25rem;
            padding-bottom: 0.25rem;
        }
        [data-testid="stSidebar"] [data-testid="stTextInput"],
        [data-testid="stSidebar"] [data-testid="stNumberInput"],
        [data-testid="stSidebar"] [data-testid="stSelectbox"],
        [data-testid="stSidebar"] [data-testid="stCheckbox"] {
            margin-bottom: 0.05rem;
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stCaption {
            margin-bottom: 0.1rem;
        }
        [data-testid="stSidebar"] hr {
            margin: 0.35rem 0 0.5rem 0;
        }
        .sidebar-section-label {
            color: rgba(80,80,90,0.7);
            margin-top: 0.25rem;
            margin-bottom: 0.1rem;
        }
        .stats-card {
            border: 0.5px solid rgba(0,0,0,0.07);
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            background: #fbfbfc;
            min-height: 124px;
            height: 124px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 0.2rem;
            margin-bottom: 0.5rem;
        }
        .stats-card-title {
            font-size: 0.82rem;
            color: #475569;
            margin-bottom: 0.2rem;
            line-height: 1.25;
            min-height: 2.05rem;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .stats-card-value {
            font-size: 1.3rem;
            font-weight: 600;
            color: #0f172a;
            line-height: 1.2;
            min-height: 1.6rem;
        }
        .stats-card-subtle {
            font-size: 0.78rem;
            color: #64748b;
            margin-top: 0.2rem;
            line-height: 1.25;
            min-height: 1.95rem;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .tool-response-card {
            border-radius: 10px;
            padding: 0.65rem 0.9rem;
            margin: 0.6rem 0 0.25rem 0;
            border: 0.5px solid rgba(0,0,0,0.07);
            background: #f8fafc;
        }
        .tool-response-ok {
            border-left: 3px solid #1D9E75;
            background: #f2fbf7;
            border-radius: 0 10px 10px 0;
        }
        .tool-response-error {
            border-left: 3px solid #E24B4A;
            background: #fff5f5;
            border-radius: 0 10px 10px 0;
        }
        .tool-response-warning {
            border-left: 3px solid #EF9F27;
            background: #fffaf0;
            border-radius: 0 10px 10px 0;
        }
        .tool-response-title {
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            margin-bottom: 0.3rem;
            color: #1e293b;
        }
        .tool-response-body {
            color: #475569;
            font-size: 0.8rem;
            font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
            white-space: pre-wrap;
            line-height: 1.55;
        }
        div[data-testid="stChatInput"] {
            border-top: 0.5px solid rgba(0,0,0,0.07);
            padding-top: 0.75rem;
            background: transparent;
        }
        div[data-testid="stChatInput"] textarea {
            border-radius: 12px !important;
            font-size: 0.9rem !important;
        }
        [data-testid="stExpander"] {
            border-radius: 8px !important;
            border: 0.5px solid rgba(0,0,0,0.07) !important;
            margin-top: 0.3rem;
        }
        [data-testid="stExpander"] summary {
            font-size: 0.78rem !important;
            color: rgba(80,80,90,0.8) !important;
            font-weight: 500 !important;
        }
        [data-testid="stChatMessage"] {
            padding: 0.5rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "events" not in st.session_state:
        st.session_state.events = []
    if "controller_transcript" not in st.session_state:
        st.session_state.controller_transcript = []
    if "runtime_ready" not in st.session_state:
        st.session_state.runtime_ready = False
    if "busy" not in st.session_state:
        st.session_state.busy = False
    if "turn_stats" not in st.session_state:
        st.session_state.turn_stats = []
    if "session_started_at" not in st.session_state:
        st.session_state.session_started_at = time.time()
    if "runtime_config_values" not in st.session_state:
        defaults = RuntimeConfig()
        st.session_state.runtime_config_values = {
            "tools_path": str(defaults.tools_path),
            "embed_checkpoint_path": str(defaults.embed_checkpoint_path),
            "qwen_model_name": defaults.qwen_model_name,
            "qwen_device": defaults.qwen_device,
            "embed_device": defaults.embed_device,
            "qwen_dtype": defaults.qwen_dtype,
            "local_files_only": defaults.local_files_only,
            "top_k": defaults.top_k,
            "max_tool_steps": defaults.max_tool_steps,
            "max_agent_passes": defaults.max_agent_passes,
        }


def clear_chat_state() -> None:
    st.session_state.messages = []
    st.session_state.events = []
    st.session_state.controller_transcript = []
    st.session_state.turn_stats = []
    st.session_state.session_started_at = time.time()
    st.session_state.busy = False


def build_runtime_config(values: dict[str, Any]) -> RuntimeConfig:
    return RuntimeConfig(
        tools_path=Path(values["tools_path"]).expanduser(),
        embed_checkpoint_path=Path(values["embed_checkpoint_path"]).expanduser(),
        qwen_model_name=str(values["qwen_model_name"]),
        qwen_device=str(values["qwen_device"]),
        embed_device=str(values["embed_device"]),
        qwen_dtype=str(values["qwen_dtype"]),
        local_files_only=bool(values["local_files_only"]),
        top_k=int(values["top_k"]),
        max_tool_steps=int(values["max_tool_steps"]),
        max_agent_passes=int(values["max_agent_passes"]),
    )


if is_streamlit_runtime():

    @st.cache_resource(show_spinner="Loading NTILC runtime…")
    def get_cached_resources(config: RuntimeConfig):
        return build_agent_resources(config)

else:

    def get_cached_resources(config: RuntimeConfig):
        return build_agent_resources(config)


def render_sidebar() -> RuntimeConfig:
    st.sidebar.markdown("### Runtime")

    if st.sidebar.button("Clear chat", use_container_width=True):
        clear_chat_state()
        st.rerun()

    with st.sidebar.form("runtime_config_form"):
        values = st.session_state.runtime_config_values

        st.markdown('<div class="sidebar-section-label">Paths</div>', unsafe_allow_html=True)
        tools_path = st.text_input("Tool catalog", value=values["tools_path"])
        embed_checkpoint_path = st.text_input("Embedding checkpoint", value=values["embed_checkpoint_path"])

        st.markdown('<div class="sidebar-section-label">Model</div>', unsafe_allow_html=True)
        qwen_model_name = st.text_input("Qwen model", value=values["qwen_model_name"])

        col1, col2 = st.columns(2)
        with col1:
            qwen_device = st.text_input("Qwen device", value=values["qwen_device"])
        with col2:
            embed_device = st.text_input("Embed device", value=values["embed_device"])

        qwen_dtype = st.selectbox(
            "Dtype",
            options=["bfloat16", "float16", "float32", "auto"],
            index=["bfloat16", "float16", "float32", "auto"].index(values["qwen_dtype"]),
        )
        local_files_only = st.checkbox("Local files only", value=values["local_files_only"])

        st.markdown('<div class="sidebar-section-label">Limits</div>', unsafe_allow_html=True)
        col3, col4, col5 = st.columns(3)
        with col3:
            top_k = st.number_input("Top-k", min_value=1, max_value=10, value=values["top_k"])
        with col4:
            max_tool_steps = st.number_input("Steps", min_value=1, max_value=999, value=values["max_tool_steps"])
        with col5:
            max_agent_passes = st.number_input("Passes", min_value=1, max_value=999, value=values["max_agent_passes"])

        apply_settings = st.form_submit_button("Apply settings", use_container_width=True)

    if apply_settings:
        st.session_state.runtime_config_values = {
            "tools_path": tools_path,
            "embed_checkpoint_path": embed_checkpoint_path,
            "qwen_model_name": qwen_model_name,
            "qwen_device": qwen_device,
            "embed_device": embed_device,
            "qwen_dtype": qwen_dtype,
            "local_files_only": local_files_only,
            "top_k": int(top_k),
            "max_tool_steps": int(max_tool_steps),
            "max_agent_passes": int(max_agent_passes),
        }
        st.rerun()

    return build_runtime_config(st.session_state.runtime_config_values)


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def render_stats_card(label: str, value: str, subtitle: str | None = None) -> None:
    subtitle_html = (
        f'<div class="stats-card-subtle">{subtitle}</div>' if subtitle else ""
    )
    st.markdown(
        f"""
        <div class="stats-card">
            <div class="stats-card-title">{label}</div>
            <div class="stats-card-value">{value}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_session_stats(resources) -> dict[str, Any]:
    turn_stats = st.session_state.turn_stats
    messages = st.session_state.messages
    tools_counter = Counter(
        tool_name
        for stats in turn_stats
        for tool_name in stats.get("tools_used", [])
    )
    queries_counter = Counter(
        query
        for stats in turn_stats
        for query in stats.get("search_queries", [])
    )

    total_input_tokens = sum(stats.get("model_input_tokens", 0) for stats in turn_stats)
    total_output_tokens = sum(stats.get("model_output_tokens", 0) for stats in turn_stats)
    current_transcript_tokens = 0
    if resources is not None:
        current_transcript_tokens = sum(
            resources.model_adapter.count_text_tokens(message["content"])
            for message in st.session_state.controller_transcript
        )

    return {
        "turn_count": len(turn_stats),
        "user_turn_count": sum(1 for message in messages if message["role"] == "user"),
        "assistant_turn_count": sum(1 for message in messages if message["role"] == "assistant"),
        "user_message_tokens": sum(stats.get("user_message_tokens", 0) for stats in turn_stats),
        "assistant_prompt_tokens": sum(stats.get("assistant_prompt_tokens", 0) for stats in turn_stats),
        "assistant_output_tokens": sum(stats.get("assistant_output_tokens", 0) for stats in turn_stats),
        "dispatch_prompt_tokens": sum(stats.get("dispatch_prompt_tokens", 0) for stats in turn_stats),
        "dispatch_output_tokens": sum(stats.get("dispatch_output_tokens", 0) for stats in turn_stats),
        "search_query_tokens": sum(stats.get("search_query_tokens", 0) for stats in turn_stats),
        "model_input_tokens": total_input_tokens,
        "model_output_tokens": total_output_tokens,
        "total_model_tokens": total_input_tokens + total_output_tokens,
        "search_count": sum(stats.get("search_count", 0) for stats in turn_stats),
        "candidates_returned": sum(stats.get("candidates_returned", 0) for stats in turn_stats),
        "dispatch_count": sum(stats.get("dispatch_count", 0) for stats in turn_stats),
        "tool_call_count": sum(stats.get("tool_call_count", 0) for stats in turn_stats),
        "tool_success_count": sum(stats.get("tool_success_count", 0) for stats in turn_stats),
        "tool_error_count": sum(stats.get("tool_error_count", 0) for stats in turn_stats),
        "controller_error_count": sum(stats.get("controller_error_count", 0) for stats in turn_stats),
        "agent_passes": sum(stats.get("agent_passes", 0) for stats in turn_stats),
        "duration_seconds": time.time() - st.session_state.session_started_at,
        "current_transcript_messages": len(st.session_state.controller_transcript),
        "current_transcript_tokens": current_transcript_tokens,
        "tools_counter": tools_counter,
        "queries_counter": queries_counter,
        "turn_rows": turn_stats,
    }


def render_stats_tab(resources) -> None:
    session_stats = compute_session_stats(resources)

    st.subheader("Current Session", anchor=False)
    if session_stats["turn_count"] == 0:
        st.info("No session activity yet. Send a message in the Chat tab to populate stats.")

    row1 = st.columns(4)
    with row1[0]:
        render_stats_card("Turns", str(session_stats["turn_count"]), f"{session_stats['user_turn_count']} user")
    with row1[1]:
        render_stats_card("Model Input Tokens", str(session_stats["model_input_tokens"]))
    with row1[2]:
        render_stats_card("Model Output Tokens", str(session_stats["model_output_tokens"]))
    with row1[3]:
        render_stats_card("Total Model Tokens", str(session_stats["total_model_tokens"]))

    row2 = st.columns(4)
    with row2[0]:
        render_stats_card("Tool Searches", str(session_stats["search_count"]))
    with row2[1]:
        render_stats_card("Dispatches", str(session_stats["dispatch_count"]))
    with row2[2]:
        render_stats_card("Tool Calls", str(session_stats["tool_call_count"]))
    with row2[3]:
        render_stats_card(
            "Successful Calls",
            str(session_stats["tool_success_count"]),
            f"{session_stats['tool_error_count']} failed",
        )

    row3 = st.columns(4)
    with row3[0]:
        render_stats_card("Unique Tools", str(len(session_stats["tools_counter"])))
    with row3[1]:
        render_stats_card("Search Query Tokens", str(session_stats["search_query_tokens"]))
    with row3[2]:
        render_stats_card("Assistant Passes", str(session_stats["agent_passes"]))
    with row3[3]:
        render_stats_card("Session Age", format_duration(session_stats["duration_seconds"]))

    row4 = st.columns(4)
    with row4[0]:
        render_stats_card("User Message Tokens", str(session_stats["user_message_tokens"]))
    with row4[1]:
        render_stats_card("Current Transcript Messages", str(session_stats["current_transcript_messages"]))
    with row4[2]:
        render_stats_card("Current Transcript Tokens", str(session_stats["current_transcript_tokens"]))
    with row4[3]:
        render_stats_card("Controller Errors", str(session_stats["controller_error_count"]))

    tool_rows = [
        {"tool": tool_name, "count": count}
        for tool_name, count in session_stats["tools_counter"].most_common()
    ]
    query_rows = [
        {"query": query, "count": count}
        for query, count in session_stats["queries_counter"].most_common()
    ]
    turn_rows = [
        {
            "turn_id": row.get("turn_id", ""),
            "duration_s": row.get("duration_seconds", 0.0),
            "passes": row.get("agent_passes", 0),
            "input_tokens": row.get("model_input_tokens", 0),
            "output_tokens": row.get("model_output_tokens", 0),
            "searches": row.get("search_count", 0),
            "tool_calls": row.get("tool_call_count", 0),
            "tools_used": ", ".join(row.get("tools_used", [])),
        }
        for row in session_stats["turn_rows"]
    ]

    table_col1, table_col2 = st.columns(2)
    with table_col1:
        st.caption("Tool usage")
        if tool_rows:
            st.dataframe(tool_rows, use_container_width=True, hide_index=True)
        else:
            st.write("No tools used yet.")
    with table_col2:
        st.caption("Search queries")
        if query_rows:
            st.dataframe(query_rows, use_container_width=True, hide_index=True)
        else:
            st.write("No searches yet.")

    st.caption("Per-turn metrics")
    if turn_rows:
        st.dataframe(turn_rows, use_container_width=True, hide_index=True)
    else:
        st.write("No completed turns yet.")


def render_response_card(event: dict[str, Any]) -> None:
    status = str(event.get("status", "warning"))
    style = {
        "ok": "tool-response-ok",
        "error": "tool-response-error",
    }.get(status, "tool-response-warning")
    preview_source = event.get("error") or event.get("output") or "No output."
    preview = truncate_text(preview_source, 280)
    tool_name = event.get("tool", "controller")

    status_icon = {"ok": "✓", "error": "✗", "warning": "⚠"}.get(status, "·")

    st.markdown(
        f"""
        <div class="tool-response-card {style}">
            <div class="tool-response-title">{status_icon}&nbsp;&nbsp;{tool_name} · {status}</div>
            <div class="tool-response-body">{preview}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"Raw · {tool_name}", expanded=status != "ok"):
        st.code(str(event.get("block", "")), language="xml")
        st.json(event.get("raw_payload", {}))
        if event.get("output") is not None:
            st.caption("Raw output")
            if isinstance(event["output"], (dict, list)):
                st.json(event["output"])
            else:
                st.code(str(event["output"]))


def render_assistant_event(event: dict[str, Any]) -> None:
    if event["type"] == "search":
        with st.expander(f"🔍 Tool search · {event['query']}", expanded=False):
            st.code(str(event["block"]), language="xml")
            st.json(event["raw_payload"])
    elif event["type"] == "dispatch":
        with st.expander(f"⚙ Dispatch · {event['tool']}", expanded=False):
            st.code(str(event["block"]), language="xml")
            st.json(event["raw_payload"])
    elif event["type"] == "response":
        render_response_card(event)


def render_assistant_events(events: list[dict[str, Any]]) -> None:
    for event in events:
        render_assistant_event(event)


def render_assistant_blocks(blocks: list[dict[str, Any]]) -> None:
    for block in blocks:
        if block.get("type") == "text":
            st.markdown(block.get("content", "") or " ")
        else:
            render_assistant_event(block)


def get_turn_events(turn_id: str) -> list[dict[str, Any]]:
    return [event for event in st.session_state.events if event["turn_id"] == turn_id]


class LiveTurnRenderer:
    def __init__(self, container) -> None:
        self.container = container
        self.blocks: list[dict[str, Any]] = []

    def on_text(self, chunk: str) -> None:
        if not chunk:
            return
        if self.blocks and self.blocks[-1]["type"] == "text":
            self.blocks[-1]["content"] += chunk
        else:
            self.blocks.append({"type": "text", "content": chunk})
        self._render()

    def on_event(self, event: dict[str, Any]) -> None:
        self.blocks.append(dict(event))
        self._render()

    def export_blocks(self) -> list[dict[str, Any]]:
        return [dict(block) for block in self.blocks]

    def _render(self) -> None:
        with self.container.container():
            render_assistant_blocks(self.blocks)


def render_message_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                if message.get("blocks"):
                    render_assistant_blocks(message["blocks"])
                elif message.get("content"):
                    st.markdown(message["content"])
                    render_assistant_events(get_turn_events(message["turn_id"]))
            elif message.get("content"):
                st.markdown(message["content"])


def main() -> None:
    if not is_streamlit_runtime():
        print("This entrypoint must be launched with Streamlit.")
        print("Run: streamlit run agent/agent.py")
        return

    st.set_page_config(
        page_title="NTILC Tool Agent",
        page_icon="⬡",
        layout="wide",
    )
    inject_styles()
    init_session_state()

    st.title("NTILC Tool Agent")
    st.caption(
        "Qwen3-family assistant · embedding-based tool retrieval · dynamic tool specs · in-process execution"
    )

    config = render_sidebar()
    runtime_error: str | None = None
    resources = None

    try:
        resources = get_cached_resources(config)
    except Exception as exc:
        runtime_error = str(exc)
        st.session_state.runtime_ready = False
    else:
        st.session_state.runtime_ready = True

    if st.session_state.runtime_ready:
        st.sidebar.success(f"✓ Runtime ready · {config.qwen_model_name}")
        st.sidebar.caption(
            f"Catalog: `{config.tools_path}`  \n"
            f"Embed: `{config.embed_checkpoint_path}`"
        )
    else:
        st.sidebar.error("Runtime unavailable")

    chat_tab, stats_tab = st.tabs(["Chat", "Stats"])

    with chat_tab:
        chat_messages_container = st.container()
        with chat_messages_container:
            render_message_history()
            if runtime_error is not None:
                st.error(f"**Runtime error:** {runtime_error}")

    with stats_tab:
        render_stats_tab(resources)

    prompt = st.chat_input(
        "Ask the agent for a task or question…",
        disabled=(not st.session_state.runtime_ready) or st.session_state.busy,
    )
    if not prompt or resources is None:
        return

    st.session_state.busy = True
    turn_id = f"turn-{len(st.session_state.messages)}"
    st.session_state.messages.append({"role": "user", "content": prompt, "turn_id": turn_id})

    with chat_tab:
        with chat_messages_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                render_container = st.empty()
                renderer = LiveTurnRenderer(render_container)
                controller = AgentController(
                    config=config,
                    model_adapter=resources.model_adapter,
                    retriever=resources.retriever,
                    tool_by_name=resources.tool_by_name,
                )
                try:
                    result = controller.run_turn(
                        user_message=prompt,
                        transcript=st.session_state.controller_transcript,
                        on_text=renderer.on_text,
                        on_event=renderer.on_event,
                    )
                except Exception as exc:
                    st.error(f"Agent turn failed: {exc}")
                    return
                finally:
                    st.session_state.busy = False

    st.session_state.controller_transcript = result.transcript
    assistant_text = result.assistant_text or "_No assistant text returned._"
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_text,
            "turn_id": turn_id,
            "blocks": renderer.export_blocks(),
        }
    )
    stored_stats = dict(result.stats)
    stored_stats["turn_id"] = turn_id
    st.session_state.turn_stats.append(stored_stats)
    for event in result.events:
        stored_event = dict(event)
        stored_event["turn_id"] = turn_id
        st.session_state.events.append(stored_event)
    st.rerun()


if __name__ == "__main__":
    main()
