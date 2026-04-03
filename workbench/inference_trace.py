#!/usr/bin/env python3
"""Streamlit UI for NTILC vs prompt-only trace comparison."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from old.test import (
    DEFAULT_LORA_ADAPTER,
    DEFAULT_RAW_TOOLS_JSON,
    DEFAULT_RETRIEVAL_CKPT,
    TestInference,
)


st.set_page_config(
    page_title="NTILC vs Prompt Baseline",
    page_icon=":microscope:",
    layout="wide",
)


def maybe_none(text: str) -> Optional[str]:
    value = str(text).strip()
    return value or None


@st.cache_resource(show_spinner=False)
def load_runner(
    query_encoder_path: str,
    lora_adapter_path: str,
    raw_tools_path: str,
    qwen_model: Optional[str],
    device: Optional[str],
    baseline_device: Optional[str],
    max_seq_len: int,
    max_new_tokens: int,
    baseline_max_new_tokens: int,
    temperature: float,
    top_p: float,
    tool_timeout_seconds: int,
) -> TestInference:
    return TestInference(
        query_encoder_path=Path(query_encoder_path),
        lora_adapter_path=Path(lora_adapter_path) if str(lora_adapter_path).strip() else None,
        raw_tools_path=Path(raw_tools_path),
        qwen_model_name_or_path=qwen_model,
        device=device,
        baseline_device=baseline_device,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        baseline_max_new_tokens=baseline_max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        tool_timeout_seconds=tool_timeout_seconds,
    )


def ntilc_stage_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        {
            "stage": "plan",
            **result["plan"]["metrics"],
        },
    ]

    for retrieval in result.get("retrievals", []):
        rows.append(
            {
                "stage": f"retrieval action #{retrieval['action_id']}",
                **retrieval["metrics"],
            }
        )

    for step in result["steps"]:
        attempt = int(step["attempt"])
        action_id = int(step["action_id"])
        rows.append({"stage": f"dispatch a{action_id}#{attempt}", **step["dispatch"]["metrics"]})
        rows.append({"stage": f"execute a{action_id}#{attempt}", **step["execution"]["metrics"]})
        rows.append({"stage": f"response a{action_id}#{attempt}", **step["response"]["metrics"]})

    return rows


def baseline_stage_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"stage": "generation", **result["generation"]["metrics"]},
        {"stage": "dispatch", **result["dispatch"]["metrics"]},
        {"stage": "execution", **result["execution"]["metrics"]},
        {"stage": "response", **result["response"]["metrics"]},
    ]


def render_ntilc_summary(result: Dict[str, Any]) -> None:
    metrics = result["metrics"]
    cols = st.columns(5)
    cols[0].metric("Success", "Yes" if result["success"] else "No")
    cols[1].metric("Total Time", f"{metrics['elapsed_ms']:.1f} ms")
    cols[2].metric("Prompt Tokens", str(metrics["prompt_tokens"]))
    cols[3].metric("Generated Tokens", str(metrics["generated_tokens"]))
    cols[4].metric("Total Model Tokens", str(metrics["total_model_tokens"]))

    cols = st.columns(5)
    cols[0].metric("Plan Time", f"{metrics['plan_elapsed_ms']:.1f} ms")
    cols[1].metric("Retrieval Time", f"{metrics['retrieval_elapsed_ms']:.1f} ms")
    cols[2].metric("Dispatch Time", f"{metrics['dispatch_elapsed_ms']:.1f} ms")
    cols[3].metric("Execution Time", f"{metrics['execution_elapsed_ms']:.1f} ms")
    cols[4].metric("Response Time", f"{metrics['response_elapsed_ms']:.1f} ms")


def render_plan(result: Dict[str, Any]) -> None:
    st.subheader("Plan")
    cols = st.columns([3, 2])
    with cols[0]:
        st.code(result["plan"]["plan_block"], language="xml")
        if result["plan"].get("prompt"):
            with st.expander("Plan Generation Prompt"):
                st.code(result["plan"]["prompt"])
        if result["plan"].get("raw_output"):
            with st.expander("Plan Raw Model Output"):
                st.code(result["plan"]["raw_output"])
    with cols[1]:
        st.json(result["plan"]["metrics"])


def render_retrieval(result: Dict[str, Any]) -> None:
    st.subheader("Retrieval")
    retrievals = result.get("retrievals", [])
    if not retrievals:
        st.info("No retrievals were recorded.")
        return

    for retrieval in retrievals:
        label = f"Action {retrieval['action_id']}: {retrieval['action']}"
        with st.expander(label, expanded=retrieval["action_id"] == 1):
            cols = st.columns([3, 2])
            with cols[0]:
                candidates = retrieval["candidates"]
                if candidates:
                    st.dataframe(pd.DataFrame(candidates), use_container_width=True)
                else:
                    st.info("No candidates returned.")
            with cols[1]:
                st.json(retrieval["metrics"])


def render_ntilc_stage_table(result: Dict[str, Any]) -> None:
    st.subheader("Stage Metrics")
    frame = pd.DataFrame(ntilc_stage_rows(result))
    st.dataframe(frame, use_container_width=True)
    if not frame.empty:
        st.bar_chart(frame[["stage", "elapsed_ms"]].set_index("stage"), use_container_width=True)


def render_attempts(result: Dict[str, Any]) -> None:
    st.subheader("Attempts")
    if not result["steps"]:
        st.info("No attempts were made.")
        return

    for step in result["steps"]:
        candidate = step["candidate"]
        label = (
            f"Action {step['action_id']} | Attempt {step['attempt']} | {candidate['tool_name']} | "
            f"score={candidate['score']:.4f}"
        )
        with st.expander(label, expanded=step["attempt"] == 1):
            cols = st.columns([3, 2])
            with cols[0]:
                st.markdown("Action")
                st.code(step["action"])
                st.markdown("Dispatch Block")
                st.code(step["dispatch"]["dispatch_block"], language="xml")
                st.markdown("Response Block")
                st.code(step["response"]["response_block"], language="xml")
            with cols[1]:
                st.markdown("Candidate")
                st.json(candidate)
                st.markdown("Step Metrics")
                st.json(
                    {
                        "dispatch": step["dispatch"]["metrics"],
                        "execution": step["execution"]["metrics"],
                        "response": step["response"]["metrics"],
                        "step_total": step["metrics"],
                    }
                )

            if step["dispatch"]["prompt"]:
                st.markdown("Generation Prompt")
                st.code(step["dispatch"]["prompt"])
            if step["dispatch"].get("raw_output"):
                st.markdown("Raw Model Output")
                st.code(step["dispatch"]["raw_output"])

            st.markdown("Generated Command")
            st.code(step["dispatch"]["command"], language="bash")

            output = step["execution"]["stdout"] or step["execution"]["stderr"] or "[no output]"
            st.markdown("Execution Output")
            st.code(output)


def render_prompt_baseline(result: Dict[str, Any]) -> None:
    metrics = result["metrics"]
    cols = st.columns(5)
    cols[0].metric("Success", "Yes" if result["success"] else "No")
    cols[1].metric("Total Time", f"{metrics['elapsed_ms']:.1f} ms")
    cols[2].metric("Prompt Tokens", str(metrics["prompt_tokens"]))
    cols[3].metric("Generated Tokens", str(metrics["generated_tokens"]))
    cols[4].metric("Total Model Tokens", str(metrics["total_model_tokens"]))

    cols = st.columns(5)
    cols[0].metric("Generation Time", f"{metrics['generation_elapsed_ms']:.1f} ms")
    cols[1].metric("Dispatch Time", f"{metrics['dispatch_elapsed_ms']:.1f} ms")
    cols[2].metric("Execution Time", f"{metrics['execution_elapsed_ms']:.1f} ms")
    cols[3].metric("Response Time", f"{metrics['response_elapsed_ms']:.1f} ms")
    cols[4].metric("Registry Tools", str(metrics["registry_tool_count"]))

    st.subheader("Prompt")
    st.code(result["prompt"])
    with st.expander("Static Registry Prompt"):
        st.code(result["static_prompt"])

    st.subheader("Generation")
    cols = st.columns([3, 2])
    with cols[0]:
        st.code(result["generation"]["raw_output"], language="json")
    with cols[1]:
        st.json(result["dispatch"]["parse"])

    st.subheader("Blocks")
    cols = st.columns([3, 2])
    with cols[0]:
        st.markdown("Dispatch Block")
        st.code(result["dispatch"]["dispatch_block"], language="xml")
        st.markdown("Response Block")
        st.code(result["response"]["response_block"], language="xml")
    with cols[1]:
        frame = pd.DataFrame(baseline_stage_rows(result))
        st.dataframe(frame, use_container_width=True)

    st.subheader("Execution Output")
    output = result["execution"]["stdout"] or result["execution"]["stderr"] or "[no output]"
    st.code(output)


def render_comparison(result: Dict[str, Any]) -> None:
    comparison = result["comparison"]
    success_cols = st.columns(2)
    success_cols[0].metric("NTILC Success", "Yes" if comparison["success"]["ntilc"] else "No")
    success_cols[1].metric(
        "Prompt-Only Success",
        "Yes" if comparison["success"]["prompt_baseline"] else "No",
    )

    rows: List[Dict[str, Any]] = []
    for metric_name, metric_payload in comparison["metrics"].items():
        rows.append(
            {
                "metric": metric_name,
                "ntilc": metric_payload["ntilc"],
                "prompt_baseline": metric_payload["prompt_baseline"],
                "delta_ntilc_minus_baseline": metric_payload["delta"],
            }
        )
    st.subheader("Metric Comparison")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Outputs")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("NTILC")
        st.json(
            {
                "tool": comparison["outputs"]["tool"]["ntilc"],
                "command": comparison["outputs"]["command"]["ntilc"],
            }
        )
    with cols[1]:
        st.markdown("Prompt-Only")
        st.json(
            {
                "tool": comparison["outputs"]["tool"]["prompt_baseline"],
                "command": comparison["outputs"]["command"]["prompt_baseline"],
            }
        )

    cols = st.columns(2)
    with cols[0]:
        st.code(comparison["outputs"]["command"]["ntilc"], language="bash")
    with cols[1]:
        st.code(comparison["outputs"]["command"]["prompt_baseline"], language="bash")


st.title("NTILC vs Prompt-Only Tool Calling")
st.caption("Run one query both ways, inspect both traces, then compare metrics and outputs.")

with st.sidebar:
    st.header("Settings")
    query_encoder_path = st.text_input("Retrieval checkpoint", value=str(DEFAULT_RETRIEVAL_CKPT))
    lora_adapter_path = st.text_input("LoRA adapter", value=str(DEFAULT_LORA_ADAPTER))
    raw_tools_path = st.text_input("Raw tool registry", value=str(DEFAULT_RAW_TOOLS_JSON))
    qwen_model = st.text_input("Base model override", value="")
    device_choice = st.selectbox("NTILC device", options=["auto", "cpu", "cuda"], index=0)
    baseline_device_choice = st.selectbox("Baseline device", options=["auto", "cpu", "cuda"], index=0)
    max_seq_len = st.number_input("Max seq len", min_value=64, max_value=4096, value=512, step=64)
    max_new_tokens = st.number_input("NTILC max new tokens", min_value=8, max_value=512, value=96, step=8)
    baseline_max_new_tokens = st.number_input(
        "Baseline max new tokens",
        min_value=8,
        max_value=512,
        value=128,
        step=8,
    )
    temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
    top_p = st.number_input("Top p", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
    tool_timeout_seconds = st.number_input("Tool timeout", min_value=1, max_value=300, value=20, step=1)
    top_k_candidates = st.slider("Top-k candidates", min_value=1, max_value=10, value=3)
    max_retries = st.slider("Max retries", min_value=0, max_value=10, value=2)
    execute_tools = st.checkbox("Execute tools", value=False)
    if st.button("Reload cached models"):
        load_runner.clear()
        st.success("Cleared cached runner.")


query = st.text_area(
    "Query",
    value=st.session_state.get("last_query", ""),
    height=140,
    placeholder="List files and search for cuda references in the repo.",
)

run_clicked = st.button("Run comparison", type="primary", use_container_width=True)

if run_clicked:
    st.session_state["last_query"] = query
    if not query.strip():
        st.warning("Enter a query first.")
    else:
        try:
            with st.spinner("Loading models and running both traces..."):
                runner = load_runner(
                    query_encoder_path=query_encoder_path,
                    lora_adapter_path=lora_adapter_path,
                    raw_tools_path=raw_tools_path,
                    qwen_model=maybe_none(qwen_model),
                    device=None if device_choice == "auto" else device_choice,
                    baseline_device=None if baseline_device_choice == "auto" else baseline_device_choice,
                    max_seq_len=int(max_seq_len),
                    max_new_tokens=int(max_new_tokens),
                    baseline_max_new_tokens=int(baseline_max_new_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    tool_timeout_seconds=int(tool_timeout_seconds),
                )
                st.session_state["last_result"] = runner.run_comparison(
                    request=query,
                    execute_tools=execute_tools,
                    top_k_candidates=int(top_k_candidates),
                    max_retries=int(max_retries),
                )
        except Exception as exc:  # pragma: no cover
            st.exception(exc)


result = st.session_state.get("last_result")
if result:
    tabs = st.tabs(["Comparison", "NTILC Trace", "Prompt-Only Trace", "Full JSON"])
    with tabs[0]:
        render_comparison(result)
    with tabs[1]:
        render_ntilc_summary(result["ntilc"])
        render_plan(result["ntilc"])
        render_retrieval(result["ntilc"])
        render_ntilc_stage_table(result["ntilc"])
        render_attempts(result["ntilc"])
    with tabs[2]:
        render_prompt_baseline(result["prompt_baseline"])
    with tabs[3]:
        st.json(result)
