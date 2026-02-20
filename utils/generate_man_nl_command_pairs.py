#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import time
import queue
import random
import signal
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Config
# -----------------------------
INPUT_PATH = "/scratch4/home/akrik/NTILC/data/man/raw.json"
OUTPUT_JSONL_PATH = "/scratch4/home/akrik/NTILC/data/man/nl_command_pairs.jsonl"
OUTPUT_JSON_ARRAY_PATH = "/scratch4/home/akrik/NTILC/data/man/nl_command_pairs.json"

MODEL_ID = "Qwen/Qwen3-8B"

GPU_IDS = [4, 5, 6, 7]
NUM_WORKERS = len(GPU_IDS)

BATCH_SIZE = 16
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

N_EXAMPLES = 10
SEED = 42

LOG_EVERY_SEC = 10
FLUSH_EVERY_N_LINES = 50


SYSTEM = """You generate dataset examples for a tool-using assistant.

Given ONE CLI tool record, produce EXACTLY 10 diverse examples.

Each example must be a pair:
- nl_query: natural language user request (1 sentence)
- command: a realistic CLI command using the tool name and ONLY options present in the record

Rules:
- Do NOT invent options or flags not present.
- Prefer different intents (help/version/list/report/paths/modes) when supported by options.
- Commands should be concrete; if invocation requires placeholders, use realistic placeholder values.
- Output MUST be valid JSON ONLY with this exact schema:

{
  "tool": "<tool name>",
  "examples": [
    {"nl_query": "...", "command": "..."},
    ...
    (10 total)
  ]
}
"""

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def build_prompt(rec: Dict[str, Any]) -> str:
    compact = {
        "name": rec.get("name", ""),
        "one_line": rec.get("one_line", ""),
        "description": rec.get("description", ""),
        "invocation": rec.get("invocation", ""),
        "options": rec.get("options", []),
        "source_url": rec.get("source_url", ""),
    }
    return SYSTEM + "\n\nRECORD:\n" + json.dumps(compact, ensure_ascii=False)


def extract_json_obj(text: str) -> Dict[str, Any]:
    m = _JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError("No JSON object found in model output")
    return json.loads(m.group(0))


def normalize_examples(examples: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not isinstance(examples, list):
        return out
    for ex in examples[:N_EXAMPLES]:
        if not isinstance(ex, dict):
            continue
        if "nl_query" not in ex or "command" not in ex:
            continue
        out.append(
            {
                "nl_query": str(ex["nl_query"]).strip(),
                "command": str(ex["command"]).strip(),
            }
        )
    return out


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_main(
    worker_id: int,
    gpu_id: int,
    task_q: mp.Queue,
    result_q: mp.Queue,
    cfg: Dict[str, Any],
) -> None:
    """
    Each worker owns one GPU and a model instance.
    Receives tasks: (start_index, batch_records)
    Returns results: list of (index, ok, payload_or_error)
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:0")

        set_seed(cfg["seed"] + worker_id)

        log(f"worker{worker_id} init on cuda:{gpu_id} (visible cuda:0)")

        tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            torch_dtype="auto",
            device_map={"": device},
            trust_remote_code=True,
        )
        model.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        gen_kwargs = dict(
            max_new_tokens=cfg["max_new_tokens"],
            do_sample=cfg["do_sample"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        while True:
            item = task_q.get()
            if item is None:
                log(f"worker{worker_id} shutdown")
                break

            start_idx, batch = item
            prompts = [build_prompt(r) for r in batch]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg["max_input_tokens"],
            ).to(device)

            def run_generate(inp):
                with torch.no_grad():
                    out_ids = model.generate(**inp, **gen_kwargs)
                return out_ids

            try:
                out_ids = run_generate(inputs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(batch) == 1:
                    raise
                mid = len(batch) // 2
                sub_results: List[Tuple[int, bool, Any]] = []

                for sub_start, sub_batch in [(start_idx, batch[:mid]), (start_idx + mid, batch[mid:])]:
                    sub_prompts = [build_prompt(r) for r in sub_batch]
                    sub_inp = tokenizer(
                        sub_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=cfg["max_input_tokens"],
                    ).to(device)
                    out_sub = run_generate(sub_inp)
                    decoded_sub = tokenizer.batch_decode(out_sub, skip_special_tokens=True)
                    for i, rec in enumerate(sub_batch):
                        global_idx = sub_start + i
                        sub_results.append((global_idx, True, {"_raw_text": decoded_sub[i], "_rec": rec}))
                result_q.put(("BULK_RAW", worker_id, sub_results))
                continue

            decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

            results: List[Tuple[int, bool, Any]] = []
            for i, rec in enumerate(batch):
                global_idx = start_idx + i
                tool = rec.get("name", "")
                try:
                    prompt = prompts[i]
                    text = decoded[i]
                    if text.startswith(prompt):
                        gen_text = text[len(prompt):].strip()
                    else:
                        gen_text = text

                    data = extract_json_obj(gen_text)
                    exs = normalize_examples(data.get("examples"))
                    if len(exs) != cfg["n_examples"]:
                        raise ValueError(f"Expected {cfg['n_examples']} examples, got {len(exs)}")

                    payload = {
                        "tool": tool,
                        "source_url": rec.get("source_url"),
                        "examples": exs,
                    }
                    results.append((global_idx, True, payload))
                except Exception as e:
                    results.append(
                        (
                            global_idx,
                            False,
                            {
                                "tool": tool,
                                "error": str(e),
                                "traceback": traceback.format_exc(limit=3),
                            },
                        )
                    )

            result_q.put(("BULK", worker_id, results))

    except Exception as e:
        result_q.put(("FATAL", worker_id, {"error": str(e), "traceback": traceback.format_exc()}))

def run() -> None:
    records = json.loads(Path(INPUT_PATH).read_text(encoding="utf-8"))
    total = len(records)
    log(f"Loaded {total} records from {INPUT_PATH}")

    mp.set_start_method("spawn", force=True)

    task_q: mp.Queue = mp.Queue(maxsize=NUM_WORKERS * 4)
    result_q: mp.Queue = mp.Queue(maxsize=NUM_WORKERS * 8)

    cfg = dict(
        model_id=MODEL_ID,
        seed=SEED,
        batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        max_input_tokens=4096,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=DO_SAMPLE,
        n_examples=N_EXAMPLES,
    )

    workers: List[mp.Process] = []
    for wid, gid in enumerate(GPU_IDS):
        p = mp.Process(
            target=worker_main,
            args=(wid, gid, task_q, result_q, cfg),
            daemon=True,
        )
        p.start()
        workers.append(p)

    def handle_sigint(sig, frame):
        log("SIGINT received, shutting down...")
        for _ in workers:
            task_q.put(None)
        for p in workers:
            p.join(timeout=5)
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_sigint)

    next_idx = 0
    in_flight = 0

    generated_tools = 0
    generated_examples = 0
    failed_tools = 0

    last_log = time.time()
    start_time = time.time()

    out_path = Path(OUTPUT_JSONL_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open("w", encoding="utf-8")

    buffered_lines = 0

    def write_payload(payload: Dict[str, Any]) -> None:
        nonlocal buffered_lines, generated_tools, generated_examples
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        buffered_lines += 1
        generated_tools += 1
        generated_examples += len(payload["examples"])

    def write_error(err: Dict[str, Any]) -> None:
        nonlocal buffered_lines, failed_tools
        f.write(json.dumps({"_failed": True, **err}, ensure_ascii=False) + "\n")
        buffered_lines += 1
        failed_tools += 1

    while next_idx < total and in_flight < NUM_WORKERS * 2:
        batch = records[next_idx : next_idx + cfg["batch_size"]]
        task_q.put((next_idx, batch))
        in_flight += 1
        next_idx += len(batch)

    try:
        while in_flight > 0:
            try:
                kind, wid, payload = result_q.get(timeout=1.0)
            except queue.Empty:
                now = time.time()
                if now - last_log >= LOG_EVERY_SEC:
                    elapsed = now - start_time
                    rate = (generated_tools / elapsed) if elapsed > 0 else 0.0
                    log(
                        f"progress: {generated_tools+failed_tools}/{total} "
                        f"(ok={generated_tools}, fail={failed_tools}), "
                        f"in_flight={in_flight}, next_idx={next_idx}, "
                        f"tools/sec={rate:.2f}"
                    )
                    last_log = now
                continue

            if kind == "FATAL":
                log(f"❌ worker{wid} fatal: {payload['error']}\n{payload['traceback']}")
                raise RuntimeError(f"worker{wid} died")

            if kind == "BULK_RAW":
                results = payload
                for global_idx, ok, item in results:
                    rec = item["_rec"]
                    tool = rec.get("name", "")
                    try:
                        text = item["_raw_text"]
                        prompt = build_prompt(rec)
                        if text.startswith(prompt):
                            gen_text = text[len(prompt):].strip()
                        else:
                            gen_text = text
                        data = extract_json_obj(gen_text)
                        exs = normalize_examples(data.get("examples"))
                        if len(exs) != cfg["n_examples"]:
                            raise ValueError(f"Expected {cfg['n_examples']} examples, got {len(exs)}")
                        write_payload({"tool": tool, "source_url": rec.get("source_url"), "examples": exs})
                    except Exception as e:
                        write_error({"tool": tool, "error": str(e), "traceback": traceback.format_exc(limit=3)})

            elif kind == "BULK":
                results = payload
                for global_idx, ok, item in results:
                    if ok:
                        write_payload(item)
                    else:
                        write_error(item)

            else:
                log(f"⚠️ unknown message kind={kind} from worker{wid}")

            in_flight -= 1

            if buffered_lines >= FLUSH_EVERY_N_LINES:
                f.flush()
                buffered_lines = 0

            while next_idx < total and in_flight < NUM_WORKERS * 2:
                batch = records[next_idx : next_idx + cfg["batch_size"]]
                task_q.put((next_idx, batch))
                in_flight += 1
                next_idx += len(batch)

            now = time.time()
            if now - last_log >= LOG_EVERY_SEC:
                elapsed = now - start_time
                rate = (generated_tools / elapsed) if elapsed > 0 else 0.0
                log(
                    f"progress: {generated_tools+failed_tools}/{total} "
                    f"(ok={generated_tools}, fail={failed_tools}), "
                    f"in_flight={in_flight}, next_idx={next_idx}, "
                    f"tools/sec={rate:.2f}"
                )
                last_log = now

    finally:
        for _ in workers:
            task_q.put(None)
        for p in workers:
            p.join(timeout=10)
        f.flush()
        f.close()

    elapsed = time.time() - start_time
    log(
        f"Done. ok_tools={generated_tools}, failed_tools={failed_tools}, "
        f"total_examples={generated_examples}, elapsed_sec={elapsed:.1f}"
    )
    log(f"Saved JSONL to: {OUTPUT_JSONL_PATH}")

    try:
        arr_out = Path(OUTPUT_JSON_ARRAY_PATH)
        with open(OUTPUT_JSONL_PATH, "r", encoding="utf-8") as fin, open(arr_out, "w", encoding="utf-8") as fout:
            fout.write("[\n")
            first = True
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                if not first:
                    fout.write(",\n")
                fout.write(line)
                first = False
            fout.write("\n]\n")
        log(f"Also saved JSON array to: {OUTPUT_JSON_ARRAY_PATH}")
    except Exception as e:
        log(f"⚠️ could not convert JSONL->JSON array: {e}")


if __name__ == "__main__":
    run()
