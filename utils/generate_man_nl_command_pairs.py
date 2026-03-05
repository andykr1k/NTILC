#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate natural language ↔ CLI command pairs from man-page JSON records
using Qwen model in a multi-GPU batched inference setup.

Shows progress with tqdm.

Fixes applied:
  1. Added `import queue` — was missing, caused NameError on queue.Empty
  2. `buffered` counter now incremented inside the loop so FLUSH_EVERY works
  3. Task enqueueing moved to a background thread to avoid deadlock while
     workers are still loading the model (queue fillup would block main before
     the result-collection loop even started)
  4. Workers only receive one None sentinel (sent from main thread after all
     results are collected); the signal handler no longer double-sends them
"""

import json
import os
import queue
import re
import sys
import time
import random
import signal
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import multiprocessing as mp
from multiprocessing import Queue, Process

from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
#   Configuration
# ──────────────────────────────────────────────────────────────────────────────

INPUT_JSON        = Path("/scratch4/home/akrik/NTILC/data/man/raw_ai.json")
OUTPUT_JSONL      = Path("/scratch4/home/akrik/NTILC/data/man/nl_command_pairs.jsonl")
OUTPUT_JSON_ARRAY = Path("/scratch4/home/akrik/NTILC/data/man/nl_command_pairs.json")

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

GPUS         = [0, 1, 2, 3]
NUM_WORKERS  = len(GPUS)

BATCH_SIZE      = 8
MAX_NEW_TOKENS  = 2048
TEMPERATURE     = 0.12
TOP_P           = 0.82
DO_SAMPLE       = True

TARGET_EXAMPLES = 25
MIN_EXAMPLES    = 18

SEED_BASE       = 42

FLUSH_EVERY     = 40
MAX_RETRIES     = 2

# ──────────────────────────────────────────────────────────────────────────────
#   Prompt – strict format enforcement
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM = """You are a strict JSON generator. Output ONLY valid JSON — nothing else.

Forbidden:
- thinking steps
- <think>
- reasoning
- explanations
- markdown
- ```json
- prose
- "Here is"
- trailing text

You MUST output exactly this structure:

{"tool":"name","examples":[{"nl_query":"short natural language request","command":"realistic command using ONLY flags from RECORD"}, ... exactly 25 items]}

Rules:
- exactly 25 examples
- nl_query = one short sentence
- command = realistic CLI call using ONLY documented flags/options
- do NOT invent flags
- no extra text before/after JSON

Correct example (do NOT copy — create your own):

{"tool":"ls","examples":[{"nl_query":"List all files including hidden.","command":"ls -a"},{"nl_query":"Show detailed list with human sizes.","command":"ls -lh"}, ... 23 more ...]}

Generate for this RECORD now.
"""


def make_prompt(record: dict, tokenizer: AutoTokenizer) -> str:
    compact = {
        k: record[k]
        for k in ["name", "one_line", "description", "invocation", "options"]
        if k in record
    }
    user_text = "RECORD:\n" + json.dumps(compact, separators=(",", ":"), ensure_ascii=False)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user_text},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return SYSTEM + "\n\n" + user_text + "\n"


# ──────────────────────────────────────────────────────────────────────────────
#   JSON extraction – tolerant to common model mistakes
# ──────────────────────────────────────────────────────────────────────────────

def extract_json(raw: str) -> Optional[dict]:
    text = re.sub(
        r"(?is)(<think>|思考过程|reasoning|assistant\s*think).*?(?=\{|$)",
        "",
        raw,
        flags=re.DOTALL,
    )
    text = re.sub(r"(?is)^(Here is|```json|```)\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    def find_largest_dict(s: str) -> Optional[dict]:
        best, best_count = None, -1
        pos = 0
        decoder = json.JSONDecoder()
        while pos < len(s):
            if s[pos] not in "{[":
                pos += 1
                continue
            try:
                obj, end = decoder.raw_decode(s[pos:])
                if isinstance(obj, dict):
                    ex = obj.get("examples", [])
                    if isinstance(ex, list) and len(ex) > best_count:
                        best, best_count = obj, len(ex)
                pos += end
            except json.JSONDecodeError:
                pos += 1
        return best

    candidate = find_largest_dict(text)
    if candidate and isinstance(candidate.get("examples"), list):
        return candidate

    # Fallback: regex extraction
    tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', text)
    pairs = re.findall(
        r'\{\s*"nl_query"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"command"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        text,
    )
    if pairs and len(pairs) >= MIN_EXAMPLES // 2:
        return {
            "tool": tool_match.group(1) if tool_match else "unknown",
            "examples": [{"nl_query": q, "command": c} for q, c in pairs],
        }

    return None


def clean_examples(raw_examples: Any) -> List[Dict[str, str]]:
    if not isinstance(raw_examples, list):
        return []
    cleaned = []
    for item in raw_examples:
        if not isinstance(item, dict):
            continue
        q = str(item.get("nl_query", "")).strip()
        c = str(item.get("command", "")).strip()
        if q and c:
            cleaned.append({"nl_query": q, "command": c})
            if len(cleaned) == TARGET_EXAMPLES:
                break
    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
#   Worker process
# ──────────────────────────────────────────────────────────────────────────────

def worker(
    worker_id: int,
    gpu_id: int,
    task_q: Queue,
    result_q: Queue,
    cfg: dict,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")

    torch.manual_seed(cfg["seed"] + worker_id * 10007)
    random.seed(cfg["seed"] + worker_id * 10007)

    print(f"[W{worker_id}] Loading on cuda:{gpu_id} ...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    stop_tokens = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        tokenizer.convert_tokens_to_ids("<|end|>"),
    ]
    stop_tokens = [t for t in stop_tokens if t is not None]

    gen_config = {
        "max_new_tokens": cfg["max_new_tokens"],
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "do_sample": cfg["do_sample"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": stop_tokens,
    }

    print(f"[W{worker_id}] Ready", flush=True)

    while True:
        task = task_q.get()
        if task is None:
            break

        start_idx, records = task
        batch_results = []

        prompts = [make_prompt(r, tokenizer) for r in records]

        try:
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(device)

            prompt_lengths = inputs.attention_mask.sum(dim=1).tolist()

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **gen_config)

            for i, rec in enumerate(records):
                idx = start_idx + i
                tool = rec.get("name", "unknown")
                gen_ids = generated_ids[i, prompt_lengths[i]:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                data = extract_json(text)
                if not data:
                    raise ValueError("No JSON parsed")

                examples = clean_examples(data.get("examples"))
                n = len(examples)

                if n < MIN_EXAMPLES:
                    raise ValueError(f"Only {n} usable examples (min {MIN_EXAMPLES})")

                payload = {
                    "tool": tool,
                    "source_url": rec.get("source_url"),
                    "examples": examples[:TARGET_EXAMPLES],
                }
                batch_results.append((idx, True, payload))

        except Exception as batch_exc:
            # Retry each record individually if the batch failed
            for i, rec in enumerate(records):
                idx = start_idx + i
                tool = rec.get("name", "unknown")
                raw_preview = ""
                success = False

                for attempt in range(MAX_RETRIES):
                    try:
                        single_prompt = make_prompt(rec, tokenizer)
                        single_inputs = tokenizer(
                            [single_prompt], return_tensors="pt", padding=True
                        ).to(device)

                        with torch.inference_mode():
                            out = model.generate(**single_inputs, **gen_config)

                        gen_ids = out[0, single_inputs.input_ids.size(1):]
                        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                        raw_preview = text[:500]

                        data = extract_json(text)
                        if not data:
                            continue

                        examples = clean_examples(data.get("examples"))
                        if len(examples) >= MIN_EXAMPLES:
                            payload = {
                                "tool": tool,
                                "source_url": rec.get("source_url"),
                                "examples": examples[:TARGET_EXAMPLES],
                            }
                            batch_results.append((idx, True, payload))
                            success = True
                            break

                    except Exception:
                        continue

                if not success:
                    batch_results.append((idx, False, {
                        "tool": tool,
                        "error": str(batch_exc),
                        "traceback": traceback.format_exc(limit=3),
                        "raw_preview": raw_preview,
                    }))

        result_q.put(("BULK", worker_id, batch_results))

    print(f"[W{worker_id}] Finished", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
#   Background enqueue thread — prevents deadlock during model loading
# ──────────────────────────────────────────────────────────────────────────────

def enqueue_tasks(records: list, task_q: Queue, batch_size: int):
    """Push all batches into task_q from a background thread."""
    i = 0
    while i < len(records):
        batch = records[i : i + batch_size]
        task_q.put((i, batch))
        i += len(batch)


# ──────────────────────────────────────────────────────────────────────────────
#   Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if not INPUT_JSON.is_file():
        print(f"Input file not found: {INPUT_JSON}", file=sys.stderr)
        sys.exit(1)

    records = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        print("Input JSON must be a list of records", file=sys.stderr)
        sys.exit(1)

    total_records = len(records)
    print(f"Loaded {total_records} records", flush=True)

    mp.set_start_method("spawn", force=True)

    # FIX #4: queue large enough that the producer thread never blocks before
    # the result-consumer loop is running.  Workers may still be loading their
    # models when tasks are enqueued, so we allow the whole queue to fill.
    task_q   = mp.Queue(maxsize=0)   # unbounded — producer thread fills freely
    result_q = mp.Queue(maxsize=NUM_WORKERS * 10)

    cfg = {
        "model":          MODEL_NAME,
        "seed":           SEED_BASE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature":    TEMPERATURE,
        "top_p":          TOP_P,
        "do_sample":      DO_SAMPLE,
    }

    workers = []
    for wid, gid in enumerate(GPUS):
        p = Process(
            target=worker,
            args=(wid, gid, task_q, result_q, cfg),
            name=f"worker-{wid}",
        )
        p.start()
        workers.append(p)

    # FIX #3: track whether shutdown was already triggered so we only send
    # sentinel None values once.
    _shutdown_called = threading.Event()

    def shutdown(signum=None, frame=None):
        if _shutdown_called.is_set():
            return
        _shutdown_called.set()
        print("\nCaught interrupt → shutting down...", flush=True)
        for _ in workers:
            task_q.put(None)
        for p in workers:
            p.join(timeout=8)
            if p.is_alive():
                p.terminate()
                p.join(timeout=3)
        sys.exit(1)

    signal.signal(signal.SIGINT, shutdown)

    # FIX #4: enqueue in a background thread so main can immediately start
    # collecting results — workers load models in parallel with task production.
    producer = threading.Thread(
        target=enqueue_tasks,
        args=(records, task_q, BATCH_SIZE),
        daemon=True,
    )
    producer.start()

    successes  = 0
    failures   = 0
    start_time = time.time()
    # FIX #2: buffered is now correctly incremented inside the loop
    buffered   = 0

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f_out, \
         tqdm(total=total_records, desc="Generating", unit="tool", dynamic_ncols=True) as pbar:

        done = 0
        while done < total_records:
            try:
                # FIX #1: `queue` module is now imported — queue.Empty works
                typ, wid, items = result_q.get(timeout=2.0)
            except queue.Empty:   # ← was NameError before
                continue

            if typ == "BULK":
                for _, ok, payload in items:
                    if ok:
                        json.dump(payload, f_out, ensure_ascii=False)
                        f_out.write("\n")
                        successes += 1
                    else:
                        err = {"_failed": True, **payload}
                        json.dump(err, f_out, ensure_ascii=False)
                        f_out.write("\n")
                        failures += 1

                    done     += 1
                    buffered += 1   # FIX #2: was never incremented before
                    pbar.update(1)

                    pbar.set_postfix(
                        {
                            "ok":    successes,
                            "fail":  failures,
                            "ok%":   f"{successes / (successes + failures) * 100:.1f}%"
                                     if (successes + failures) > 0 else "0%",
                            "speed": f"{done / (time.time() - start_time):.1f}/s",
                        },
                        refresh=True,
                    )

                if buffered >= FLUSH_EVERY:
                    f_out.flush()
                    buffered = 0

    producer.join()

    # FIX #3: send sentinels only if shutdown hasn't already done so
    if not _shutdown_called.is_set():
        for _ in workers:
            task_q.put(None)

    for p in workers:
        p.join(timeout=10)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Success: {successes}  |  Failed: {failures}  |  Total: {total_records}")
    print(f"Output saved → {OUTPUT_JSONL}")

    # Convert to pretty JSON array
    try:
        with OUTPUT_JSONL.open("r", encoding="utf-8") as fin, \
             OUTPUT_JSON_ARRAY.open("w", encoding="utf-8") as fout:
            fout.write("[\n")
            lines = [line.strip() for line in fin if line.strip()]
            for idx, line in enumerate(lines):
                fout.write(line)
                fout.write(",\n" if idx < len(lines) - 1 else "\n")
            fout.write("]\n")
        print(f"Also saved array format → {OUTPUT_JSON_ARRAY}")
    except Exception as e:
        print(f"Could not create JSON array: {e}")


if __name__ == "__main__":
    main()