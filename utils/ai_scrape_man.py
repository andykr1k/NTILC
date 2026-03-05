#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scrape_man_pages.py
───────────────────
Scrapes Linux man-page section-1 entries from man7.org, then uses a vLLM
instruct model to extract structured JSON (name, synopsis, options, …).
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import aiohttp
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

INDEX_URL                = "https://man7.org/linux/man-pages/dir_all_alphabetic.html"
OUTPUT_JSONL_PATH        = Path("/scratch4/home/akrik/NTILC/data/man/raw_ai.jsonl")
ERRORS_PATH              = Path("/scratch4/home/akrik/NTILC/data/man/raw_errors_ai.json")
MODEL_ID                 = "Qwen/Qwen3-30B-A3B-Instruct-2507"

CUDA_VISIBLE_DEVICES     = "0,1,2,3"
TENSOR_PARALLEL_SIZE     = 4

MAX_PAGES: Optional[int] = None
REQUEST_TIMEOUT          = 20.0
HTTP_CONCURRENCY         = 32
MAX_RETRIES              = 3

TEMP                     = 0.0
MAX_NEW_TOKENS           = 4096
MAX_NEW_TOKENS_RETRY     = 8192          
MAX_MODEL_LEN            = 194400
INFERENCE_BATCH_SIZE     = 16

RESUME_FROM_DISK         = True
FLUSH_EVERY_N_BATCHES    = 1

SECTION_1_PATTERN = re.compile(r"\(1\)$")
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128 Safari/537.36"
    )
}

# Set before vLLM imports CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """\
You extract structured JSON from a Linux man-page.
Return strict JSON only — no markdown fences, no explanation, no preamble.
Use this schema exactly:

{
  "name": "string",
  "one_line": "string",
  "description": "string",
  "invocation": "string",
  "options": [
    {
      "flags": ["string"],
      "arg": "string",
      "description": "string"
    }
  ]
}

If a field is unknown use an empty string or empty list.
Only include options that are actually present in the text.
"""

EXAMPLE_OUTPUT = {
    "name": "ls",
    "one_line": "list directory contents",
    "description": "List information about files.",
    "invocation": "ls [OPTION]... [FILE]...",
    "options": [
        {
            "flags": ["-a", "--all"],
            "arg": "",
            "description": "do not ignore entries starting with .",
        }
    ],
}

EXAMPLE_OUTPUT_STR = json.dumps(EXAMPLE_OUTPUT, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# HTML → plain text
# ──────────────────────────────────────────────

def extract_page_text(html: str) -> str:
    """
    Strip man7.org boilerplate and return plain text of the content area.
    Falls back to <body> if the expected content div is absent.
    Cuts token count by ~60 % vs raw HTML.
    """
    soup = BeautifulSoup(html, "html.parser")

    content = (
        soup.find("div", {"id": "content"})
        or soup.find("div", {"class": "sect1"})
        or soup.find("body")
        or soup
    )

    for tag in content.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    return content.get_text(separator="\n", strip=True)


# ──────────────────────────────────────────────
# Chat prompt builder
# ──────────────────────────────────────────────

def build_chat_prompt(tokenizer, *, source_url: str, page_text: str) -> str:
    """
    Format messages with the model's chat template so instruct models
    receive proper role markers (<|im_start|> for Qwen, etc.).
    """
    user_content = (
        f"EXAMPLE OUTPUT:\n{EXAMPLE_OUTPUT_STR}\n\n"
        f"SOURCE URL:\n{source_url}\n\n"
        f"MAN PAGE TEXT:\n{page_text}"
    )
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ──────────────────────────────────────────────
# JSON extraction + truncation detection
# ──────────────────────────────────────────────

def is_truncated(raw: str) -> bool:
    """
    Heuristic: output was cut off before the closing brace of the root object.
    Handles trailing whitespace and markdown fence remnants.
    """
    stripped = re.sub(r"```\s*$", "", raw.strip()).rstrip()
    return not stripped.endswith("}")


def parse_json_output(raw: str) -> dict:
    """
    Strip optional markdown fences then extract the outermost { … } pair.

    On JSONDecodeError (most often caused by truncation), attempts recovery
    by trimming to the last fully-closed option object and appending the
    minimal closing tokens `]\n}` to produce valid JSON.
    """
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0 or end <= start:
        raise ValueError("No JSON object found in model output")

    json_str = raw[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Recovery: find the last fully-closed option object `},` and close
        # the options array + root object.  This preserves all complete
        # options even when the final one is cut off mid-description.
        last_complete = json_str.rfind('},\n    {')
        if last_complete != -1:
            truncated = json_str[:last_complete + 1] + "\n  ]\n}"
            try:
                recovered = json.loads(truncated)
                log.debug("Recovered partial JSON for truncated output")
                return recovered
            except json.JSONDecodeError:
                pass
        raise


# ──────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────

def run_inference_batches(
    llm: LLM,
    sampling_params: SamplingParams,
    prompt_pairs: List[Tuple[str, str]],
    out_file,
    errors: List[dict],
) -> Tuple[int, int]:
    """
    Iterate over prompt_pairs in batches, generate, parse, write.

    Returns (total_processed, total_parse_errors).
    Appends to `errors` in-place, tagging each failure with an
    `error_type` of either "truncated" or "parse_error" so the caller
    can schedule a targeted retry pass.
    """
    total_processed   = 0
    total_parse_errors = 0
    n_batches = (len(prompt_pairs) + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE

    for batch_i, batch_start in enumerate(
        tqdm(range(0, len(prompt_pairs), INFERENCE_BATCH_SIZE), desc="Inference")
    ):
        batch         = prompt_pairs[batch_start: batch_start + INFERENCE_BATCH_SIZE]
        urls_batch    = [u for u, _ in batch]
        prompts_batch = [p for _, p in batch]

        t0      = time.time()
        outputs = llm.generate(prompts_batch, sampling_params)
        elapsed = time.time() - t0
        log.info(
            f"Batch {batch_i + 1}/{n_batches}: "
            f"{len(batch)} pages in {elapsed:.1f}s "
            f"({elapsed / len(batch):.2f}s/page)"
        )

        for url, output in zip(urls_batch, outputs):
            raw_text = output.outputs[0].text
            try:
                parsed = parse_json_output(raw_text)
                parsed["source_url"] = url
                out_file.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                total_processed += 1
            except Exception as exc:
                total_parse_errors += 1
                error_type = "truncated" if is_truncated(raw_text) else "parse_error"
                errors.append({
                    "url":        url,
                    "error":      str(exc),
                    "error_type": error_type,
                    "raw":        raw_text[:2000],
                })

        if (batch_i + 1) % FLUSH_EVERY_N_BATCHES == 0:
            out_file.flush()
            os.fsync(out_file.fileno())

    return total_processed, total_parse_errors


def retry_truncated(
    llm: LLM,
    tokenizer,
    errors: List[dict],
    page_text_map: Dict[str, str],
    out_file,
) -> Tuple[int, int]:
    """
    Second pass: re-run URLs that failed with error_type="truncated" using
    MAX_NEW_TOKENS_RETRY.  Successful results are written to out_file and
    the error entry is mutated to error_type="retry_ok" so the final error
    file is accurate.  Returns (recovered, still_failed).
    """
    truncated_errors = [e for e in errors if e.get("error_type") == "truncated"]
    if not truncated_errors:
        log.info("No truncated outputs to retry.")
        return 0, 0

    log.info(f"Retrying {len(truncated_errors)} truncated outputs with max_tokens={MAX_NEW_TOKENS_RETRY} …")

    retry_sampling = SamplingParams(temperature=TEMP, max_tokens=MAX_NEW_TOKENS_RETRY)
    MAX_PROMPT_TOKENS_RETRY = MAX_MODEL_LEN - MAX_NEW_TOKENS_RETRY

    retry_pairs: List[Tuple[str, str]] = []
    for entry in truncated_errors:
        url = entry["url"]
        page_text = page_text_map.get(url)
        if page_text is None:
            log.warning(f"[retry] No cached page text for {url} — skipping")
            continue
        prompt = build_chat_prompt(tokenizer, source_url=url, page_text=page_text)
        n_tokens = len(tokenizer.encode(prompt))
        if n_tokens > MAX_PROMPT_TOKENS_RETRY:
            log.warning(f"[retry-skip] {url} — {n_tokens:,} tokens still exceeds retry limit")
            continue
        retry_pairs.append((url, prompt))

    if not retry_pairs:
        return 0, len(truncated_errors)

    recovered    = 0
    still_failed = 0
    retry_url_set = {u for u, _ in retry_pairs}

    n_batches = (len(retry_pairs) + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE
    for batch_i, batch_start in enumerate(
        tqdm(range(0, len(retry_pairs), INFERENCE_BATCH_SIZE), desc="Retry inference")
    ):
        batch         = retry_pairs[batch_start: batch_start + INFERENCE_BATCH_SIZE]
        urls_batch    = [u for u, _ in batch]
        prompts_batch = [p for _, p in batch]

        outputs = llm.generate(prompts_batch, retry_sampling)

        for url, output in zip(urls_batch, outputs):
            raw_text = output.outputs[0].text
            try:
                parsed = parse_json_output(raw_text)
                parsed["source_url"] = url
                out_file.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                # Mark the original error entry as recovered.
                for entry in errors:
                    if entry.get("url") == url and entry.get("error_type") == "truncated":
                        entry["error_type"] = "retry_ok"
                        break
                recovered += 1
            except Exception as exc:
                still_failed += 1
                error_type = "truncated_retry" if is_truncated(raw_text) else "parse_error_retry"
                for entry in errors:
                    if entry.get("url") == url and entry.get("error_type") == "truncated":
                        entry["error_type"] = error_type
                        entry["retry_error"] = str(exc)
                        break

        out_file.flush()
        os.fsync(out_file.fileno())

    log.info(f"Retry pass: recovered={recovered}, still_failed={still_failed}")
    return recovered, still_failed


# ──────────────────────────────────────────────
# Async HTTP fetcher with retry / back-off
# ──────────────────────────────────────────────

async def fetch_one(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    retries: int = MAX_RETRIES,
) -> Tuple[str, Optional[str]]:
    """
    Fetch a single URL.  Retries on 429 / 5xx with exponential back-off.
    Returns (url, html_text) or (url, None) on permanent failure.
    """
    async with semaphore:
        for attempt in range(retries):
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        wait = 2 ** attempt
                        log.warning(f"HTTP {resp.status} for {url} — retrying in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    return url, await resp.text()
            except asyncio.TimeoutError:
                log.warning(f"Timeout ({attempt+1}/{retries}) for {url}")
                await asyncio.sleep(2 ** attempt)
            except aiohttp.ClientError as exc:
                log.warning(f"Client error for {url}: {exc}")
                break
        log.error(f"Giving up on {url} after {retries} attempts")
        return url, None


async def fetch_all(urls: List[str]) -> Dict[str, Optional[str]]:
    semaphore = asyncio.Semaphore(HTTP_CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=HTTP_CONCURRENCY, ssl=False)
    async with aiohttp.ClientSession(
        headers=DEFAULT_HEADERS, connector=connector
    ) as session:
        tasks = [fetch_one(session, url, semaphore) for url in urls]
        results: Dict[str, Optional[str]] = {}
        for coro in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Fetching HTML"
        ):
            url, html = await coro
            results[url] = html
    return results


# ──────────────────────────────────────────────
# Resume helper
# ──────────────────────────────────────────────

def load_done_urls(path: Path) -> set:
    done: set = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "source_url" in obj:
                    done.add(obj["source_url"])
            except json.JSONDecodeError:
                pass
    return done


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:

    # ── Step 1: collect links ─────────────────────────────────────────
    log.info("Fetching index …")
    resp = requests.get(INDEX_URL, timeout=30, headers=DEFAULT_HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    all_links: List[str] = sorted({
        urljoin(INDEX_URL, a["href"])
        for a in soup.find_all("a")
        if (
            a.get_text(strip=True)
            and a.get("href")
            and SECTION_1_PATTERN.search(a.get_text(strip=True))
        )
    })

    if MAX_PAGES:
        all_links = all_links[:MAX_PAGES]

    log.info(f"Found {len(all_links)} section-1 links")

    # ── Step 2: resume ────────────────────────────────────────────────
    done_urls = load_done_urls(OUTPUT_JSONL_PATH) if RESUME_FROM_DISK else set()
    todo_links = [u for u in all_links if u not in done_urls]
    log.info(f"{len(done_urls)} already done — {len(todo_links)} remaining")

    if not todo_links:
        log.info("Nothing to do.")
        return

    # ── Step 3: fetch HTML ────────────────────────────────────────────
    loop = asyncio.get_event_loop()
    html_map: Dict[str, Optional[str]] = loop.run_until_complete(fetch_all(todo_links))

    fetched_ok = sum(v is not None for v in html_map.values())
    log.info(f"Fetched {fetched_ok}/{len(todo_links)} pages successfully")

    # ── Step 4: strip HTML → plain text, accumulate errors ───────────
    errors: List[dict] = []
    page_pairs: List[Tuple[str, str]] = []   # (url, plain_text)
    # Keyed by URL so the retry pass can rebuild prompts without re-fetching.
    page_text_map: Dict[str, str] = {}

    for url in todo_links:
        html = html_map.get(url)
        if html is None:
            errors.append({"url": url, "error": "fetch_failed", "error_type": "fetch_failed"})
            continue
        try:
            page_text = extract_page_text(html)
        except Exception as exc:
            errors.append({
                "url":        url,
                "error":      f"html_parse_failed: {exc}",
                "error_type": "html_parse_failed",
            })
            continue
        page_pairs.append((url, page_text))
        page_text_map[url] = page_text

    log.info(f"Built {len(page_pairs)} page texts — {len(errors)} fetch/parse errors so far")

    # ── Step 5: load model ────────────────────────────────────────────
    log.info("Loading vLLM model …")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        enable_chunked_prefill=True,
        max_num_batched_tokens=32768,
        max_model_len=MAX_MODEL_LEN,
    )

    sampling_params = SamplingParams(
        temperature=TEMP,
        max_tokens=MAX_NEW_TOKENS,
    )

    tokenizer = llm.get_tokenizer()

    # ── Step 6: build chat-formatted prompts + token-length guard ─────
    MAX_PROMPT_TOKENS = MAX_MODEL_LEN - MAX_NEW_TOKENS

    prompt_pairs: List[Tuple[str, str]] = []   # (url, formatted_prompt)

    for url, page_text in tqdm(page_pairs, desc="Formatting prompts"):
        prompt = build_chat_prompt(tokenizer, source_url=url, page_text=page_text)
        n_tokens = len(tokenizer.encode(prompt))
        if n_tokens > MAX_PROMPT_TOKENS:
            log.warning(f"[skip] {url} — {n_tokens:,} tokens > limit {MAX_PROMPT_TOKENS:,}")
            errors.append({
                "url":        url,
                "error":      f"prompt_too_long_{n_tokens}_tokens",
                "error_type": "prompt_too_long",
            })
        else:
            prompt_pairs.append((url, prompt))

    log.info(f"{len(prompt_pairs)} within token limit, {len(errors)} errors total so far")

    # ── Step 7: inference + save ──────────────────────────────────────
    OUTPUT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSONL_PATH.open("a", encoding="utf-8") as out_file:
        total_processed, total_parse_errors = run_inference_batches(
            llm, sampling_params, prompt_pairs, out_file, errors
        )

        # ── Step 7b: retry truncated outputs ─────────────────────────
        recovered, still_failed = retry_truncated(
            llm, tokenizer, errors, page_text_map, out_file
        )
        total_processed    += recovered
        total_parse_errors -= recovered  # those are now successes

        out_file.flush()
        os.fsync(out_file.fileno())

    # ── Step 8: write errors ──────────────────────────────────────────
    with ERRORS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(errors, fh, ensure_ascii=False, indent=2)

    log.info(
        f"\nDone. processed={total_processed}, "
        f"parse_errors={total_parse_errors}, "
        f"total_errors={len(errors)}, "
        f"truncation_retries_recovered={recovered}"
    )
    log.info(f"Results → {OUTPUT_JSONL_PATH}")
    log.info(f"Errors  → {ERRORS_PATH}")


if __name__ == "__main__":
    main()