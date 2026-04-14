from __future__ import annotations

import ast
import csv
import io
import json
import math
import mimetypes
import os
import re
import shlex
import shutil
import smtplib
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from ddgs import DDGS

import requests


MEMORY_FILE = Path("tool_memory.json")


class ToolExecutionError(Exception):
    pass


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_memory_store(memory_file: Path = MEMORY_FILE) -> Dict[str, str]:
    if not memory_file.exists():
        return {}
    try:
        with memory_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ToolExecutionError("Memory store is not a valid JSON object.")
        return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError as e:
        raise ToolExecutionError(f"Failed to parse memory file: {e}") from e


def _save_memory_store(data: Dict[str, str], memory_file: Path = MEMORY_FILE) -> None:
    _ensure_parent_dir(memory_file)
    with memory_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# information_retrieval
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search the web using DuckDuckGo Instant Answer API."""
    if not query.strip():
        raise ToolExecutionError("Query must not be empty.")

    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=max_results))
    results = [{"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in raw]
    return {"query": query, "results": results, "count": len(results)}


def fetch_url(url: str, format: str = "markdown") -> Dict[str, Any]:
    """Fetch and parse the content of a webpage by URL."""
    if not url.strip():
        raise ToolExecutionError("URL must not be empty.")
    valid_formats = {"text", "markdown", "html"}
    if format not in valid_formats:
        raise ToolExecutionError(f"format must be one of {valid_formats}.")

    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except requests.RequestException as e:
        raise ToolExecutionError(f"Failed to fetch URL: {e}") from e

    content_type = response.headers.get("Content-Type", "")
    raw_text = response.text

    if format == "html":
        body = raw_text
    elif format == "text":
        # Strip HTML tags simply
        body = re.sub(r"<[^>]+>", "", raw_text)
        body = re.sub(r"\s{2,}", " ", body).strip()
    else:  # markdown — best-effort conversion
        # Remove script/style blocks
        body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", "", raw_text)
        # Convert headings
        for level in range(6, 0, -1):
            body = re.sub(
                rf"(?i)<h{level}[^>]*>(.*?)</h{level}>",
                lambda m, lv=level: "\n" + "#" * lv + " " + m.group(1).strip() + "\n",
                body,
            )
        # Convert links
        body = re.sub(r'(?i)<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r"[\2](\1)", body)
        # Convert paragraphs and breaks
        body = re.sub(r"(?i)<br\s*/?>", "\n", body)
        body = re.sub(r"(?i)<p[^>]*>", "\n\n", body)
        # Strip remaining tags
        body = re.sub(r"<[^>]+>", "", body)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()

    return {
        "url": url,
        "format": format,
        "content_type": content_type,
        "content": body,
        "length": len(body),
    }


def search_knowledge_base(
    query: str, top_k: int = 5, namespace: Optional[str] = None
) -> Dict[str, Any]:
    """
    Semantic search over a private document store / embeddings index.

    This implementation uses simple TF-style keyword matching over a local
    JSON knowledge-base file (knowledge_base.json) when no vector store is
    configured.  Replace the body with your vector-store client (Pinecone,
    Weaviate, Chroma, etc.) as needed.
    """
    if not query.strip():
        raise ToolExecutionError("Query must not be empty.")

    kb_path = Path("knowledge_base.json")
    if not kb_path.exists():
        return {
            "query": query,
            "namespace": namespace,
            "results": [],
            "note": "No knowledge_base.json found. Populate it or connect a vector store.",
        }

    try:
        documents: List[Dict[str, Any]] = json.loads(kb_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise ToolExecutionError(f"Failed to load knowledge base: {e}") from e

    if namespace:
        documents = [d for d in documents if d.get("namespace") == namespace]

    query_terms = set(re.findall(r"\w+", query.lower()))

    def score(doc: Dict[str, Any]) -> float:
        text = " ".join(str(v) for v in doc.values()).lower()
        tokens = set(re.findall(r"\w+", text))
        return len(query_terms & tokens) / max(len(query_terms), 1)

    ranked = sorted(documents, key=score, reverse=True)[:top_k]
    return {"query": query, "namespace": namespace, "results": ranked, "count": len(ranked)}


# ---------------------------------------------------------------------------
# computation
# ---------------------------------------------------------------------------

def run_python(code: str) -> Dict[str, Any]:
    """Execute Python code."""
    if not code.strip():
        raise ToolExecutionError("Code must not be empty.")

    stdout_buffer = io.StringIO()
    local_ns: Dict[str, Any] = {}

    try:
        with redirect_stdout(stdout_buffer):
            exec(code, {"__builtins__": __builtins__}, local_ns)  # noqa: S102
    except Exception as e:
        raise ToolExecutionError(f"Python execution failed: {e}") from e

    return {
        "stdout": stdout_buffer.getvalue(),
        "locals": {k: repr(v) for k, v in local_ns.items() if not k.startswith("__")},
    }


def run_javascript(code: str) -> Dict[str, Any]:
    """Execute JavaScript via Node.js."""
    if not code.strip():
        raise ToolExecutionError("Code must not be empty.")

    node_path = shutil.which("node") or shutil.which("nodejs")
    if not node_path:
        raise ToolExecutionError("Node.js is not installed or not in PATH.")

    try:
        completed = subprocess.run(
            [node_path, "--input-type=module"],
            input=code,
            text=True,
            capture_output=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as e:
        raise ToolExecutionError(f"JavaScript execution timed out: {e}") from e
    except Exception as e:
        raise ToolExecutionError(f"JavaScript execution failed: {e}") from e

    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_sql(query: str, connection: str = "") -> Dict[str, Any]:
    """
    Execute a SQL query against a connected database.

    Supports SQLite file paths (e.g. 'mydb.sqlite') and connection strings
    starting with 'sqlite://'.  For other databases install the appropriate
    driver (psycopg2, pymysql, etc.) and extend the factory below.
    """
    if not query.strip():
        raise ToolExecutionError("Query must not be empty.")

    try:
        import sqlite3

        db_path = connection.replace("sqlite:///", "").replace("sqlite://", "") or ":memory:"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description] if cursor.description else []
        conn.commit()
        conn.close()
    except Exception as e:
        raise ToolExecutionError(f"SQL execution failed: {e}") from e

    return {
        "query": query,
        "connection": connection or ":memory:",
        "columns": columns,
        "rows": rows,
        "rowcount": len(rows),
    }


def run_notebook_cell(
    notebook_path: str,
    cell_index: int = 0,
    code: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a cell in a Jupyter notebook and return output."""
    nb_path = Path(notebook_path)
    if not nb_path.exists():
        raise ToolExecutionError(f"Notebook not found: {notebook_path}")

    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise ToolExecutionError(f"Failed to load notebook: {e}") from e

    cells = nb.get("cells", [])
    if cell_index < 0 or cell_index >= len(cells):
        raise ToolExecutionError(
            f"cell_index {cell_index} out of range (notebook has {len(cells)} cells)."
        )

    cell = cells[cell_index]
    cell_source = code if code is not None else "".join(cell.get("source", []))

    if not cell_source.strip():
        raise ToolExecutionError("Cell source is empty.")

    # Execute the cell source as Python
    result = run_python(cell_source)
    return {
        "notebook_path": notebook_path,
        "cell_index": cell_index,
        "source": cell_source,
        **result,
    }


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ToolExecutionError("Only numeric constants are allowed.")
    if isinstance(node, ast.Num):
        return float(node.n)
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        ops = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b,
            ast.Pow: lambda a, b: a ** b,
        }
        op_func = ops.get(type(node.op))
        if op_func is None:
            raise ToolExecutionError("Unsupported binary operator.")
        return op_func(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ToolExecutionError("Unsupported unary operator.")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ToolExecutionError("Only simple math function calls are allowed.")
        allowed_funcs = {
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "exp": math.exp, "ceil": math.ceil, "floor": math.floor,
        }
        func_name = node.func.id
        if func_name not in allowed_funcs:
            raise ToolExecutionError(f"Function '{func_name}' is not allowed.")
        args = [_safe_eval(arg) for arg in node.args]
        return float(allowed_funcs[func_name](*args))
    if isinstance(node, ast.Name):
        constants = {"pi": math.pi, "e": math.e}
        if node.id in constants:
            return float(constants[node.id])
        raise ToolExecutionError(f"Name '{node.id}' is not allowed.")
    raise ToolExecutionError("Unsupported expression.")


def calculate(expression: str) -> Dict[str, Any]:
    """Evaluate a mathematical expression."""
    if not expression.strip():
        raise ToolExecutionError("Expression must not be empty.")
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval(parsed)
    except Exception as e:
        if isinstance(e, ToolExecutionError):
            raise
        raise ToolExecutionError(f"Failed to evaluate expression: {e}") from e
    return {"expression": expression, "result": result}


# ---------------------------------------------------------------------------
# file_system
# ---------------------------------------------------------------------------

def read_file(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise ToolExecutionError(f"File does not exist: {path}")
    if not file_path.is_file():
        raise ToolExecutionError(f"Path is not a file: {path}")
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolExecutionError("File is not UTF-8 text.")
    except OSError as e:
        raise ToolExecutionError(f"Failed to read file: {e}") from e
    return {"path": str(file_path), "content": content}


def write_file(path: str, content: str) -> Dict[str, Any]:
    file_path = Path(path)
    try:
        _ensure_parent_dir(file_path)
        file_path.write_text(content, encoding="utf-8")
    except OSError as e:
        raise ToolExecutionError(f"Failed to write file: {e}") from e
    return {"path": str(file_path), "bytes_written": len(content.encode("utf-8"))}


def edit_file(path: str, instruction: str) -> Dict[str, Any]:
    """
    Edit a file with simple directives:
      append: <text>
      prepend: <text>
      replace: <old> -> <new>
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ToolExecutionError(f"File does not exist: {path}")
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ToolExecutionError(f"Failed to read file before edit: {e}") from e

    instruction = instruction.strip()
    if instruction.lower().startswith("append:"):
        updated = content + instruction[len("append:"):].lstrip()
    elif instruction.lower().startswith("prepend:"):
        updated = instruction[len("prepend:"):].lstrip() + content
    elif instruction.lower().startswith("replace:"):
        body = instruction[len("replace:"):].strip()
        if "->" not in body:
            raise ToolExecutionError("Replace instruction must look like: replace: old text -> new text")
        old, new = body.split("->", 1)
        old, new = old.strip(), new.strip()
        if old not in content:
            raise ToolExecutionError(f"Text to replace not found: {old!r}")
        updated = content.replace(old, new, 1)
    else:
        raise ToolExecutionError(
            "Unsupported edit instruction. Use 'append:', 'prepend:', or 'replace: old -> new'."
        )

    try:
        file_path.write_text(updated, encoding="utf-8")
    except OSError as e:
        raise ToolExecutionError(f"Failed to write edited file: {e}") from e
    return {"path": str(file_path), "status": "edited"}


def list_directory(path: str, recursive: bool = False) -> Dict[str, Any]:
    """List files and directories within a given path."""
    dir_path = Path(path)
    if not dir_path.exists():
        raise ToolExecutionError(f"Path does not exist: {path}")
    if not dir_path.is_dir():
        raise ToolExecutionError(f"Path is not a directory: {path}")

    entries: List[Dict[str, Any]] = []
    iterator = dir_path.rglob("*") if recursive else dir_path.iterdir()

    for entry in sorted(iterator):
        stat = entry.stat()
        entries.append(
            {
                "name": entry.name,
                "path": str(entry),
                "type": "directory" if entry.is_dir() else "file",
                "size_bytes": stat.st_size if entry.is_file() else None,
            }
        )

    return {"path": str(dir_path), "recursive": recursive, "entries": entries, "count": len(entries)}


def move_file(source: str, destination: str) -> Dict[str, Any]:
    """Move or rename a file or directory."""
    src = Path(source)
    dst = Path(destination)
    if not src.exists():
        raise ToolExecutionError(f"Source does not exist: {source}")
    try:
        _ensure_parent_dir(dst)
        shutil.move(str(src), str(dst))
    except OSError as e:
        raise ToolExecutionError(f"Failed to move: {e}") from e
    return {"source": str(src), "destination": str(dst), "status": "moved"}


def copy_file(source: str, destination: str) -> Dict[str, Any]:
    """Copy a file or directory to a new location."""
    src = Path(source)
    dst = Path(destination)
    if not src.exists():
        raise ToolExecutionError(f"Source does not exist: {source}")
    try:
        _ensure_parent_dir(dst)
        if src.is_dir():
            shutil.copytree(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
    except OSError as e:
        raise ToolExecutionError(f"Failed to copy: {e}") from e
    return {"source": str(src), "destination": str(dst), "status": "copied"}


def delete_file(path: str, recursive: bool = False) -> Dict[str, Any]:
    """Delete a file or directory from disk."""
    target = Path(path)
    if not target.exists():
        raise ToolExecutionError(f"Path does not exist: {path}")
    try:
        if target.is_dir():
            if recursive:
                shutil.rmtree(str(target))
            else:
                target.rmdir()  # raises if non-empty
        else:
            target.unlink()
    except OSError as e:
        raise ToolExecutionError(f"Failed to delete: {e}") from e
    return {"path": str(target), "status": "deleted"}


def fetch_file_metadata(path: str) -> Dict[str, Any]:
    """Get metadata for a file without reading its contents."""
    file_path = Path(path)
    if not file_path.exists():
        raise ToolExecutionError(f"Path does not exist: {path}")
    try:
        stat = file_path.stat()
    except OSError as e:
        raise ToolExecutionError(f"Failed to stat file: {e}") from e

    mime_type, encoding = mimetypes.guess_type(str(file_path))
    return {
        "path": str(file_path),
        "size_bytes": stat.st_size,
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "is_file": file_path.is_file(),
        "is_dir": file_path.is_dir(),
        "mime_type": mime_type,
        "encoding": encoding,
        "permissions": oct(stat.st_mode),
    }


def grep(pattern: str, path: str) -> Dict[str, Any]:
    """Search for a pattern in files and return matching lines."""
    target = Path(path)
    if not target.exists():
        raise ToolExecutionError(f"Path does not exist: {path}")
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ToolExecutionError(f"Invalid regex pattern: {e}") from e

    matches: List[Dict[str, Any]] = []
    files = [target] if target.is_file() else [p for p in target.rglob("*") if p.is_file()]

    for file_path in files:
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line_no, line in enumerate(lines, start=1):
            if regex.search(line):
                matches.append({"file": str(file_path), "line_number": line_no, "line": line})

    return {"pattern": pattern, "path": str(target), "matches": matches, "count": len(matches)}


def run_bash_command(command: str) -> Dict[str, Any]:
    """Execute a bash command in the system shell."""
    if not command.strip():
        raise ToolExecutionError("Command must not be empty.")
    try:
        completed = subprocess.run(
            command, shell=True, text=True, capture_output=True, timeout=60
        )
    except subprocess.TimeoutExpired as e:
        raise ToolExecutionError(f"Command timed out: {e}") from e
    except Exception as e:
        raise ToolExecutionError(f"Command execution failed: {e}") from e
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------

def store_memory(key: str, value: str) -> Dict[str, Any]:
    if not key.strip():
        raise ToolExecutionError("Memory key must not be empty.")
    data = _load_memory_store()
    data[key] = value
    _save_memory_store(data)
    return {"key": key, "value": value, "status": "stored"}


def retrieve_memory(query: str) -> Dict[str, Any]:
    if not query.strip():
        raise ToolExecutionError("Memory query must not be empty.")
    data = _load_memory_store()
    query_lower = query.lower()
    exact, partial = [], []
    for key, value in data.items():
        if query_lower == key.lower():
            exact.append({"key": key, "value": value})
        elif query_lower in key.lower() or query_lower in value.lower():
            partial.append({"key": key, "value": value})
    return {"query": query, "exact_matches": exact, "partial_matches": partial}


def update_memory(key: str, value: str, merge: bool = False) -> Dict[str, Any]:
    """Atomically update an existing memory entry."""
    if not key.strip():
        raise ToolExecutionError("Memory key must not be empty.")
    data = _load_memory_store()
    if merge and key in data:
        data[key] = data[key] + "\n" + value
    else:
        data[key] = value
    _save_memory_store(data)
    return {"key": key, "value": data[key], "merged": merge, "status": "updated"}


def delete_memory(key: str) -> Dict[str, Any]:
    """Delete a stored memory entry by key."""
    if not key.strip():
        raise ToolExecutionError("Memory key must not be empty.")
    data = _load_memory_store()
    if key not in data:
        raise ToolExecutionError(f"Memory key not found: {key}")
    del data[key]
    _save_memory_store(data)
    return {"key": key, "status": "deleted"}


def list_memories(prefix: str = "") -> Dict[str, Any]:
    """List all stored memory keys, optionally filtered by prefix."""
    data = _load_memory_store()
    keys = [k for k in data.keys() if k.startswith(prefix)]
    return {"prefix": prefix or None, "keys": keys, "count": len(keys)}


# ---------------------------------------------------------------------------
# external_integrations
# ---------------------------------------------------------------------------

def send_http_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    timeout_ms: int = 10000,
) -> Dict[str, Any]:
    """Make a generic HTTP request to any external REST API."""
    if not url.strip():
        raise ToolExecutionError("URL must not be empty.")
    valid_methods = {"GET", "POST", "PUT", "PATCH", "DELETE"}
    method = method.upper()
    if method not in valid_methods:
        raise ToolExecutionError(f"method must be one of {valid_methods}.")

    timeout_s = timeout_ms / 1000.0
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers or {},
            data=body,
            timeout=timeout_s,
        )
    except requests.RequestException as e:
        raise ToolExecutionError(f"HTTP request failed: {e}") from e

    try:
        response_body = response.json()
    except Exception:
        response_body = response.text

    return {
        "url": url,
        "method": method,
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "body": response_body,
    }


# ---------------------------------------------------------------------------
# structured_data
# ---------------------------------------------------------------------------

def parse_document(
    path: str,
    extract: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Extract structured data or clean text from documents.

    Supports plain text, JSON, CSV, XML, HTML, and PDF (if pypdf is installed).
    For DOCX/XLSX install the optional deps (python-docx, openpyxl).
    """
    if extract is None:
        extract = ["text"]

    file_path = Path(path) if not path.startswith("http") else None
    result: Dict[str, Any] = {"path": path, "extracted": {}}

    # Remote URL — fetch first
    raw: str = ""
    if file_path is None:
        resp = fetch_url(path, format="text")
        raw = resp["content"]
        extension = ""
    else:
        if not file_path.exists():
            raise ToolExecutionError(f"File does not exist: {path}")
        extension = file_path.suffix.lower()

        if extension in {".txt", ".md", ".log", ".csv", ".json", ".xml", ".html", ".htm"}:
            raw = file_path.read_text(encoding="utf-8")
        elif extension == ".pdf":
            try:
                from pypdf import PdfReader  # type: ignore
                reader = PdfReader(str(file_path))
                raw = "\n".join(p.extract_text() or "" for p in reader.pages)
                result["page_count"] = len(reader.pages)
            except ImportError:
                raise ToolExecutionError("pypdf not installed. Run: pip install pypdf")
        elif extension in {".docx"}:
            try:
                import docx  # type: ignore
                doc = docx.Document(str(file_path))
                raw = "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                raise ToolExecutionError("python-docx not installed. Run: pip install python-docx")
        elif extension in {".xlsx", ".xlsm"}:
            try:
                import openpyxl  # type: ignore
                wb = openpyxl.load_workbook(str(file_path), read_only=True, data_only=True)
                rows_out = []
                for ws in wb.worksheets:
                    for row in ws.iter_rows(values_only=True):
                        rows_out.append(list(row))
                if "tables" in extract:
                    result["extracted"]["tables"] = rows_out
                raw = "\n".join("\t".join(str(c) for c in r) for r in rows_out)
            except ImportError:
                raise ToolExecutionError("openpyxl not installed. Run: pip install openpyxl")
        else:
            raw = file_path.read_text(encoding="utf-8", errors="replace")

    if "text" in extract:
        result["extracted"]["text"] = raw

    if "metadata" in extract and file_path:
        result["extracted"]["metadata"] = fetch_file_metadata(path)

    if "tables" in extract and extension == ".csv":
        reader = csv.reader(io.StringIO(raw))
        result["extracted"]["tables"] = list(reader)

    if "images" in extract:
        result["extracted"]["images"] = []  # placeholder — requires document-specific parsing

    return result


def convert_format(input: str, from_format: str, to_format: str) -> Dict[str, Any]:
    """
    Transform data between structured formats:
    json, csv, xml, yaml, markdown, toml.
    """
    valid_formats = {"json", "csv", "xml", "yaml", "markdown", "toml"}
    if from_format not in valid_formats:
        raise ToolExecutionError(f"from_format must be one of {valid_formats}.")
    if to_format not in valid_formats:
        raise ToolExecutionError(f"to_format must be one of {valid_formats}.")
    if not input.strip():
        raise ToolExecutionError("Input must not be empty.")

    # --- parse input ---
    data: Any = None

    if from_format == "json":
        try:
            data = json.loads(input)
        except json.JSONDecodeError as e:
            raise ToolExecutionError(f"Invalid JSON input: {e}") from e

    elif from_format == "csv":
        reader = csv.DictReader(io.StringIO(input))
        data = list(reader)

    elif from_format == "xml":
        try:
            root = ET.fromstring(input)

            def _xml_to_dict(el: ET.Element) -> Any:
                children = list(el)
                if not children:
                    return el.text or ""
                result: Dict[str, Any] = {}
                for child in children:
                    child_val = _xml_to_dict(child)
                    if child.tag in result:
                        if not isinstance(result[child.tag], list):
                            result[child.tag] = [result[child.tag]]
                        result[child.tag].append(child_val)
                    else:
                        result[child.tag] = child_val
                return result

            data = {root.tag: _xml_to_dict(root)}
        except ET.ParseError as e:
            raise ToolExecutionError(f"Invalid XML input: {e}") from e

    elif from_format == "yaml":
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(input)
        except ImportError:
            raise ToolExecutionError("PyYAML not installed. Run: pip install pyyaml")
        except Exception as e:
            raise ToolExecutionError(f"Invalid YAML input: {e}") from e

    elif from_format == "toml":
        try:
            import tomllib  # Python 3.11+
            data = tomllib.loads(input)
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore
                data = tomllib.loads(input)
            except ImportError:
                raise ToolExecutionError("tomli not installed. Run: pip install tomli")
        except Exception as e:
            raise ToolExecutionError(f"Invalid TOML input: {e}") from e

    elif from_format == "markdown":
        # Treat markdown as raw text; wrap in a dict for downstream formats
        data = {"content": input}

    # --- serialize output ---
    output: str = ""

    if to_format == "json":
        output = json.dumps(data, indent=2, ensure_ascii=False)

    elif to_format == "csv":
        buf = io.StringIO()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            writer = csv.DictWriter(buf, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
        elif isinstance(data, dict):
            writer = csv.writer(buf)
            for k, v in data.items():
                writer.writerow([k, v])
        else:
            raise ToolExecutionError("Data cannot be represented as CSV.")
        output = buf.getvalue()

    elif to_format == "xml":
        def _dict_to_xml(tag: str, value: Any) -> ET.Element:
            el = ET.Element(tag)
            if isinstance(value, dict):
                for k, v in value.items():
                    el.append(_dict_to_xml(k, v))
            elif isinstance(value, list):
                for item in value:
                    el.append(_dict_to_xml("item", item))
            else:
                el.text = str(value)
            return el

        root_tag = "root"
        if isinstance(data, dict) and len(data) == 1:
            root_tag, data = next(iter(data.items()))
        xml_el = _dict_to_xml(root_tag, data)
        ET.indent(xml_el)
        output = ET.tostring(xml_el, encoding="unicode")

    elif to_format == "yaml":
        try:
            import yaml  # type: ignore
            output = yaml.dump(data, allow_unicode=True, default_flow_style=False)
        except ImportError:
            raise ToolExecutionError("PyYAML not installed. Run: pip install pyyaml")

    elif to_format == "toml":
        try:
            import tomli_w  # type: ignore
            output = tomli_w.dumps(data if isinstance(data, dict) else {"value": data})
        except ImportError:
            raise ToolExecutionError("tomli-w not installed. Run: pip install tomli-w")

    elif to_format == "markdown":
        if isinstance(data, list) and data and isinstance(data[0], dict):
            headers = list(data[0].keys())
            lines = ["| " + " | ".join(headers) + " |"]
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in data:
                lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
            output = "\n".join(lines)
        elif isinstance(data, dict):
            lines = []
            for k, v in data.items():
                lines.append(f"**{k}**: {v}")
            output = "\n".join(lines)
        else:
            output = str(data)

    return {
        "from_format": from_format,
        "to_format": to_format,
        "output": output,
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    name: str
    func: Callable[..., Dict[str, Any]]


TOOL_REGISTRY: Dict[str, ToolSpec] = {
    # information_retrieval
    "web_search": ToolSpec("web_search", web_search),
    "fetch_url": ToolSpec("fetch_url", fetch_url),
    "search_knowledge_base": ToolSpec("search_knowledge_base", search_knowledge_base),
    # computation
    "run_python": ToolSpec("run_python", run_python),
    "run_javascript": ToolSpec("run_javascript", run_javascript),
    "run_sql": ToolSpec("run_sql", run_sql),
    "run_notebook_cell": ToolSpec("run_notebook_cell", run_notebook_cell),
    "calculate": ToolSpec("calculate", calculate),
    # file_system
    "read_file": ToolSpec("read_file", read_file),
    "write_file": ToolSpec("write_file", write_file),
    "edit_file": ToolSpec("edit_file", edit_file),
    "list_directory": ToolSpec("list_directory", list_directory),
    "move_file": ToolSpec("move_file", move_file),
    "copy_file": ToolSpec("copy_file", copy_file),
    "delete_file": ToolSpec("delete_file", delete_file),
    "fetch_file_metadata": ToolSpec("fetch_file_metadata", fetch_file_metadata),
    "grep": ToolSpec("grep", grep),
    "run_bash_command": ToolSpec("run_bash_command", run_bash_command),
    # memory
    "store_memory": ToolSpec("store_memory", store_memory),
    "retrieve_memory": ToolSpec("retrieve_memory", retrieve_memory),
    "update_memory": ToolSpec("update_memory", update_memory),
    "delete_memory": ToolSpec("delete_memory", delete_memory),
    "list_memories": ToolSpec("list_memories", list_memories),
    # external_integrations
    "send_http_request": ToolSpec("send_http_request", send_http_request),
    # structured_data
    "parse_document": ToolSpec("parse_document", parse_document),
    "convert_format": ToolSpec("convert_format", convert_format),
}


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        raise ToolExecutionError(f"Unknown tool: {tool_name}")
    try:
        return spec.func(**arguments)
    except TypeError as e:
        raise ToolExecutionError(
            f"Invalid arguments for tool '{tool_name}': {e}"
        ) from e


def _matches_schema_type(value: Any, spec: Dict[str, Any]) -> bool:
    schema_type = spec.get("type")
    if schema_type is None:
        return True
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "array":
        if not isinstance(value, list):
            return False
        item_spec = spec.get("items")
        if isinstance(item_spec, dict):
            return all(_matches_schema_type(item, item_spec) for item in value)
        return True
    if schema_type == "object":
        return isinstance(value, dict)
    return True


def _validate_arguments_against_schema(
    tool_name: str,
    arguments: Dict[str, Any],
    tool_spec: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if tool_spec is None:
        return dict(arguments)

    schema = tool_spec.get("parameters", tool_spec)
    if not isinstance(schema, dict):
        raise ToolExecutionError(f"Tool spec for '{tool_name}' does not contain a valid schema.")

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ToolExecutionError(f"Tool schema for '{tool_name}' has invalid properties.")

    required = set(schema.get("required", []))
    normalized = dict(arguments)

    extra_keys = sorted(set(normalized) - set(properties))
    if extra_keys:
        raise ToolExecutionError(
            f"Unexpected arguments for tool '{tool_name}': {extra_keys}"
        )

    for name, raw_spec in properties.items():
        spec = raw_spec if isinstance(raw_spec, dict) else {}
        if name not in normalized and spec.get("default") is not None:
            normalized[name] = spec["default"]

    missing_required = sorted(name for name in required if name not in normalized)
    if missing_required:
        raise ToolExecutionError(
            f"Missing required arguments for tool '{tool_name}': {missing_required}"
        )

    for name, value in normalized.items():
        spec = properties.get(name)
        if not isinstance(spec, dict):
            continue
        if "enum" in spec and value not in spec["enum"]:
            raise ToolExecutionError(
                f"Argument '{name}' for tool '{tool_name}' must be one of {spec['enum']}."
            )
        if not _matches_schema_type(value, spec):
            expected_type = spec.get("type", "valid")
            raise ToolExecutionError(
                f"Argument '{name}' for tool '{tool_name}' must match schema type '{expected_type}'."
            )

    return normalized


def dispatch_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    tool_spec: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved_arguments = dict(arguments)
    try:
        if tool_name not in TOOL_REGISTRY:
            raise ToolExecutionError(f"Unknown tool: {tool_name}")
        resolved_arguments = _validate_arguments_against_schema(
            tool_name,
            resolved_arguments,
            tool_spec,
        )
        output = execute_tool(tool_name, resolved_arguments)
        return {
            "tool": tool_name,
            "arguments": resolved_arguments,
            "status": "ok",
            "output": output,
            "error": None,
        }
    except ToolExecutionError as exc:
        return {
            "tool": tool_name,
            "arguments": resolved_arguments,
            "status": "error",
            "output": None,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "tool": tool_name,
            "arguments": resolved_arguments,
            "status": "error",
            "output": None,
            "error": f"Unexpected tool execution failure: {exc}",
        }
