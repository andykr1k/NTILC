from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = None


def _fallback_dotenv_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue

        key, raw_value = stripped.split("=", 1)
        env_key = key.strip()
        if not env_key:
            continue
        try:
            parsed = shlex.split(raw_value, comments=True, posix=True)
            env_value = parsed[0] if parsed else ""
        except ValueError:
            env_value = raw_value.strip().strip('"').strip("'")
        values[env_key] = env_value
    return values


def read_env_file(path: Path | str) -> dict[str, str]:
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return {}
    if dotenv_values is None:
        return _fallback_dotenv_values(dotenv_path)

    parsed_values = dotenv_values(dotenv_path)
    return {
        str(key): str(value)
        for key, value in parsed_values.items()
        if str(key).strip() and value is not None
    }


def load_env_file(path: Path | str) -> list[str]:
    loaded_keys: list[str] = []
    for env_key, env_value in read_env_file(path).items():
        if os.environ.get(env_key):
            continue
        os.environ[env_key] = env_value
        loaded_keys.append(env_key)
    return loaded_keys


def get_env_file_value(path: Path | str, key: str, default: Any = "") -> str:
    value = read_env_file(path).get(key)
    if value is None:
        return str(default)
    return value
