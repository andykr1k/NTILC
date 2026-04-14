#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLISH_MANIFEST = REPO_ROOT / "registry" / "models" / "publish_manifest.json"
DEFAULT_RELEASES_PATH = REPO_ROOT / "registry" / "models" / "releases.yaml"
DEFAULT_DOTENV_PATH = REPO_ROOT / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish prepared NTILC model releases to Hugging Face.")
    parser.add_argument("--publish-manifest", type=Path, default=DEFAULT_PUBLISH_MANIFEST)
    parser.add_argument("--releases-path", type=Path, default=DEFAULT_RELEASES_PATH)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--visibility", choices=("public", "private"), default="public")
    parser.add_argument("--release-id", action="append", dest="release_ids")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def get_hf_api() -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is not installed") from exc
    return HfApi()


def load_token_from_dotenv(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        current_key, raw = stripped.split("=", 1)
        if current_key.strip() != key:
            continue
        return raw.strip().strip('"').strip("'")
    return ""


def create_repo_via_http(repo_id: str, token: str, private: bool) -> None:
    org, name = repo_id.split("/", 1)
    payload = (
        f'{{"name":"{name}","organization":"{org}","private":{str(private).lower()},"type":"model"}}'
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://huggingface.co/api/repos/create",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            return
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to create repo {repo_id}: {exc.code} {body}") from exc


def run_git(args: list[str], cwd: Path, token: str) -> None:
    env = os.environ.copy()
    basic_auth = base64.b64encode(f"__token__:{token}".encode("utf-8")).decode("ascii")
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "http.extraheader"
    env["GIT_CONFIG_VALUE_0"] = f"Authorization: Basic {basic_auth}"
    completed = subprocess.run(args, cwd=str(cwd), env=env, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(args)}):\n{completed.stdout}\n{completed.stderr}".strip()
        )


def publish_via_git(repo_id: str, token: str, checkpoint_path: Path, metrics_path: Path, card_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="ntilc-hf-") as temp_dir:
        repo_dir = Path(temp_dir)
        run_git(["git", "init"], cwd=repo_dir, token=token)
        run_git(["git", "checkout", "-b", "main"], cwd=repo_dir, token=token)
        run_git(["git", "config", "user.name", "Codex"], cwd=repo_dir, token=token)
        run_git(["git", "config", "user.email", "codex@openai.invalid"], cwd=repo_dir, token=token)
        run_git(["git", "remote", "add", "origin", f"https://huggingface.co/{repo_id}"], cwd=repo_dir, token=token)
        run_git(["git", "lfs", "install", "--local"], cwd=repo_dir, token=token)
        run_git(["git", "config", "lfs.basictransfersonly", "true"], cwd=repo_dir, token=token)
        run_git(["git", "config", "lfs.concurrenttransfers", "1"], cwd=repo_dir, token=token)
        run_git(["git", "lfs", "track", "*.pt"], cwd=repo_dir, token=token)

        shutil.copy2(checkpoint_path, repo_dir / "best.pt")
        shutil.copy2(metrics_path, repo_dir / "metrics.json")
        shutil.copy2(card_path, repo_dir / "README.md")

        run_git(["git", "add", ".gitattributes", "best.pt", "metrics.json", "README.md"], cwd=repo_dir, token=token)
        run_git(["git", "commit", "-m", "Add NTILC model release"], cwd=repo_dir, token=token)
        run_git(["git", "push", "--force", "-u", "origin", "main"], cwd=repo_dir, token=token)


def main() -> None:
    args = parse_args()
    manifest = load_json(args.publish_manifest.resolve())
    releases_payload = load_yaml(args.releases_path.resolve()) or {}
    releases = releases_payload.get("releases")
    require(isinstance(releases, list), f"Invalid releases payload: {args.releases_path}")

    selected_ids = set(args.release_ids or [])
    publish_releases = [
        item for item in manifest.get("releases", []) if not selected_ids or item.get("id") in selected_ids
    ]
    require(publish_releases, "No releases selected for publication.")

    token = os.environ.get(args.token_env, "").strip() or load_token_from_dotenv(DEFAULT_DOTENV_PATH, args.token_env)
    require(token or args.dry_run, f"Missing Hugging Face token in {args.token_env}")

    if args.dry_run:
        for item in publish_releases:
            print(f"[dry-run] would publish {item['repo_id']} from {item['checkpoint_path']}")
        return

    api = None
    use_git_fallback = False
    try:
        api = get_hf_api()
    except RuntimeError:
        use_git_fallback = True

    for item in publish_releases:
        repo_id = str(item["repo_id"])
        checkpoint_path = Path(str(item["checkpoint_path"]))
        metrics_path = Path(str(item["metrics_path"]))
        card_path = Path(str(item["card_path"]))
        if use_git_fallback:
            create_repo_via_http(repo_id=repo_id, token=token, private=args.visibility == "private")
            publish_via_git(
                repo_id=repo_id,
                token=token,
                checkpoint_path=checkpoint_path,
                metrics_path=metrics_path,
                card_path=card_path,
            )
        else:
            api.create_repo(
                repo_id=repo_id,
                token=token,
                private=args.visibility == "private",
                exist_ok=True,
                repo_type="model",
            )
            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo="best.pt",
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
            api.upload_file(
                path_or_fileobj=str(metrics_path),
                path_in_repo="metrics.json",
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
            api.upload_file(
                path_or_fileobj=str(card_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
        print(f"Published {repo_id}")

    published_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    release_by_id = {str(item.get("id")): item for item in publish_releases}
    for release in releases:
        release_id = str(release.get("id", "")).strip()
        if release_id not in release_by_id:
            continue
        publish_item = release_by_id[release_id]
        release["status"] = "published"
        release["published_at"] = published_at
        release["repository_url"] = str(publish_item["repository_url"])
        release["download_url"] = str(publish_item["download_url"])
        notes = str(release.get("notes", "")).strip()
        if "Published to the OpenToolEmbeddings Hugging Face organization." not in notes:
            release["notes"] = (
                f"{notes} Published to the OpenToolEmbeddings Hugging Face organization."
            ).strip()

    write_yaml(args.releases_path.resolve(), releases_payload)
    print(f"Updated {args.releases_path} with published URLs and timestamp {published_at}")


if __name__ == "__main__":
    main()
