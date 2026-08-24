from __future__ import annotations

import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .models import MAX_MARKET_PLUGINS, PluginRecord
from .network_safety import PUBLIC_HTTPS_OPENER, validate_public_https_url

DEFAULT_MARKET_URL = "https://cloud.astrbot.app/api/v1/market/plugins.json"
GITHUB_API = "https://api.github.com"


class ApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int = 0,
        retry_after: int = 0,
        rate_remaining: int | None = None,
        rate_reset: int = 0,
        rate_limited: bool = False,
    ):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.rate_remaining = rate_remaining
        self.rate_reset = rate_reset
        self.rate_limited = rate_limited


@dataclass(slots=True)
class GitHubObservation:
    commit_sha: str
    tree: list[dict[str, Any]]
    packages: list[str]
    tree_ok: bool
    sbom_ok: bool
    errors: list[str]
    rate_remaining: int | None = None
    tree_sha: str = ""
    commit_ok: bool = False
    commit_api: str = ""


def _safe_json_request(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
    max_bytes: int = 16 * 1024 * 1024,
) -> tuple[Any, dict[str, str]]:
    validate_public_https_url(url, label="plugin market URL")
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with PUBLIC_HTTPS_OPENER.open(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                raise ApiError(f"response too large: {length} bytes")
            payload = response.read(max_bytes + 1)
            if len(payload) > max_bytes:
                raise ApiError(f"response exceeded {max_bytes} bytes")
            return json.loads(payload.decode("utf-8")), {
                key.lower(): value for key, value in response.headers.items()
            }
    except urllib.error.HTTPError as exc:
        retry_after = int(exc.headers.get("Retry-After") or 0)
        remaining_raw = exc.headers.get("X-RateLimit-Remaining")
        remaining = (
            int(remaining_raw) if remaining_raw and remaining_raw.isdigit() else None
        )
        reset_raw = exc.headers.get("X-RateLimit-Reset")
        reset = int(reset_raw) if reset_raw and reset_raw.isdigit() else 0
        try:
            body = exc.read(4096).decode("utf-8", errors="replace").lower()
        except OSError:
            body = ""
        rate_limited = exc.code == 429 or "rate limit" in body or "abuse" in body
        raise ApiError(
            f"HTTP {exc.code} for {url}",
            status=exc.code,
            retry_after=retry_after,
            rate_remaining=remaining,
            rate_reset=reset,
            rate_limited=rate_limited,
        ) from exc
    except (
        urllib.error.URLError,
        TimeoutError,
        http.client.IncompleteRead,
        ConnectionError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise ApiError(f"request failed for {url}: {exc}") from exc


def load_market(
    url: str = DEFAULT_MARKET_URL,
    *,
    timeout: float = 20.0,
    max_retries: int = 3,
    deadline_seconds: float | None = None,
) -> tuple[dict[str, Any], list[PluginRecord]]:
    raw: Any = None
    retries = max(0, min(6, int(max_retries)))
    deadline = (
        time.monotonic() + max(0.1, float(deadline_seconds))
        if deadline_seconds is not None
        else None
    )
    for attempt in range(retries + 1):
        remaining = deadline - time.monotonic() if deadline is not None else None
        if remaining is not None and remaining <= 0:
            raise ApiError("market request deadline exceeded")
        try:
            raw, _ = _safe_json_request(
                url,
                headers={"User-Agent": "astrbot-plugin-advisor/0.1"},
                timeout=max(0.1, min(timeout, remaining)) if remaining else timeout,
            )
            break
        except ApiError as exc:
            if exc.status or attempt >= retries:
                raise
            delay = backoff_seconds(attempt + 1)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ApiError("market request deadline exceeded") from exc
                delay = min(delay, remaining)
            time.sleep(delay)
    if not isinstance(raw, dict):
        raise ApiError("market root must be an object")
    metadata = raw.get("$meta")
    if not isinstance(metadata, dict):
        raise ApiError("market metadata $meta is missing")
    if len(raw) - 1 > MAX_MARKET_PLUGINS:
        raise ApiError(f"market exceeds {MAX_MARKET_PLUGINS} plugin records")
    records: list[PluginRecord] = []
    for key, value in raw.items():
        if key == "$meta":
            continue
        if not isinstance(value, dict):
            continue
        record = PluginRecord.from_market(key, value)
        if (
            record.author
            and record.name
            and record.repo.startswith("https://github.com/")
        ):
            records.append(record)
    return metadata, records


def parse_github_repo(repo_url: str) -> tuple[str, str, str]:
    parsed = urllib.parse.urlparse(repo_url)
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise ValueError("only public github.com HTTPS repositories are supported")
    parts = [urllib.parse.unquote(x) for x in parsed.path.strip("/").split("/")]
    if len(parts) < 2:
        raise ValueError("invalid GitHub repository URL")
    owner, repo = parts[0], parts[1].removesuffix(".git")
    branch = "HEAD"
    if len(parts) >= 4 and parts[2] == "tree":
        branch = parts[3]
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if not owner or not repo or not set(owner) <= allowed or not set(repo) <= allowed:
        raise ValueError("unsafe GitHub owner or repository name")
    return owner, repo, branch


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        *,
        timeout: float = 20.0,
        api_version: str = "2022-11-28",
        max_retries: int = 4,
        min_interval: float = 0.35,
    ) -> None:
        self.token = token or os.getenv("GITHUB_TOKEN", "")
        self.timeout = timeout
        self.api_version = api_version
        self.max_retries = max(0, min(8, max_retries))
        self.min_interval = max(0.0, min(10.0, min_interval))
        self._throttle_lock = threading.Lock()
        self._last_request = 0.0
        self._blocked_until = 0.0

    @property
    def authenticated(self) -> bool:
        return bool(self.token)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "astrbot-plugin-advisor/0.1",
            "X-GitHub-Api-Version": self.api_version,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get(
        self,
        path: str,
        *,
        max_bytes: int = 16 * 1024 * 1024,
        deadline: float | None = None,
    ) -> tuple[Any, dict[str, str]]:
        if not path.startswith("/"):
            raise ValueError("GitHub API path must be absolute")
        for attempt in range(self.max_retries + 1):
            with self._throttle_lock:
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    raise ApiError("GitHub observation deadline exceeded")
                wait_for = max(
                    0.0,
                    self._blocked_until - now,
                    self.min_interval - (now - self._last_request),
                )
                if wait_for:
                    if deadline is not None and now + wait_for >= deadline:
                        raise ApiError("GitHub observation deadline exceeded")
                    time.sleep(wait_for)
                self._last_request = time.monotonic()
            try:
                request_timeout = self.timeout
                if deadline is not None:
                    request_timeout = min(
                        request_timeout,
                        max(0.1, deadline - time.monotonic()),
                    )
                return _safe_json_request(
                    GITHUB_API + path,
                    headers=self._headers(),
                    timeout=request_timeout,
                    max_bytes=max_bytes,
                )
            except ApiError as exc:
                transient = exc.status == 0
                if (
                    not exc.rate_limited and not transient
                ) or attempt >= self.max_retries:
                    raise
                delay = backoff_seconds(attempt + 1, exc.retry_after)
                if exc.rate_remaining == 0 and exc.rate_reset:
                    delay = max(delay, min(300.0, exc.rate_reset - time.time() + 2.0))
                if deadline is not None and time.monotonic() + delay >= deadline:
                    raise ApiError("GitHub observation deadline exceeded") from exc
                with self._throttle_lock:
                    self._blocked_until = max(
                        self._blocked_until, time.monotonic() + delay
                    )
        raise AssertionError("unreachable")

    def rate_status(self) -> dict[str, int]:
        raw, _ = self._get("/rate_limit", max_bytes=512 * 1024)
        core = raw.get("resources", {}).get("core", {}) if isinstance(raw, dict) else {}
        return {
            "limit": max(0, int(core.get("limit") or 0)),
            "used": max(0, int(core.get("used") or 0)),
            "remaining": max(0, int(core.get("remaining") or 0)),
            "reset": max(0, int(core.get("reset") or 0)),
        }

    @staticmethod
    def _valid_object_id(value: str) -> bool:
        return len(value) in {40, 64} and all(
            char in "0123456789abcdef" for char in value.lower()
        )

    def observe(
        self,
        repo_url: str,
        cached: GitHubObservation | None = None,
        *,
        include_sbom: bool = True,
        deadline_seconds: float | None = None,
    ) -> GitHubObservation:
        owner, repo, branch = parse_github_repo(repo_url)
        deadline = (
            time.monotonic() + max(0.1, float(deadline_seconds))
            if deadline_seconds is not None
            else None
        )
        quoted_ref = urllib.parse.quote(branch, safe="")
        errors = list(cached.errors) if cached else []
        tree = list(cached.tree) if cached else []
        packages = list(cached.packages) if cached else []
        tree_sha = str(cached.tree_sha) if cached else ""
        cached_commit_is_metadata_only = bool(
            cached and cached.commit_api == "list_commits_metadata"
        )
        commit_sha = (
            str(cached.commit_sha)
            if cached
            and cached.commit_ok
            and cached_commit_is_metadata_only
            and self._valid_object_id(cached.commit_sha)
            else ""
        )
        rate_remaining = cached.rate_remaining if cached else None
        commit_ok = bool(commit_sha)
        commit_api = cached.commit_api if commit_ok and cached else ""
        tree_ok = bool(cached and cached.tree_ok)
        sbom_ok = bool(cached and cached.sbom_ok)

        # Git Trees returns a tree object SHA, not the commit SHA.  Resolve the
        # repository ref through the Commits API so every profile can be bound
        # to the exact source revision it assessed.
        if not commit_ok:
            errors = [item for item in errors if not item.startswith("commit:")]
            try:
                commit_path = f"/repos/{owner}/{repo}/commits?per_page=1"
                if branch != "HEAD":
                    commit_path += f"&sha={quoted_ref}"
                commit_rows, headers = self._get(
                    commit_path,
                    max_bytes=512 * 1024,
                    deadline=deadline,
                )
                rate = headers.get("x-ratelimit-remaining")
                rate_remaining = (
                    int(rate) if rate and rate.isdigit() else rate_remaining
                )
                commit_raw = (
                    commit_rows[0]
                    if isinstance(commit_rows, list)
                    and commit_rows
                    and isinstance(commit_rows[0], dict)
                    else {}
                )
                candidate = str(commit_raw.get("sha") or "").lower()
                if not self._valid_object_id(candidate):
                    raise ApiError(
                        "GitHub commit response did not contain a valid object ID"
                    )
                commit_sha = candidate
                commit_ok = True
                commit_api = "list_commits_metadata"
                commit_tree = (
                    commit_raw.get("commit", {}).get("tree", {})
                    if isinstance(commit_raw, dict)
                    else {}
                )
                candidate_tree_sha = str(commit_tree.get("sha") or "").lower()
                if self._valid_object_id(candidate_tree_sha):
                    tree_sha = candidate_tree_sha
            except ApiError as exc:
                if exc.rate_limited:
                    raise
                errors.append(f"commit:{exc.status or 'error'}")

        if not tree_ok:
            errors = [item for item in errors if not item.startswith("tree:")]
            try:
                tree_ref = (
                    urllib.parse.quote(commit_sha, safe="") if commit_ok else quoted_ref
                )
                tree_raw, headers = self._get(
                    f"/repos/{owner}/{repo}/git/trees/{tree_ref}?recursive=1",
                    deadline=deadline,
                )
                rate = headers.get("x-ratelimit-remaining")
                rate_remaining = (
                    int(rate) if rate and rate.isdigit() else rate_remaining
                )
                if isinstance(tree_raw, dict):
                    response_tree_sha = str(tree_raw.get("sha") or "").lower()
                    if not tree_sha and self._valid_object_id(response_tree_sha):
                        tree_sha = response_tree_sha
                    raw_items = tree_raw.get("tree") or []
                    tree = [
                        {
                            "path": str(item.get("path") or "")[:1024],
                            "type": str(item.get("type") or ""),
                            "size": max(0, int(item.get("size") or 0)),
                        }
                        for item in raw_items[:100_000]
                        if isinstance(item, dict)
                    ]
                    tree_ok = True
                    if tree_raw.get("truncated"):
                        errors.append("github_tree_truncated")
            except ApiError as exc:
                if exc.rate_limited:
                    raise
                errors.append(f"tree:{exc.status or 'error'}")

        permanent_sbom_miss = any(item.startswith("sbom:404") for item in errors)
        if include_sbom and not sbom_ok and not permanent_sbom_miss:
            errors = [item for item in errors if not item.startswith("sbom:")]
            try:
                sbom_raw, headers = self._get(
                    f"/repos/{owner}/{repo}/dependency-graph/sbom",
                    max_bytes=8 * 1024 * 1024,
                    deadline=deadline,
                )
                rate = headers.get("x-ratelimit-remaining")
                if rate and rate.isdigit():
                    rate_remaining = int(rate)
                sbom = sbom_raw.get("sbom", {}) if isinstance(sbom_raw, dict) else {}
                for package in sbom.get("packages") or []:
                    if not isinstance(package, dict):
                        continue
                    name = str(package.get("name") or "").strip().lower()
                    if name and len(name) <= 200:
                        packages.append(name)
                packages = sorted(set(packages))
                sbom_ok = True
            except ApiError as exc:
                if exc.rate_limited:
                    raise
                errors.append(f"sbom:{exc.status or 'error'}")

        return GitHubObservation(
            commit_sha=commit_sha,
            tree=tree,
            packages=packages,
            tree_ok=tree_ok,
            sbom_ok=sbom_ok,
            errors=errors,
            rate_remaining=rate_remaining,
            tree_sha=tree_sha,
            commit_ok=commit_ok,
            commit_api=commit_api,
        )


def backoff_seconds(attempt: int, retry_after: int = 0) -> float:
    if retry_after > 0:
        return min(300.0, float(retry_after))
    return min(60.0, 2.0 ** max(0, attempt - 1) + (time.time() % 1.0))
