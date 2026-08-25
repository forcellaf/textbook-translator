"""
MinerU Cloud API (v4) client — "precision parse" (精准解析) API.

Reference: https://mineru.net/apiManage/docs

Local-file flow (what this module implements)
----------------------------------------------
The `/extract/task` endpoint does **not** accept direct file uploads — it
only accepts a publicly reachable ``url``. To parse a local PDF, MinerU
requires a 3-step signed-upload flow instead:

1. ``POST /file-urls/batch``            -> {batch_id, file_urls: [<signed PUT url>]}
2. ``PUT <signed url>`` with raw bytes  -> uploads the file (no Content-Type header)
   The server auto-detects the completed upload and starts parsing; there is
   no separate "submit" call.
3. ``GET /extract-results/batch/{batch_id}`` -> poll `extract_result[i].state`
   ("running" -> "done" | "failed"). On success, the entry contains a
   ``full_zip_url`` pointing to a ZIP archive (containing the markdown output
   plus images), not inline markdown text.

Auth: ``Authorization: <MINERU_API_KEY>`` (the v4 docs show the header value
as the raw token, with no "Bearer " prefix).
"""

from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from pathlib import Path

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src import config

logger = logging.getLogger(__name__)

# Errors worth retrying: network hiccups / transient server errors. Auth and
# validation errors (4xx other than these) are not retried since retrying
# won't help.
_RETRYABLE_EXCEPTIONS = (httpx.TransportError, httpx.TimeoutException)

_TERMINAL_STATES_DONE = {"done", "success"}
_TERMINAL_STATES_FAILED = {"failed"}


class MinerUCloudError(RuntimeError):
    """Raised when the MinerU Cloud API fails, times out, or misbehaves."""


class MinerUCloudClient:
    """Client for MinerU's v4 "precision parse" Cloud API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        work_dir: Path | None = None,
        request_timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or config.MINERU_API_KEY
        if not self.api_key:
            raise MinerUCloudError(
                "MINERU_API_KEY is missing. Set it in your .env file, or pass "
                "--mode local to use the local MinerU pipeline instead."
            )

        self.base_url = (base_url or config.MINERU_API_BASE).rstrip("/")
        self.model_version = config.MINERU_MODEL_VERSION
        # Directory to mirror raw responses into for debugging (optional).
        self.work_dir = work_dir
        if self.work_dir is not None:
            self.work_dir.mkdir(parents=True, exist_ok=True)

        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": self.api_key,
                "Content-Type": "application/json",
                "Accept": "*/*",
            },
            timeout=request_timeout,
        )
        # Remember which file name to look up in batch results.
        self._file_name: str | None = None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MinerUCloudClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ── Public API ───────────────────────────────────────────────────────

    def upload_pdf(self, pdf_path: str) -> str:
        """Request a signed upload URL, PUT the PDF to it, and return the
        assigned batch_id (used for polling / result retrieval).

        The upload to Aliyun OSS can be slow/unstable on some networks. If
        the PUT fails partway (timeout/connection reset), a *fresh* signed
        URL is requested and the upload is retried from scratch, up to
        MINERU_UPLOAD_MAX_RETRIES times — reusing a signed URL after a
        failed/partial PUT is not reliable.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > config.MINERU_MAX_FILE_MB:
            raise MinerUCloudError(
                f"'{path.name}' is {size_mb:.1f} MB, which exceeds the "
                f"configured MinerU cloud limit of {config.MINERU_MAX_FILE_MB} MB "
                "(set MINERU_MAX_FILE_MB to override). Try --mode local instead."
            )

        self._file_name = path.name
        file_bytes = path.read_bytes()
        max_attempts = max(1, config.MINERU_UPLOAD_MAX_RETRIES)
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return self._request_url_and_upload(path, file_bytes, size_mb, attempt, max_attempts)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                logger.warning(
                    "Upload attempt %d/%d for %s failed (%s); requesting a fresh "
                    "signed URL and retrying...",
                    attempt,
                    max_attempts,
                    path.name,
                    exc,
                )

        raise MinerUCloudError(
            f"Failed to upload '{path.name}' to MinerU Cloud after {max_attempts} "
            f"attempts: {last_exc}\n"
            "This looks like a slow/unstable network path to MinerU's storage "
            "(Aliyun OSS, mainland China). If this keeps happening, consider "
            "configuring HTTP_PROXY/HTTPS_PROXY in .env, or try --mode local."
        ) from last_exc

    def _request_url_and_upload(
        self, path: Path, file_bytes: bytes, size_mb: float, attempt: int, max_attempts: int
    ) -> str:
        """One full attempt: request a fresh signed URL, then PUT the file to it."""
        # Step 1: request a pre-signed upload URL.
        logger.info(
            "Requesting MinerU upload URL for %s (%.1f MB) [attempt %d/%d]...",
            path.name,
            size_mb,
            attempt,
            max_attempts,
        )
        payload = {
            "files": [{"name": path.name, "data_id": path.stem}],
            "model_version": self.model_version,
        }
        response = self._request("POST", "/file-urls/batch", json=payload)
        data = self._parse_json(response, context="file-urls/batch")
        self._dump_debug("upload_request_response.json", data)

        if data.get("code") not in (0, None):
            raise MinerUCloudError(
                f"MinerU file-urls/batch request failed: {data.get('msg', data)}"
            )

        batch_id = (data.get("data") or {}).get("batch_id")
        file_urls = (data.get("data") or {}).get("file_urls") or []
        if not batch_id or not file_urls:
            raise MinerUCloudError(
                f"MinerU file-urls/batch response missing batch_id/file_urls: {data}"
            )
        upload_url = file_urls[0]

        # Step 2: PUT the raw file bytes to the signed URL (no Content-Type,
        # no auth header — this goes straight to object storage, not MinerU).
        # Generous, phase-specific timeouts: the connection to Aliyun OSS can
        # be slow on some networks, so the write phase gets the most room.
        logger.info("Uploading %s to signed URL...", path.name)
        upload_timeout = httpx.Timeout(
            connect=30.0, read=60.0, write=config.MINERU_UPLOAD_TIMEOUT_SECONDS, pool=30.0
        )
        upload_resp = httpx.put(upload_url, content=file_bytes, timeout=upload_timeout)
        if upload_resp.status_code >= 300:
            raise MinerUCloudError(
                f"Uploading '{path.name}' to signed URL failed: "
                f"HTTP {upload_resp.status_code} - {upload_resp.text[:500]}"
            )

        logger.info("MinerU Cloud batch created: %s", batch_id)
        self._dump_debug("batch_id.txt", batch_id, as_json=False)
        return batch_id

    def wait_for_completion(self, batch_id: str, timeout_minutes: int = 30) -> str:
        """Poll the batch status until the task reaches a terminal state.

        Returns the extracted markdown content on success. Raises
        ``MinerUCloudError`` on failure or ``TimeoutError`` if the task does
        not finish within ``timeout_minutes``.
        """
        poll_interval = config.MINERU_POLL_INTERVAL_SECONDS
        deadline = time.monotonic() + timeout_minutes * 60
        attempt = 0

        while True:
            attempt += 1
            entry = self._get_result_entry(batch_id)
            state = entry.get("state")
            progress = entry.get("extract_progress") or {}
            logger.info(
                "MinerU Cloud batch %s status: %s (%s/%s pages) [poll #%d]",
                batch_id,
                state,
                progress.get("extracted_pages", "?"),
                progress.get("total_pages", "?"),
                attempt,
            )

            if state in _TERMINAL_STATES_DONE:
                return self._download_markdown(entry, batch_id)

            if state in _TERMINAL_STATES_FAILED:
                raise MinerUCloudError(
                    f"MinerU Cloud batch {batch_id} failed: {entry.get('err_msg', 'unknown error')}"
                )

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"MinerU Cloud batch {batch_id} did not complete within "
                    f"{timeout_minutes} minutes (last state: {state}). "
                    "Try again later, or use --mode local."
                )

            time.sleep(poll_interval)

    def get_result(self, batch_id: str) -> str:
        """Fetch the current batch result and return markdown if done.

        Raises ``MinerUCloudError`` if the task isn't finished yet or failed.
        """
        entry = self._get_result_entry(batch_id)
        state = entry.get("state")
        if state in _TERMINAL_STATES_DONE:
            return self._download_markdown(entry, batch_id)
        if state in _TERMINAL_STATES_FAILED:
            raise MinerUCloudError(
                f"MinerU Cloud batch {batch_id} failed: {entry.get('err_msg', 'unknown error')}"
            )
        raise MinerUCloudError(
            f"MinerU Cloud batch {batch_id} is not finished yet (state: {state}); "
            "call wait_for_completion() to poll until it's done."
        )

    # ── Internals ────────────────────────────────────────────────────────

    def _get_result_entry(self, batch_id: str) -> dict:
        """GET /extract-results/batch/{batch_id} and return the entry that
        matches the file we uploaded (or the first entry as a fallback)."""
        response = self._request("GET", f"/extract-results/batch/{batch_id}")
        data = self._parse_json(response, context="extract-results")
        self._dump_debug("status_last.json", data)

        if data.get("code") not in (0, None):
            raise MinerUCloudError(
                f"MinerU extract-results request failed: {data.get('msg', data)}"
            )

        results = (data.get("data") or {}).get("extract_result") or []
        if not results:
            raise MinerUCloudError(
                f"MinerU extract-results response has no extract_result entries: {data}"
            )

        if self._file_name:
            for entry in results:
                if entry.get("file_name") == self._file_name:
                    return entry

        return results[0]

    def _download_markdown(self, entry: dict, batch_id: str) -> str:
        """Download the result ZIP (or plain markdown link) and return the
        markdown text, extracting the archive into work_dir for debugging."""
        zip_url = entry.get("full_zip_url")
        if not zip_url:
            # Some response shapes may return markdown directly.
            markdown = entry.get("markdown") or entry.get("md")
            if markdown:
                self._dump_debug("cloud_full.md", markdown, as_json=False)
                return markdown
            raise MinerUCloudError(
                f"MinerU Cloud result for batch {batch_id} has no full_zip_url: {entry}"
            )

        logger.info("Downloading MinerU result archive...")
        zip_resp = httpx.get(zip_url, timeout=300.0)
        zip_resp.raise_for_status()

        markdown = self._extract_markdown_from_zip(zip_resp.content)
        self._dump_debug("cloud_full.md", markdown, as_json=False)
        return markdown

    def _extract_markdown_from_zip(self, zip_bytes: bytes) -> str:
        extract_dir = self.work_dir / "cloud_extracted" if self.work_dir else None
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            if extract_dir is not None:
                extract_dir.mkdir(parents=True, exist_ok=True)
                zf.extractall(extract_dir)

            md_names = [n for n in zf.namelist() if n.lower().endswith(".md")]
            if not md_names:
                raise MinerUCloudError(
                    f"MinerU result archive has no .md file (contents: {zf.namelist()})"
                )
            # Prefer "full.md" if present, matching the local pipeline's output name.
            preferred = next((n for n in md_names if Path(n).name == "full.md"), md_names[0])
            return zf.read(preferred).decode("utf-8")

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except _RETRYABLE_EXCEPTIONS as exc:
            logger.warning("MinerU Cloud request %s %s failed: %s (retrying)", method, path, exc)
            raise

        if response.status_code >= 500:
            # Treat server errors as retryable via the same exception path.
            raise httpx.TransportError(
                f"MinerU Cloud returned HTTP {response.status_code} for {method} {path}"
            )

        if response.status_code >= 400:
            raise MinerUCloudError(
                f"MinerU Cloud request failed ({method} {path}): "
                f"HTTP {response.status_code} - {response.text[:500]}"
            )

        return response

    @staticmethod
    def _parse_json(response: httpx.Response, context: str) -> dict:
        try:
            return response.json()
        except ValueError as exc:
            raise MinerUCloudError(
                f"MinerU Cloud {context} response was not valid JSON: {response.text[:500]}"
            ) from exc

    def _dump_debug(self, filename: str, content, as_json: bool = True) -> None:
        """Best-effort mirror of intermediate results for debugging."""
        if self.work_dir is None:
            return
        try:
            path = self.work_dir / filename
            if as_json:
                path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
            else:
                path.write_text(str(content), encoding="utf-8")
        except OSError as exc:  # noqa: BLE001 - debugging aid only, never fatal
            logger.debug("Could not write debug file %s: %s", filename, exc)
