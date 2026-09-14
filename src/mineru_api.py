"""
MinerU cloud API (v4) -- the project's only PDF -> Markdown parser.

Flow
----
The ``/extract/task`` endpoint only accepts a publicly reachable URL, so a
local PDF goes up through the batch signed-upload flow instead:

1. ``POST /file-urls/batch``  -> ``{batch_id, file_urls: [<signed PUT url>, ...]}``
2. ``PUT <signed url>`` with the raw bytes. Uploading IS the submission --
   there is no separate "start parsing" call.
3. ``GET /extract-results/batch/{batch_id}`` -> poll each file's ``state``
   (``running`` -> ``done`` | ``failed``), then download ``full_zip_url``.

Two details that cost real debugging time and must not be "cleaned up":

* The upload PUT deliberately sends **no** ``Content-Type`` header. The docs
  say not to set one, and setting it invalidates the pre-signed URL's
  signature.
* The PUT also carries no ``Authorization`` header: it goes straight to
  object storage, not to MinerU. That is why storage traffic uses a separate
  client from the API traffic.

Why the PDF is split first
--------------------------
The API caps each file at 200 pages and 200 MB. A 516-page textbook goes up
as three parts, submitted in one batch (the endpoint takes up to 50 files)
and parsed in parallel. Splitting is delegated to ``src.splitter``, with zero
overlap: MinerU parses each part independently and the parts are concatenated
verbatim, so overlapping pages would be duplicated outright rather than
stitched.

Model choice
------------
``vlm`` (the default) produces markedly cleaner LaTeX than ``pipeline``:
``$10^{-20}$`` where pipeline gives ``$1 0 ^ { - 2 0 }$``, real heading
levels, and merged table cells preserved instead of collapsed. The tradeoff
is that a VLM backend can hallucinate -- for a physics textbook that means a
plausible equation with a wrong exponent, which no automated check will
catch. ``pipeline`` remains selectable per book for that reason.

Quota is 1000 pages/day at top priority, so the page count is logged before
anything is uploaded.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src import config
from src.splitter import split_pdf

logger = logging.getLogger(__name__)

MERGED_NAME = "merged.md"
IMAGES_DIRNAME = "images"
PARTS_DIRNAME = "parts"

# Hard API limits. `config.SPLIT_MAX_PAGES` / `SPLIT_MAX_SIZE_MB` are the
# (lower) targets we actually split to; these are the ceilings the service
# itself enforces, checked before upload so a violation is a local error with
# a clear message rather than an opaque -60005/-60006 from the server.
API_MAX_PAGES_PER_FILE = 200
API_MAX_FILE_MB = 200

_TERMINAL_DONE = {"done", "success"}
_TERMINAL_FAILED = {"failed"}

# Documented error codes worth translating into something actionable. The
# server's own `msg` is always appended, so an unlisted code still surfaces
# whatever the API said.
_ERROR_CODES: dict[str, str] = {
    "A0202": (
        "MINERU_TOKEN was rejected. Check the value in .env against the token "
        "shown at https://mineru.net/apiManage/token"
    ),
    "A0211": (
        "MINERU_TOKEN has expired. Issue a fresh one at "
        "https://mineru.net/apiManage/token and update .env."
    ),
    "-60005": (
        f"A part exceeded MinerU's {API_MAX_FILE_MB} MB per-file limit. Lower "
        "SPLIT_MAX_SIZE_MB so parts are rendered smaller."
    ),
    "-60006": (
        f"A part exceeded MinerU's {API_MAX_PAGES_PER_FILE}-page per-file "
        "limit. Lower SPLIT_MAX_PAGES."
    ),
    "-60018": (
        "The daily task limit has been reached. MinerU allows "
        f"{config.MINERU_DAILY_PAGE_QUOTA} pages/day at top priority; retry "
        "tomorrow or split the book across days."
    ),
}

_RETRYABLE = (httpx.TransportError, httpx.TimeoutException)


class MinerUError(RuntimeError):
    """Raised when the MinerU cloud API fails, times out, or misbehaves."""


def _explain(code: object, msg: object) -> str:
    """Render an API error code plus the server's own message."""
    key = str(code)
    hint = _ERROR_CODES.get(key)
    detail = f"{msg}" if msg else "no message"
    return f"{hint} (code {key}: {detail})" if hint else f"MinerU error {key}: {detail}"


@dataclass(frozen=True)
class PartResult:
    """One uploaded part's outcome, in document order."""

    index: int
    file_name: str
    state: str
    err_msg: str = ""
    pages: int = 0

    @property
    def ok(self) -> bool:
        return self.state in _TERMINAL_DONE


@dataclass(frozen=True)
class ParseResult:
    """Everything ``parse_pdf`` produced for one book."""

    merged_path: Path
    images_dir: Path
    work_dir: Path
    total_pages: int
    parts: tuple[PartResult, ...]

    @property
    def failed_parts(self) -> tuple[PartResult, ...]:
        return tuple(p for p in self.parts if not p.ok)


class MinerUClient:
    """Thin client for MinerU v4's batch extract endpoints.

    ``transport`` is injected by the test suite (an ``httpx.MockTransport``)
    so the whole split -> upload -> poll -> merge flow can be exercised with
    no network access.
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        base_url: str | None = None,
        model_version: str | None = None,
        language: str | None = None,
        transport: httpx.BaseTransport | None = None,
        request_timeout: float = 60.0,
    ) -> None:
        self.token = token or config.MINERU_TOKEN
        if not self.token:
            raise MinerUError(
                "MINERU_TOKEN is missing. Create a `.env` file in the project "
                "root and set MINERU_TOKEN=<your-token>, or pass --token. Get "
                "a token at https://mineru.net/apiManage/token"
            )

        self.base_url = (base_url or config.MINERU_API_BASE).rstrip("/")
        self.model_version = model_version or config.MINERU_MODEL_VERSION
        self.language = language or config.MINERU_LANGUAGE

        self._api = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "*/*",
            },
            timeout=request_timeout,
            transport=transport,
        )
        # Object storage (pre-signed OSS URLs) must see neither our
        # Authorization header nor a Content-Type -- either one breaks the
        # signature. Hence a second, header-free client.
        self._storage = httpx.Client(timeout=request_timeout, transport=transport)

    def close(self) -> None:
        self._api.close()
        self._storage.close()

    def __enter__(self) -> "MinerUClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ── Submit ───────────────────────────────────────────────────────────

    def submit(self, parts: list[Path]) -> str:
        """Request upload URLs for ``parts``, PUT each file, return the batch id.

        Uploading a file automatically submits its parse task.
        """
        for part in parts:
            size_mb = part.stat().st_size / (1024 * 1024)
            if size_mb > API_MAX_FILE_MB:
                raise MinerUError(
                    f"'{part.name}' is {size_mb:.1f} MB, over MinerU's hard "
                    f"{API_MAX_FILE_MB} MB per-file limit. Lower SPLIT_MAX_SIZE_MB."
                )

        payload = {
            "files": [{"name": p.name, "is_ocr": True, "data_id": p.stem} for p in parts],
            "model_version": self.model_version,
            "language": self.language,
            "enable_formula": True,
            "enable_table": True,
        }
        data = self._json(self._request("POST", "/file-urls/batch", json=payload))
        if data.get("code") not in (0, None):
            raise MinerUError(_explain(data.get("code"), data.get("msg")))

        body = data.get("data") or {}
        batch_id = body.get("batch_id")
        urls = body.get("file_urls") or []
        if not batch_id or len(urls) != len(parts):
            raise MinerUError(
                f"file-urls/batch returned {len(urls)} URL(s) for {len(parts)} "
                f"part(s) and batch_id={batch_id!r}: {data}"
            )

        logger.info("MinerU batch %s created for %d part(s)", batch_id, len(parts))
        for part, url in zip(parts, urls):
            self._upload(part, url)
        return batch_id

    def _upload(self, part: Path, url: str) -> None:
        """PUT one part to its pre-signed URL, retrying the whole request.

        A failed/partial PUT is retried against the same URL; if the URL
        itself has gone stale the caller sees the final error rather than a
        silent half-upload.
        """
        size_mb = part.stat().st_size / (1024 * 1024)
        attempts = max(1, config.MINERU_UPLOAD_MAX_RETRIES)
        timeout = httpx.Timeout(
            connect=30.0, read=60.0, write=config.MINERU_UPLOAD_TIMEOUT_SECONDS, pool=30.0
        )
        last: Exception | None = None

        for attempt in range(1, attempts + 1):
            logger.info(
                "Uploading %s (%.1f MB) [attempt %d/%d]", part.name, size_mb, attempt, attempts
            )
            try:
                # No Content-Type, no Authorization: both break the signature.
                response = self._storage.put(url, content=part.read_bytes(), timeout=timeout)
            except _RETRYABLE as exc:
                last = exc
                logger.warning("Upload of %s failed (%s); retrying", part.name, exc)
                continue

            if response.status_code in (200, 201):
                return
            raise MinerUError(
                f"Uploading '{part.name}' failed: HTTP {response.status_code} - "
                f"{response.text[:500]}"
            )

        raise MinerUError(
            f"Failed to upload '{part.name}' after {attempts} attempt(s): {last}. "
            "MinerU's storage sits behind Aliyun OSS; if this keeps happening, "
            "set HTTP_PROXY/HTTPS_PROXY in .env."
        ) from last

    # ── Poll ─────────────────────────────────────────────────────────────

    def poll(
        self,
        batch_id: str,
        *,
        interval: float | None = None,
        timeout_minutes: int | None = None,
    ) -> list[dict]:
        """Poll until every file in the batch is ``done`` or ``failed``.

        Returns the raw result entries. A part that failed is reported, not
        raised on -- a 3-part book where part 2 failed should still produce
        the two parts that worked, with a loud warning.
        """
        interval = config.MINERU_POLL_INTERVAL_SECONDS if interval is None else interval
        limit = timeout_minutes if timeout_minutes is not None else config.MINERU_TIMEOUT_MINUTES
        deadline = time.monotonic() + limit * 60

        while True:
            data = self._json(self._request("GET", f"/extract-results/batch/{batch_id}"))
            if data.get("code") not in (0, None):
                raise MinerUError(_explain(data.get("code"), data.get("msg")))

            results = (data.get("data") or {}).get("extract_result") or []
            if not results:
                raise MinerUError(f"Batch {batch_id} returned no extract_result entries: {data}")

            logger.info("MinerU batch %s: %s", batch_id, _progress_line(results))

            if all(r.get("state") in _TERMINAL_DONE | _TERMINAL_FAILED for r in results):
                return results

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"MinerU batch {batch_id} did not finish within {limit} minutes "
                    f"({_progress_line(results)}). The batch keeps running server-side; "
                    f"re-run with MINERU_TIMEOUT_MINUTES raised, or check "
                    f"{self.base_url}/extract-results/batch/{batch_id}"
                )

            time.sleep(interval)

    # ── Download ─────────────────────────────────────────────────────────

    def download_part(self, entry: dict, images_dir: Path) -> str:
        """Download one part's result ZIP; return its markdown and extract its
        images into the shared flat ``images_dir``.

        Image filenames are content hashes, so they are unique across parts
        and share one directory without collision.
        """
        url = entry.get("full_zip_url")
        if not url:
            raise MinerUError(
                f"Result for {entry.get('file_name')!r} has no full_zip_url: {entry}"
            )

        response = self._storage.get(url, timeout=1800.0, follow_redirects=True)
        response.raise_for_status()
        return _unpack_zip(response.content, images_dir, entry.get("file_name", "?"))

    # ── Internals ────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        try:
            response = self._api.request(method, path, **kwargs)
        except _RETRYABLE as exc:
            logger.warning("MinerU %s %s failed: %s (retrying)", method, path, exc)
            raise

        if response.status_code >= 500:
            # Route server errors through the same retry path.
            raise httpx.TransportError(f"MinerU returned HTTP {response.status_code} for {path}")
        if response.status_code >= 400:
            raise MinerUError(
                f"MinerU {method} {path} failed: HTTP {response.status_code} - "
                f"{response.text[:500]}"
            )
        return response

    @staticmethod
    def _json(response: httpx.Response) -> dict:
        try:
            payload = response.json()
        except ValueError as exc:
            raise MinerUError(
                f"MinerU response was not valid JSON: {response.text[:500]}"
            ) from exc
        if not isinstance(payload, dict):
            raise MinerUError(f"MinerU response was not a JSON object: {payload!r}")
        return payload


def _progress_line(results: list[dict]) -> str:
    """One-line per-file progress, e.g. ``part01: 120/190 | part02: done``."""
    fields = []
    for entry in results:
        name = entry.get("file_name", "?")
        state = entry.get("state")
        if state == "running":
            progress = entry.get("extract_progress") or {}
            extracted = progress.get("extracted_pages", "?")
            total = progress.get("total_pages", "?")
            fields.append(f"{name}: {extracted}/{total} pages")
        else:
            fields.append(f"{name}: {state}")
    return " | ".join(fields)


def _unpack_zip(blob: bytes, images_dir: Path, label: str) -> str:
    """Pull the markdown out of a result archive and flatten its images."""
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
        md_names = [n for n in names if n.lower().endswith(".md")]
        if not md_names:
            raise MinerUError(f"Result archive for {label} has no .md file (contents: {names})")
        # 'full.md' is the documented markdown output name.
        chosen = next((n for n in md_names if Path(n).name == "full.md"), md_names[0])
        markdown = archive.read(chosen).decode("utf-8")

        images_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            if name.endswith("/") or f"/{IMAGES_DIRNAME}/" not in f"/{name}":
                continue
            target = images_dir / Path(name).name
            if not target.exists():
                target.write_bytes(archive.read(name))

    return markdown


def _page_count(pdf_path: Path) -> int:
    import pymupdf

    try:
        with pymupdf.open(pdf_path) as doc:
            return doc.page_count
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RuntimeError(f"Failed to open PDF '{pdf_path}': {exc}") from exc


def parse_pdf(
    pdf_path: Path,
    work_dir: Path,
    *,
    client: MinerUClient | None = None,
    token: str | None = None,
    model_version: str | None = None,
    language: str | None = None,
    force: bool = False,
) -> ParseResult:
    """Parse ``pdf_path`` with the MinerU cloud API into ``work_dir``.

    Writes ``work_dir/merged.md`` (all parts concatenated, in document order)
    and ``work_dir/images/`` (flat, shared by every part). Part PDFs land in
    ``work_dir/parts/``.

    Args:
        pdf_path: The book's PDF. Need not live under ``data/input``.
        work_dir: This book's ``data/work/<book_name>`` directory.
        client: Pre-built client, mainly for tests. One is constructed from
            config when omitted.
        token: Overrides ``MINERU_TOKEN``.
        model_version: ``"vlm"`` (default) or ``"pipeline"``.
        language: MinerU OCR language hint, default ``"ch"``.
        force: Re-parse even when ``merged.md`` is already up to date.

    Returns:
        A ``ParseResult``. Parts that failed server-side are reported in
        ``ParseResult.failed_parts`` rather than raised on, so a partial book
        is still usable.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        MinerUError: On a token, quota or protocol failure.
        TimeoutError: If the batch does not finish in time.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    merged_path = work_dir / MERGED_NAME
    images_dir = work_dir / IMAGES_DIRNAME

    if not force and merged_path.exists() and merged_path.stat().st_mtime > pdf_path.stat().st_mtime:
        logger.info("Already parsed, skipping: %s", merged_path)
        return ParseResult(
            merged_path=merged_path,
            images_dir=images_dir,
            work_dir=work_dir,
            total_pages=_page_count(pdf_path),
            parts=(),
        )

    total_pages = _page_count(pdf_path)
    quota = config.MINERU_DAILY_PAGE_QUOTA
    logger.info(
        "'%s' is %d page(s) -- %.0f%% of MinerU's %d pages/day top-priority quota",
        pdf_path.name,
        total_pages,
        100 * total_pages / quota if quota else 0,
        quota,
    )

    # Zero overlap: MinerU parses each part independently and the parts are
    # concatenated verbatim, so overlapping pages would appear twice.
    split = split_pdf(
        pdf_path,
        work_dir / PARTS_DIRNAME,
        max_pages=min(config.SPLIT_MAX_PAGES, API_MAX_PAGES_PER_FILE),
        max_size_mb=min(config.SPLIT_MAX_SIZE_MB, API_MAX_FILE_MB),
        overlap_pages=0,
    )
    parts = [chunk.output_path for chunk in split.chunks]
    logger.info("Submitting %d part(s) to MinerU", len(parts))

    owned = client is None
    client = client or MinerUClient(
        token=token, model_version=model_version, language=language
    )
    try:
        batch_id = client.submit(parts)
        results = client.poll(batch_id)

        by_name = {entry.get("file_name"): entry for entry in results}
        sections: list[str] = []
        part_results: list[PartResult] = []

        for chunk in split.chunks:
            name = chunk.output_path.name
            entry = by_name.get(name)
            if entry is None:
                part_results.append(
                    PartResult(chunk.index, name, "missing", "no result entry returned")
                )
                logger.error("MinerU returned no result for %s", name)
                continue

            state = str(entry.get("state"))
            if state not in _TERMINAL_DONE:
                err = str(entry.get("err_msg") or "unknown error")
                part_results.append(PartResult(chunk.index, name, state, err))
                logger.error("Part %s %s: %s", name, state, err)
                continue

            sections.append(client.download_part(entry, images_dir))
            part_results.append(
                PartResult(chunk.index, name, state, pages=chunk.page_count)
            )
    finally:
        if owned:
            client.close()

    if not sections:
        raise MinerUError(
            f"Every part of '{pdf_path.name}' failed to parse: "
            + "; ".join(f"{p.file_name}: {p.err_msg}" for p in part_results)
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    merged_path.write_text("\n\n".join(s.strip("\n") for s in sections) + "\n", encoding="utf-8")

    text_len = len(merged_path.read_text(encoding="utf-8"))
    n_images = sum(1 for _ in images_dir.iterdir()) if images_dir.exists() else 0
    logger.info("Wrote %s (%d chars) and %d image(s)", merged_path, text_len, n_images)

    result = ParseResult(
        merged_path=merged_path,
        images_dir=images_dir,
        work_dir=work_dir,
        total_pages=total_pages,
        parts=tuple(part_results),
    )
    for failed in result.failed_parts:
        logger.warning(
            "Part %s is MISSING from %s (%s: %s)",
            failed.file_name,
            merged_path.name,
            failed.state,
            failed.err_msg,
        )
    return result


__all__ = [
    "API_MAX_FILE_MB",
    "API_MAX_PAGES_PER_FILE",
    "MERGED_NAME",
    "MinerUClient",
    "MinerUError",
    "ParseResult",
    "PartResult",
    "parse_pdf",
]
