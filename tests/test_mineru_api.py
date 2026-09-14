"""Tests for src.mineru_api (split -> upload -> poll -> merge).

Every HTTP call is served by an ``httpx.MockTransport``: the suite must never
touch the network or spend a page of the daily quota.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import httpx
import pymupdf
import pytest

from src import config
from src.mineru_api import (
    MinerUClient,
    MinerUError,
    parse_pdf,
)

BATCH_ID = "batch-abc123"
ZIP_URL = "https://storage.example.com/results/{name}.zip"
UPLOAD_URL = "https://oss.example.com/upload/{name}?signature=deadbeef"


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_pdf(path: Path, num_pages: int) -> None:
    doc = pymupdf.open()
    for i in range(num_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}")
    doc.save(path)
    doc.close()


def _make_zip(markdown: str, images: dict[str, bytes] | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("full.md", markdown)
        archive.writestr("layout.json", "{}")
        for name, blob in (images or {}).items():
            archive.writestr(f"images/{name}", blob)
    return buffer.getvalue()


class FakeAPI:
    """Scripted MinerU. Records every request so tests can assert on headers.

    ``poll_states`` is a list of per-poll state lists, so a test can script a
    batch that reports ``running`` before it reports ``done``.
    """

    def __init__(
        self,
        *,
        markdown: dict[str, str] | None = None,
        images: dict[str, bytes] | None = None,
        poll_states: list[list[str]] | None = None,
        submit_code: int = 0,
        submit_msg: str = "ok",
        err_msgs: dict[str, str] | None = None,
        upload_status: int = 200,
    ) -> None:
        self.markdown = markdown or {}
        self.images = images or {}
        self.poll_states = poll_states
        self.submit_code = submit_code
        self.submit_msg = submit_msg
        self.err_msgs = err_msgs or {}
        self.upload_status = upload_status

        self.requests: list[httpx.Request] = []
        self.uploaded: dict[str, bytes] = {}
        self.file_names: list[str] = []
        self.submit_payload: dict | None = None
        self.poll_count = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)

        if request.method == "POST" and "/file-urls/batch" in url:
            return self._submit(request)
        if request.method == "PUT":
            name = url.split("/upload/")[1].split("?")[0]
            self.uploaded[name] = request.content
            return httpx.Response(self.upload_status)
        if request.method == "GET" and "/extract-results/batch/" in url:
            return self._poll()
        if request.method == "GET" and "/results/" in url:
            name = url.split("/results/")[1].removesuffix(".zip")
            return httpx.Response(200, content=_make_zip(self.markdown[name], self.images))

        return httpx.Response(404, text=f"unexpected request: {request.method} {url}")

    def _submit(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        self.submit_payload = payload
        self.file_names = [f["name"] for f in payload["files"]]
        if self.submit_code != 0:
            return httpx.Response(
                200, json={"code": self.submit_code, "msg": self.submit_msg, "data": None}
            )
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": "ok",
                "data": {
                    "batch_id": BATCH_ID,
                    "file_urls": [UPLOAD_URL.format(name=n) for n in self.file_names],
                },
            },
        )

    def _poll(self) -> httpx.Response:
        states = (
            self.poll_states[min(self.poll_count, len(self.poll_states) - 1)]
            if self.poll_states
            else ["done"] * len(self.file_names)
        )
        self.poll_count += 1

        entries = []
        for name, state in zip(self.file_names, states):
            entry: dict = {"file_name": name, "state": state}
            if state == "running":
                entry["extract_progress"] = {"extracted_pages": 40, "total_pages": 190}
            elif state == "done":
                entry["full_zip_url"] = ZIP_URL.format(name=name.removesuffix(".pdf"))
            else:
                entry["err_msg"] = self.err_msgs.get(name, "something went wrong")
            entries.append(entry)

        return httpx.Response(200, json={"code": 0, "data": {"extract_result": entries}})


@pytest.fixture(autouse=True)
def _no_poll_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MINERU_POLL_INTERVAL_SECONDS", 0)


# ── Token handling ──────────────────────────────────────────────────────────


def test_missing_token_raises_with_actionable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MINERU_TOKEN", None)
    with pytest.raises(MinerUError, match="MINERU_TOKEN"):
        MinerUClient()


def test_token_is_sent_as_a_bearer_header() -> None:
    api = FakeAPI(markdown={"book": "# Title"})
    with MinerUClient(token="tok", transport=api.transport()) as client:
        client.submit([])
    assert api.requests[0].headers["Authorization"] == "Bearer tok"


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("A0202", "rejected"),
        ("A0211", "expired"),
        ("-60005", "MB per-file limit"),
        ("-60006", "page per-file"),
        ("-60018", "daily task limit"),
    ],
)
def test_documented_error_codes_become_useful_messages(
    tmp_path: Path, code: str, expected: str
) -> None:
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf, 2)
    api = FakeAPI(submit_code=code, submit_msg="server said no")

    with MinerUClient(token="tok", transport=api.transport()) as client:
        with pytest.raises(MinerUError) as excinfo:
            client.submit([pdf])

    message = str(excinfo.value)
    assert expected in message
    # The server's own message is always appended, so an unlisted code still
    # surfaces whatever the API said.
    assert "server said no" in message


def test_undocumented_error_code_still_surfaces_the_server_message(tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf, 2)
    api = FakeAPI(submit_code=-99999, submit_msg="brand new failure")

    with MinerUClient(token="tok", transport=api.transport()) as client:
        with pytest.raises(MinerUError, match="brand new failure"):
            client.submit([pdf])


# ── Upload ──────────────────────────────────────────────────────────────────


def test_upload_sends_no_content_type_and_no_auth_header(tmp_path: Path) -> None:
    """Both headers break the pre-signed URL's signature."""
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf, 2)
    api = FakeAPI()

    with MinerUClient(token="tok", transport=api.transport()) as client:
        client.submit([pdf])

    put = next(r for r in api.requests if r.method == "PUT")
    assert "content-type" not in {k.lower() for k in put.headers}
    assert "authorization" not in {k.lower() for k in put.headers}
    assert api.uploaded["book.pdf"] == pdf.read_bytes()


def test_submit_requests_vlm_and_ocr_by_default(tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf, 2)
    api = FakeAPI()

    with MinerUClient(token="tok", transport=api.transport()) as client:
        client.submit([pdf])

    assert api.submit_payload["model_version"] == "vlm"
    assert api.submit_payload["enable_formula"] is True
    assert api.submit_payload["enable_table"] is True
    assert api.submit_payload["files"][0]["is_ocr"] is True


def test_pipeline_model_is_selectable(tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf, 2)
    api = FakeAPI()

    with MinerUClient(token="tok", model_version="pipeline", transport=api.transport()) as client:
        client.submit([pdf])

    assert api.submit_payload["model_version"] == "pipeline"


def test_failed_upload_status_raises(tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf, 2)
    api = FakeAPI(upload_status=403)

    with MinerUClient(token="tok", transport=api.transport()) as client:
        with pytest.raises(MinerUError, match="403"):
            client.submit([pdf])


# ── Poll ────────────────────────────────────────────────────────────────────


def test_poll_waits_for_running_then_returns_terminal_states(tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf, 2)
    api = FakeAPI(markdown={"book": "# Title"}, poll_states=[["running"], ["running"], ["done"]])

    with MinerUClient(token="tok", transport=api.transport()) as client:
        client.submit([pdf])
        results = client.poll(BATCH_ID)

    assert api.poll_count == 3
    assert [r["state"] for r in results] == ["done"]


def test_poll_times_out_rather_than_hanging(tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf, 2)
    api = FakeAPI(poll_states=[["running"]])

    with MinerUClient(token="tok", transport=api.transport()) as client:
        client.submit([pdf])
        with pytest.raises(TimeoutError, match=BATCH_ID):
            client.poll(BATCH_ID, timeout_minutes=0)


# ── Full flow ───────────────────────────────────────────────────────────────


def test_parse_pdf_splits_uploads_polls_and_merges(tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf, 400)  # over SPLIT_MAX_PAGES, so this must split
    work_dir = tmp_path / "work"

    api = FakeAPI(
        markdown={
            "book_part_001": "# Part one\n\n![](images/aaa.jpg)",
            "book_part_002": "# Part two\n\n![](images/bbb.jpg)",
            "book_part_003": "# Part three",
        },
        images={"aaa.jpg": b"\x89PNG-a", "bbb.jpg": b"\x89PNG-b"},
    )

    with MinerUClient(token="tok", transport=api.transport()) as client:
        result = parse_pdf(pdf, work_dir, client=client)

    assert result.total_pages == 400
    assert len(api.uploaded) == 3
    assert not result.failed_parts

    merged = result.merged_path.read_text(encoding="utf-8")
    # Parts land in document order, not upload or completion order.
    assert merged.index("Part one") < merged.index("Part two") < merged.index("Part three")

    # Images from every part share one flat directory: MinerU names them by
    # content hash, so they cannot collide.
    assert (work_dir / "images" / "aaa.jpg").read_bytes() == b"\x89PNG-a"
    assert (work_dir / "images" / "bbb.jpg").read_bytes() == b"\x89PNG-b"


def test_small_pdf_is_sent_as_one_part(tmp_path: Path) -> None:
    pdf = tmp_path / "small.pdf"
    _make_pdf(pdf, 5)
    api = FakeAPI(markdown={"small_part_001": "# Small"})

    with MinerUClient(token="tok", transport=api.transport()) as client:
        result = parse_pdf(pdf, tmp_path / "work", client=client)

    assert len(api.uploaded) == 1
    assert "Small" in result.merged_path.read_text(encoding="utf-8")


def test_one_failed_part_is_reported_and_the_rest_still_merge(tmp_path: Path) -> None:
    """A 3-part book where part 2 failed still produces the two that worked."""
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf, 400)
    work_dir = tmp_path / "work"

    api = FakeAPI(
        markdown={"book_part_001": "# Part one", "book_part_003": "# Part three"},
        poll_states=[["done", "failed", "done"]],
        err_msgs={"book_part_002.pdf": "page 214 is not renderable"},
    )

    with MinerUClient(token="tok", transport=api.transport()) as client:
        result = parse_pdf(pdf, work_dir, client=client)

    failed = result.failed_parts
    assert len(failed) == 1
    assert failed[0].file_name == "book_part_002.pdf"
    assert "not renderable" in failed[0].err_msg

    merged = result.merged_path.read_text(encoding="utf-8")
    assert "Part one" in merged and "Part three" in merged


def test_every_part_failing_raises_rather_than_writing_an_empty_book(tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf, 5)
    api = FakeAPI(poll_states=[["failed"]], err_msgs={"book_part_001.pdf": "quota exhausted"})

    with MinerUClient(token="tok", transport=api.transport()) as client:
        with pytest.raises(MinerUError, match="quota exhausted"):
            parse_pdf(pdf, tmp_path / "work", client=client)


def test_reparse_is_skipped_when_merged_md_is_current(tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf, 5)
    work_dir = tmp_path / "work"
    api = FakeAPI(markdown={"book_part_001": "# Book"})

    with MinerUClient(token="tok", transport=api.transport()) as client:
        parse_pdf(pdf, work_dir, client=client)
        first_uploads = len(api.uploaded)
        parse_pdf(pdf, work_dir, client=client)

    assert len(api.uploaded) == first_uploads  # nothing re-uploaded


def test_missing_pdf_raises_file_not_found(tmp_path: Path) -> None:
    api = FakeAPI()
    with MinerUClient(token="tok", transport=api.transport()) as client:
        with pytest.raises(FileNotFoundError):
            parse_pdf(tmp_path / "nope.pdf", tmp_path / "work", client=client)
