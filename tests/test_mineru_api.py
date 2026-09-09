#!/usr/bin/env python3
"""
Parse a PDF with the MinerU cloud API (v4 precise-parsing endpoint).

Handles the whole flow: split -> request upload URLs -> PUT files -> poll ->
download result zips -> merge into one markdown + one images/ folder.

Usage:
    export MINERU_TOKEN=...            # or pass --token
    python mineru_cloud.py book.pdf -o out/

Why splitting is needed
-----------------------
The API caps each file at 200 pages and 200 MB. A 516-page textbook therefore
goes up as three files. The batch endpoint accepts up to 50 files per request,
so all parts are submitted together and parsed in parallel.

Model choice
------------
`vlm` is what MinerU's own docs recommend, and it produces markedly cleaner
LaTeX than `pipeline` -- compare "10^{-20}" against pipeline's "1 0 ^ { - 2 0 }".
The tradeoff is that VLM backends can hallucinate, which for a physics textbook
means a plausible-looking equation with a wrong exponent that no automated check
will catch. Parse a chapter with each and diff the formulas before committing to
`vlm` for a whole book.

Quota: 1000 pages/day at highest priority; beyond that, lower priority.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

API = "https://mineru.net/api/v4"
MAX_PAGES = 200  # hard API limit per file


def split_pdf(pdf: Path, out_dir: Path, max_pages: int = MAX_PAGES) -> list[Path]:
    """Split into <=max_pages parts. Returns the parts in order."""
    import pymupdf

    doc = pymupdf.open(pdf)
    total = doc.page_count
    if total <= max_pages:
        doc.close()
        return [pdf]

    out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    for i, start in enumerate(range(0, total, max_pages), 1):
        end = min(start + max_pages, total) - 1
        part = pymupdf.open()
        part.insert_pdf(doc, from_page=start, to_page=end)
        path = out_dir / f"{pdf.stem}_part{i:02d}.pdf"
        part.save(path)
        part.close()
        parts.append(path)
        print(f"  part {i}: pages {start + 1}-{end + 1} -> {path.name}")
    doc.close()
    return parts


def submit(parts: list[Path], token: str, model: str, language: str) -> str:
    """Request upload URLs, PUT each file, return the batch id.

    Uploading a file automatically submits its parse task -- there is no
    separate submit call.
    """
    body = {
        "files": [{"name": p.name, "is_ocr": True} for p in parts],
        "model_version": model,
        "language": language,
        "enable_formula": True,
        "enable_table": True,
    }
    r = requests.post(
        f"{API}/file-urls/batch",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        json=body,
        timeout=60,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"submit failed: {payload.get('msg')} (code {payload.get('code')})")

    batch_id = payload["data"]["batch_id"]
    urls = payload["data"]["file_urls"]
    print(f"batch {batch_id}")

    for path, url in zip(parts, urls):
        size_mb = path.stat().st_size / 1e6
        print(f"  uploading {path.name} ({size_mb:.1f} MB)...", end=" ", flush=True)
        with path.open("rb") as f:
            # Deliberately no Content-Type header: the docs say not to set one.
            up = requests.put(url, data=f, timeout=3600)
        print("ok" if up.status_code in (200, 201) else f"FAILED {up.status_code}")
        if up.status_code not in (200, 201):
            raise RuntimeError(f"upload failed for {path.name}")
    return batch_id


def poll(batch_id: str, token: str, interval: int = 10, timeout: int = 7200) -> list[dict]:
    """Poll until every file reaches done or failed."""
    url = f"{API}/extract-results/batch/{batch_id}"
    headers = {"Authorization": f"Bearer {token}"}
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        results = r.json()["data"]["extract_result"]

        parts = []
        for res in results:
            state = res.get("state")
            if state == "running":
                p = res.get("extract_progress") or {}
                parts.append(f"{res['file_name']}: {p.get('extracted_pages', '?')}/{p.get('total_pages', '?')}")
            else:
                parts.append(f"{res['file_name']}: {state}")
        print(f"  [{int(time.time() - start):>4}s] " + " | ".join(parts))

        if all(r_.get("state") in ("done", "failed") for r_ in results):
            return results
        time.sleep(interval)
    raise TimeoutError(f"timed out after {timeout}s; batch_id={batch_id}")


def download_and_merge(results: list[dict], out_dir: Path) -> Path:
    """Download each result zip and merge into one markdown + images/ folder.

    Image filenames are content hashes, so they are unique across parts and can
    share one flat images/ directory without collision.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    images = out_dir / "images"
    images.mkdir(exist_ok=True)

    chunks: list[str] = []
    for res in sorted(results, key=lambda r: r["file_name"]):
        if res.get("state") != "done":
            print(f"  SKIPPING {res['file_name']}: {res.get('state')} - {res.get('err_msg')}")
            continue
        print(f"  downloading {res['file_name']}...", end=" ", flush=True)
        blob = requests.get(res["full_zip_url"], timeout=1800).content
        print(f"{len(blob) / 1e6:.1f} MB")

        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            md_names = [n for n in z.namelist() if n.endswith(".md")]
            # 'full.md' is the documented markdown output name.
            md_name = next((n for n in md_names if Path(n).name == "full.md"), md_names[0])
            chunks.append(z.read(md_name).decode("utf-8"))
            for name in z.namelist():
                if "/images/" in name and not name.endswith("/"):
                    target = images / Path(name).name
                    if not target.exists():
                        target.write_bytes(z.read(name))

    merged = out_dir / "merged.md"
    merged.write_text("\n\n".join(chunks), encoding="utf-8")
    n_images = sum(1 for _ in images.iterdir())
    print(f"\nwrote {merged} ({len(merged.read_text(encoding='utf-8')):,} chars)")
    print(f"      {images} ({n_images} files)")
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("mineru_out"))
    ap.add_argument("--token", default=os.environ.get("MINERU_TOKEN"))
    ap.add_argument("--model", default="vlm", choices=["vlm", "pipeline"])
    ap.add_argument("--language", default="ch")
    args = ap.parse_args()

    if not args.token:
        print("error: set MINERU_TOKEN or pass --token")
        return 1
    if not args.pdf.exists():
        print(f"error: {args.pdf} not found")
        return 1

    print(f"splitting {args.pdf.name} (max {MAX_PAGES} pages per part)...")
    parts = split_pdf(args.pdf, args.out / "parts")

    print(f"\nsubmitting {len(parts)} file(s), model={args.model}...")
    batch_id = submit(parts, args.token, args.model, args.language)

    print("\npolling...")
    results = poll(batch_id, args.token)

    failed = [r for r in results if r.get("state") == "failed"]
    if failed:
        print(f"\n{len(failed)} file(s) failed:")
        for r in failed:
            print(f"  {r['file_name']}: {r.get('err_msg')}")

    print("\ndownloading results...")
    download_and_merge(results, args.out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())