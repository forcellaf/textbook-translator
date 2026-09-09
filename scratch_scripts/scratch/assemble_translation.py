#!/usr/bin/env python3
"""
Reassemble a manually-translated book.

Takes the translation kit produced by the Kaggle notebook plus your translated
chunks, and produces a single Markdown file with the real image filenames
restored -- optionally converting it to PDF with pandoc.

Usage
-----
    python assemble_translation.py <kit-folder>
    python assemble_translation.py <kit-folder> --pdf

Expected layout inside <kit-folder>:

    chunks/001.md, 002.md, ...      the source chunks you pasted into AI Studio
    translated/001.md, 002.md, ...  the replies you saved (same numbers)
    images/                          the extracted figures
    image_map.json                   IMG token -> real filename
    source_clean.md                  the untranslated source, for comparison

Why this script exists rather than just `cat`
----------------------------------------------
Three things need checking, and all three fail silently otherwise:

1. **Image tokens.** The notebook replaced 64-character hash filenames with
   short `IMG_nnnn` tokens because models drop characters from long opaque
   strings (measured at ~8% in testing). This restores the real paths and
   verifies every token survived -- a corrupted or dropped token becomes a
   missing figure at PDF-build time with no earlier warning.
2. **Untranslated chunks.** A model can return a fluent, correctly-formatted
   response that is still in the source language. This reports per-chunk
   source-script density so a skipped chunk is obvious.
3. **Truncated replies.** A reply that got cut off mid-document looks fine in
   isolation. Comparing each translation's length against its source catches it.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

IMG_TOKEN_RE = re.compile(r"IMG_(\d{4})")
IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
MATH_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$", re.DOTALL)


def cjk_ratio(text: str) -> float:
    """Fraction of non-whitespace, non-math characters in CJK ranges."""
    stripped = MATH_RE.sub("", text)
    chars = [c for c in stripped if not c.isspace()]
    if not chars:
        return 0.0
    cjk = sum(1 for c in chars if "\u4e00" <= c <= "\u9fff")
    return cjk / len(chars)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kit", type=Path, help="the unzipped translation kit folder")
    ap.add_argument("--pdf", action="store_true", help="also run pandoc to build a PDF")
    ap.add_argument(
        "--cjk-threshold",
        type=float,
        default=0.05,
        help="warn when a chunk is more than this fraction CJK (default 0.05)",
    )
    args = ap.parse_args()

    kit: Path = args.kit
    chunks_dir = kit / "chunks"
    trans_dir = kit / "translated"

    if not chunks_dir.is_dir():
        print(f"error: {chunks_dir} not found - is that the right folder?")
        return 1
    if not trans_dir.is_dir():
        print(f"error: {trans_dir} not found.")
        print("Create it and save each AI Studio reply as translated/<same-number>.md")
        return 1

    sources = sorted(chunks_dir.glob("*.md"))
    print(f"{len(sources)} source chunk(s) in {chunks_dir}\n")

    parts: list[str] = []
    problems = 0

    print(f"{'chunk':>7} {'source':>9} {'translated':>11} {'CJK':>7}  status")
    for src_path in sources:
        tgt_path = trans_dir / src_path.name
        if not tgt_path.exists():
            print(f"{src_path.stem:>7} {len(src_path.read_text(encoding='utf-8')):>9,} "
                  f"{'MISSING':>11} {'-':>7}  no translation saved")
            problems += 1
            continue

        src = src_path.read_text(encoding="utf-8")
        tgt = tgt_path.read_text(encoding="utf-8")

        # A reply wrapped in ```markdown fences is common; strip them.
        tgt = re.sub(r"^\s*```(?:markdown)?\s*\n", "", tgt)
        tgt = re.sub(r"\n```\s*$", "", tgt)

        ratio = cjk_ratio(tgt)
        flags = []
        if ratio > args.cjk_threshold:
            flags.append("NOT TRANSLATED")
        if len(tgt) < len(src) * 0.4:
            flags.append("possibly truncated")

        src_tokens = set(IMG_TOKEN_RE.findall(src))
        tgt_tokens = set(IMG_TOKEN_RE.findall(tgt))
        if src_tokens - tgt_tokens:
            flags.append(f"lost {len(src_tokens - tgt_tokens)} image token(s)")
        if tgt_tokens - src_tokens:
            flags.append(f"invented {len(tgt_tokens - src_tokens)} image token(s)")

        status = "; ".join(flags) if flags else "ok"
        if flags:
            problems += 1
        print(f"{src_path.stem:>7} {len(src):>9,} {len(tgt):>11,} {ratio*100:>6.1f}%  {status}")
        parts.append(tgt.strip())

    merged = "\n\n".join(parts)

    # --- restore real image filenames ---------------------------------------
    map_path = kit / "image_map.json"
    if map_path.exists():
        token_to_path = json.loads(map_path.read_text(encoding="utf-8"))
        unknown: list[str] = []

        def _restore(match: re.Match) -> str:
            token = match.group(1)
            real = token_to_path.get(token)
            if real is None:
                unknown.append(token)
                return match.group(0)
            return real

        merged = re.sub(
            r"(?<=\()(IMG_\d{4})(?=\))", _restore, merged
        )
        print(f"\nrestored image paths from {len(token_to_path)} mapping(s)")
        if unknown:
            print(f"  WARNING: {len(unknown)} unknown token(s), left as-is: "
                  f"{sorted(set(unknown))[:5]}")
            problems += 1

        used = set(IMAGE_REF_RE.findall(merged))
        expected = set(token_to_path.values())
        never_used = expected - used
        if never_used:
            print(f"  WARNING: {len(never_used)} image(s) from the source never appear "
                  f"in the translation")
            problems += 1

        missing_files = [p for p in used if not (kit / p).exists() and not (kit / "images" / Path(p).name).exists()]
        if missing_files:
            print(f"  WARNING: {len(missing_files)} referenced file(s) not found on disk")
            problems += 1

    out_md = kit / "translated_book.md"
    out_md.write_text(merged, encoding="utf-8")
    overall = cjk_ratio(merged)
    print(f"\nwrote {out_md}  ({len(merged):,} chars, {overall*100:.2f}% CJK overall)")

    if problems:
        print(f"\n{problems} issue(s) above - fix those chunks and re-run before building a PDF.")
    else:
        print("\nno issues detected.")

    if args.pdf:
        print("\nrunning pandoc...")
        cmd = [
            "pandoc", str(out_md), "-o", str(kit / "translated_book.pdf"),
            "--pdf-engine=xelatex",
            "--toc",
            "-V", "documentclass=book",
            "-V", "geometry:margin=2.5cm",
            "-V", "CJKmainfont=Noto Sans CJK SC",
            "--resource-path", f"{kit}:{kit / 'images'}",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except FileNotFoundError:
            print("  pandoc not found. Install it from https://pandoc.org/installing.html")
            return 1
        except subprocess.TimeoutExpired:
            print("  pandoc timed out after 30 minutes")
            return 1
        if proc.returncode != 0:
            print("  pandoc failed:\n")
            print((proc.stderr or proc.stdout)[-3000:])
            return 1
        print(f"  wrote {kit / 'translated_book.pdf'}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
