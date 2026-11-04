"""
Chinese text inside figures: find it, have it redrawn in English, swap it in.

Why this module exists
----------------------
The translation replaces every word of the prose, but a figure is copied as
the scanned image, so its labels ("电源", "开关", axis titles, annotations)
reach the English book in Chinese. ``src.figures``' classifier now says, for
every kept image, whether Chinese text is printed inside it
(``FigureRecord.chinese_text``). This module turns those images into tasks
for the ``diagram-translator`` agent, compiles what it produces, and gives
assembly the English versions to use instead of the originals.

Three ways to translate a figure
--------------------------------
The agent chooses one per image and writes ``kit/figure_text/<stem>.json``:

.. code-block:: json

    {"image": "<filename>", "method": "overlay",
     "labels": [{"zh": "电源", "en": "power supply",
                 "box": [0.12, 0.30, 0.25, 0.36], "where": "upper left"}],
     "accepted": true, "notes": "one sentence"}

``tikz``
    The diagram is redrawn from scratch: the agent writes a complete
    standalone document, ``<stem>.tex``. Best for line drawings (circuits,
    force diagrams, plots, block diagrams), which come out sharper than
    the scan.
``overlay``
    The scan is kept; each Chinese label's ``box`` (fractions of the
    image's width and height, origin top left) is painted white and its
    English text set over it. For photographs and drawings too complex to
    redraw. This module writes the ``.tex``.
``legend``
    The fallback: the scan is kept unchanged, with an English legend under
    it that names each label by where it is (``"where": "upper left"``) --
    the book's pdflatex preamble cannot typeset Chinese, so the legend
    cannot quote the original. This module writes the ``.tex``.

Every method ends as ``<stem>.pdf``, rendered once more as
``<stem>.preview.png`` so the agent can look at its own result before it
sets ``"accepted": true``. The standalone page is ``width`` = the figure's
width in the book (``figures.json`` ``width`` x the text width), so a label
set in ``\\footnotesize`` is printed at that size, not scaled.

What assembly uses
------------------
`translated_images` maps an image filename to ``figure_text/<stem>.pdf``
only when the spec is accepted, the last render succeeded, and the spec has
not changed since that render (a hash of method, labels and the TikZ
source). Anything else falls back to the original scan, which is never
modified. The preamble's ``\\graphicspath{{images/}{./}}`` finds the PDF
relative to the kit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.figures import FIGURES_NAME, load_figures

logger = logging.getLogger(__name__)

FIGURE_TEXT_DIRNAME = "figure_text"
TASKS_NAME = "tasks.json"
METHODS: tuple[str, ...] = ("tikz", "overlay", "legend")
SIZES: tuple[str, ...] = ("tiny", "scriptsize", "footnotesize", "small", "normalsize")
DEFAULT_SIZE = "footnotesize"

# assets/preamble.tex: a4paper (21 cm) with 2.5 cm side margins.
TEXT_WIDTH_CM = 16.0
PREVIEW_DPI = 150
COMPILE_TIMEOUT_SECONDS = 120
# Context the agent gets around the figure, for the book's own terminology.
_CONTEXT_CHARS = 700

# Kit-level names, repeated here so this module does not import src.kit
# (which imports this one at assembly time).
_IMAGE_MAP_NAME = "image_map.json"
_TRANSLATED_DIRNAME = "translated"
_CHUNKS_DIRNAME = "chunks"

# Chinese characters and full-width punctuation: pdflatex cannot set them.
CJK_RE = re.compile(r"[　-〿㐀-䶿一-鿿＀-￯]")


def figure_dir(kit_dir: Path) -> Path:
    return kit_dir / FIGURE_TEXT_DIRNAME


def stem_of(image: str) -> str:
    return Path(image).stem


def spec_path(kit_dir: Path, stem: str) -> Path:
    return figure_dir(kit_dir) / f"{stem}.json"


def render_path(kit_dir: Path, stem: str) -> Path:
    return figure_dir(kit_dir) / f"{stem}.render.json"


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def image_file(kit_dir: Path, name: str) -> Path | None:
    for directory in (kit_dir / "images", kit_dir.parent / "images"):
        if (directory / name).exists():
            return directory / name
    return None


# ── Tasks ───────────────────────────────────────────────────────────────────


def _token_for(mapping: dict[str, str], name: str) -> str | None:
    return next((token for token, path in mapping.items() if Path(path).name == name), None)


def _context(directory: Path, suffix: str, token: str) -> tuple[str | None, str]:
    """``(chunk, text around the token)`` from the first file holding it."""
    if not directory.is_dir():
        return None, ""
    for path in sorted(directory.glob(f"*{suffix}")):
        text = path.read_text(encoding="utf-8")
        at = text.find(token)
        if at >= 0:
            start = max(0, at - _CONTEXT_CHARS)
            return path.stem, text[start : at + len(token) + _CONTEXT_CHARS]
    return None, ""


def status_of(kit_dir: Path, name: str) -> str:
    """``done`` (assembly will use it), ``rejected`` (the agent gave up or
    has not accepted it), ``stale`` (changed since the last render),
    ``failed`` (the last render did not compile) or ``todo``."""
    stem = stem_of(name)
    spec = _read_json(spec_path(kit_dir, stem))
    if spec is None:
        return "todo"
    render = _read_json(render_path(kit_dir, stem))
    if render is None:
        return "todo"
    if not render.get("ok"):
        return "failed"
    if render.get("hash") != spec_hash(kit_dir, spec):
        return "stale"
    if not spec.get("accepted"):
        return "rejected"
    pdf = figure_dir(kit_dir) / f"{stem}.pdf"
    return "done" if pdf.exists() else "todo"


def build_tasks(kit_dir: Path) -> dict:
    """One task per kept image with Chinese text inside; written to
    ``kit/figure_text/tasks.json`` and returned.

    ``not_asked`` lists kept images whose record never got the text answer
    (a manual or agent record, or a classifier that failed): they may hold
    Chinese text nobody looked for.
    """
    records = load_figures(kit_dir / FIGURES_NAME)
    mapping = _read_json(kit_dir / _IMAGE_MAP_NAME) or {}
    tasks: list[dict] = []
    not_asked: list[str] = []
    for name, record in records.items():
        if not record.keep:
            continue
        if record.chinese_text is None:
            not_asked.append(name)
            continue
        if not record.chinese_text:
            continue
        token = _token_for(mapping, name)
        chunk, translated = _context(kit_dir / _TRANSLATED_DIRNAME, ".tex", token) if token else (None, "")
        _, source = _context(kit_dir / _CHUNKS_DIRNAME, ".md", token) if token else (None, "")
        path = image_file(kit_dir, name)
        tasks.append(
            {
                "image": name,
                "stem": stem_of(name),
                "token": token,
                "chunk": chunk,
                "number": record.number or record.panel_of,
                "image_path": str(path) if path else None,
                "target_width_cm": round(record.width * TEXT_WIDTH_CM, 1),
                "status": status_of(kit_dir, name),
                "translated_context": translated,
                "source_context": source,
            }
        )
    payload = {
        "tasks": tasks,
        "todo": [t["stem"] for t in tasks if t["status"] != "done" and t["image_path"]],
        "missing_image": [t["image"] for t in tasks if not t["image_path"]],
        "not_asked": not_asked,
    }
    _write_json(figure_dir(kit_dir) / TASKS_NAME, payload)
    return payload


def _task(kit_dir: Path, stem: str) -> dict | None:
    tasks = (_read_json(figure_dir(kit_dir) / TASKS_NAME) or {}).get("tasks", [])
    return next((t for t in tasks if t.get("stem") == stem), None)


# ── The agent's spec ────────────────────────────────────────────────────────


def validate_spec(spec: dict) -> list[str]:
    """Everything wrong with a spec; empty when it can be rendered."""
    problems: list[str] = []
    method = spec.get("method")
    if method not in METHODS:
        return [f"method {method!r} is not one of {', '.join(METHODS)}"]
    labels = spec.get("labels")
    if not isinstance(labels, list) or not labels:
        return ["'labels' must be a non-empty list (one entry per Chinese label)"]
    for index, label in enumerate(labels, start=1):
        if not isinstance(label, dict):
            problems.append(f"label {index}: not an object")
            continue
        english = label.get("en")
        if not isinstance(english, str) or not english.strip():
            problems.append(f"label {index}: no 'en' text")
        elif CJK_RE.search(english):
            problems.append(f"label {index}: 'en' still contains Chinese: {english!r}")
        size = label.get("size", DEFAULT_SIZE)
        if size not in SIZES:
            problems.append(f"label {index}: size {size!r} is not one of {', '.join(SIZES)}")
        if method == "overlay":
            box = label.get("box")
            if not (
                isinstance(box, list)
                and len(box) == 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box)
                and 0 <= box[0] < box[2] <= 1
                and 0 <= box[1] < box[3] <= 1
            ):
                problems.append(
                    f"label {index}: 'box' must be [x0, y0, x1, y1], fractions 0-1 of the "
                    "image (origin top left), with x0 < x1 and y0 < y1"
                )
        if method == "legend":
            where = label.get("where")
            if not isinstance(where, str) or not where.strip():
                problems.append(f"label {index}: a legend entry needs 'where' (e.g. \"upper left\")")
            elif CJK_RE.search(where):
                problems.append(f"label {index}: 'where' contains Chinese")
    return problems


def spec_hash(kit_dir: Path, spec: dict) -> str:
    """What the render depends on: method, labels and, for TikZ, its source.
    ``accepted`` and ``notes`` are left out: the agent sets them after
    looking at the render."""
    payload = json.dumps(
        {"method": spec.get("method"), "labels": spec.get("labels")}, sort_keys=True, ensure_ascii=False
    )
    if spec.get("method") == "tikz":
        tex = figure_dir(kit_dir) / f"{stem_of(str(spec.get('image', '')))}.tex"
        payload += tex.read_text(encoding="utf-8") if tex.exists() else ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# A bare %, & or # in a label ends the line, breaks the alignment or the
# macro: escape the ones not already escaped.
_SPECIAL_RE = re.compile(r"(?<!\\)([%&#])")


def latex_text(text: str) -> str:
    return _SPECIAL_RE.sub(r"\\\1", text.strip())


_HEADER = r"""\documentclass[border=1pt]{standalone}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{tikz}
\begin{document}
"""


def overlay_tex(spec: dict, image_ref: str, width_cm: float) -> str:
    """The scan with each label's box painted white and the English set on it."""
    lines = [
        _HEADER.rstrip("\n"),
        r"\begin{tikzpicture}",
        (
            rf"\node[anchor=south west, inner sep=0] (img) at (0,0) "
            rf"{{\includegraphics[width={width_cm:.2f}cm]{{{image_ref}}}}};"
        ),
        r"\begin{scope}[x={(img.south east)}, y={(img.north west)}]",
    ]
    for label in spec["labels"]:
        x0, y0, x1, y1 = (float(v) for v in label["box"])
        # Box y is measured from the top; TikZ's from the bottom.
        bottom, top = 1 - y1, 1 - y0
        size = label.get("size", DEFAULT_SIZE)
        lines.append(rf"\fill[white] ({x0:.4f},{bottom:.4f}) rectangle ({x1:.4f},{top:.4f});")
        lines.append(
            rf"\node[font=\{size}, fill=white, inner sep=0.5pt, align=center] "
            rf"at ({(x0 + x1) / 2:.4f},{(bottom + top) / 2:.4f}) {{{latex_text(label['en'])}}};"
        )
    lines += [r"\end{scope}", r"\end{tikzpicture}", r"\end{document}", ""]
    return "\n".join(lines)


def legend_tex(spec: dict, image_ref: str, width_cm: float) -> str:
    """The unchanged scan with an English legend under it."""
    entries = "; ".join(
        f"{latex_text(label['where'])}: {latex_text(label['en'])}" for label in spec["labels"]
    )
    return "\n".join(
        [
            _HEADER.rstrip("\n"),
            rf"\begin{{minipage}}{{{width_cm:.2f}cm}}\centering",
            rf"\includegraphics[width=\linewidth]{{{image_ref}}}\par\smallskip",
            rf"\raggedright\footnotesize\textit{{Labels}} --- {entries}.",
            r"\end{minipage}",
            r"\end{document}",
            "",
        ]
    )


# ── Rendering ───────────────────────────────────────────────────────────────


@dataclass
class RenderResult:
    stem: str
    ok: bool
    method: str | None = None
    errors: list[str] = field(default_factory=list)
    pdf: str | None = None
    preview: str | None = None
    original: str | None = None

    def to_json(self) -> dict:
        return {
            "stem": self.stem,
            "ok": self.ok,
            "method": self.method,
            "errors": self.errors,
            "pdf": self.pdf,
            "preview": self.preview,
            "original": self.original,
        }


def latex_errors(log_text: str) -> list[str]:
    """``! message`` lines from a LaTeX log, each with its ``l.NN`` line."""
    errors: list[str] = []
    lines = log_text.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("! "):
            continue
        where = next((nxt.strip() for nxt in lines[index + 1 : index + 12] if nxt.startswith("l.")), "")
        errors.append(f"{line[2:].strip()}{'  [' + where + ']' if where else ''}")
    return errors[:10]


def run_pdflatex(tex_path: Path) -> tuple[bool, list[str]]:
    """Compile a standalone page beside itself. ``(ok, errors)``."""
    engine = shutil.which("pdflatex")
    if engine is None:
        return False, ["pdflatex not found on PATH"]
    try:
        completed = subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            capture_output=True,
            timeout=COMPILE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, [f"pdflatex timed out after {COMPILE_TIMEOUT_SECONDS}s"]
    log = tex_path.with_suffix(".log")
    log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    ok = completed.returncode == 0 and tex_path.with_suffix(".pdf").exists()
    errors = latex_errors(log_text)
    if not ok and not errors:
        errors = [f"pdflatex exited with {completed.returncode}"]
    return ok, errors


def render_preview(pdf: Path, out: Path) -> None:
    import pymupdf

    with pymupdf.open(pdf) as doc:
        doc[0].get_pixmap(dpi=PREVIEW_DPI).save(out)


def render(kit_dir: Path, stem: str) -> RenderResult:
    """Write (overlay / legend) or take (tikz) ``<stem>.tex``, compile it,
    and render a preview. The outcome is saved to ``<stem>.render.json``."""
    directory = figure_dir(kit_dir)
    result = RenderResult(stem=stem, ok=False)
    spec = _read_json(spec_path(kit_dir, stem))
    if spec is None:
        result.errors = [f"{spec_path(kit_dir, stem)} is missing or not valid JSON"]
        return _save_render(kit_dir, result, None)
    result.method = spec.get("method")
    problems = validate_spec(spec)
    image = str(spec.get("image") or "")
    if stem_of(image) != stem:
        problems.append(f"'image' {image!r} does not match the file name {stem}.json")
    source = image_file(kit_dir, image) if image else None
    if source is None:
        problems.append(f"image {image!r} not found in the kit's images")
    if problems:
        result.errors = problems
        return _save_render(kit_dir, result, spec)
    result.original = str(source)

    tex = directory / f"{stem}.tex"
    if spec["method"] == "tikz":
        if not tex.exists():
            result.errors = [f"method tikz: write the standalone document to {tex} first"]
            return _save_render(kit_dir, result, spec)
        found = CJK_RE.findall(tex.read_text(encoding="utf-8"))
        if found:
            result.errors = [f"{tex.name} still contains Chinese ({''.join(found[:8])}); pdflatex cannot set it"]
            return _save_render(kit_dir, result, spec)
    else:
        task = _task(kit_dir, stem) or {}
        width = float(task.get("target_width_cm") or 0.6 * TEXT_WIDTH_CM)
        # Relative to figure_text/, where pdflatex runs: no absolute paths
        # (spaces, drive letters) in \includegraphics.
        reference = Path("..") / source.relative_to(kit_dir) if source.is_relative_to(kit_dir) else source
        build = overlay_tex if spec["method"] == "overlay" else legend_tex
        tex.write_text(build(spec, reference.as_posix(), width), encoding="utf-8", newline="\n")

    pdf = tex.with_suffix(".pdf")
    pdf.unlink(missing_ok=True)
    ok, errors = run_pdflatex(tex)
    result.ok, result.errors = ok, errors
    if ok:
        preview = directory / f"{stem}.preview.png"
        render_preview(pdf, preview)
        result.pdf, result.preview = str(pdf), str(preview)
    return _save_render(kit_dir, result, spec)


def _save_render(kit_dir: Path, result: RenderResult, spec: dict | None) -> RenderResult:
    data = result.to_json()
    data["hash"] = spec_hash(kit_dir, spec) if spec is not None else None
    data["at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_json(render_path(kit_dir, result.stem), data)
    return result


# ── Assembly and the report ─────────────────────────────────────────────────


def translated_images(kit_dir: Path) -> dict[str, str]:
    """``{image filename: "figure_text/<stem>.pdf"}`` for every finished one."""
    directory = figure_dir(kit_dir)
    if not directory.is_dir():
        return {}
    found: dict[str, str] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == TASKS_NAME or path.name.endswith(".render.json"):
            continue
        spec = _read_json(path)
        image = str((spec or {}).get("image") or "")
        if image and stem_of(image) == path.stem and status_of(kit_dir, image) == "done":
            found[image] = f"{FIGURE_TEXT_DIRNAME}/{path.stem}.pdf"
    return found


def substitute(mapping: dict[str, str], kit_dir: Path) -> tuple[dict[str, str], int]:
    """The image map with each finished figure pointing at its English PDF."""
    english = translated_images(kit_dir)
    if not english:
        return mapping, 0
    swapped = {
        token: english.get(Path(path).name, path) for token, path in mapping.items()
    }
    count = sum(1 for token in mapping if swapped[token] != mapping[token])
    return swapped, count


def figure_text_report(kit_dir: Path) -> tuple[dict, str]:
    """``(summary, text)`` for the run log, from tasks.json and the files."""
    payload = build_tasks(kit_dir)
    by_status: dict[str, list[str]] = {}
    methods: dict[str, int] = {}
    entries: list[dict] = []
    for task in payload["tasks"]:
        by_status.setdefault(task["status"], []).append(task["image"])
        spec = _read_json(spec_path(kit_dir, task["stem"])) or {}
        render_data = _read_json(render_path(kit_dir, task["stem"])) or {}
        method = spec.get("method")
        if task["status"] == "done" and method:
            methods[method] = methods.get(method, 0) + 1
        entries.append(
            {
                "image": task["image"],
                "number": task["number"],
                "chunk": task["chunk"],
                "status": task["status"],
                "method": method,
                "labels": len(spec.get("labels") or []),
                "notes": str(spec.get("notes") or "")[:200],
                "errors": render_data.get("errors", [])[:3] if not render_data.get("ok") else [],
            }
        )
    summary = {
        "images": len(payload["tasks"]),
        "done": len(by_status.get("done", [])),
        "methods": methods,
        "by_status": {k: len(v) for k, v in sorted(by_status.items())},
        "not_done": [e for e in entries if e["status"] != "done"],
        "not_asked": payload["not_asked"],
        "missing_image": payload["missing_image"],
        "figures": entries,
    }
    lines = [
        (
            f"Figure text: {summary['done']} of {summary['images']} image(s) with Chinese text "
            f"translated ({', '.join(f'{k} {v}' for k, v in sorted(methods.items())) or 'none'})."
        ),
    ]
    for entry in entries:
        figure = f"figure {entry['number']}" if entry["number"] else "uncaptioned"
        lines.append(
            f"  {entry['status']:<8} {entry['method'] or '-':<7} {entry['image']} ({figure}, "
            f"chunk {entry['chunk'] or '?'}, {entry['labels']} label(s)) {entry['notes']}".rstrip()
        )
        for error in entry["errors"]:
            lines.append(f"           error: {error}")
    if payload["not_asked"]:
        lines.append(f"Never asked whether they hold Chinese text: {', '.join(payload['not_asked'])}")
    if payload["missing_image"]:
        lines.append(f"Image file missing: {', '.join(payload['missing_image'])}")
    return summary, "\n".join(lines) + "\n"


__all__ = [
    "CJK_RE",
    "FIGURE_TEXT_DIRNAME",
    "METHODS",
    "TASKS_NAME",
    "TEXT_WIDTH_CM",
    "RenderResult",
    "build_tasks",
    "figure_text_report",
    "latex_errors",
    "legend_tex",
    "overlay_tex",
    "render",
    "run_pdflatex",
    "spec_hash",
    "status_of",
    "substitute",
    "translated_images",
    "validate_spec",
]
