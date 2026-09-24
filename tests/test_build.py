"""Tests for src.build (assembled LaTeX -> PDF).

No TeX distribution is needed: the engine is stubbed out and what is asserted
is the argv and the failure handling. Every failure mode here has to come
back as a value rather than an exception -- the caller prints it and moves
on.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.build import DEFAULT_ENGINE, build_pdf, log_excerpt, parse_log_errors


class FakeRun:
    """Records each engine invocation and returns a scripted exit code."""

    def __init__(self, returncode: int = 0, *, make_pdf: bool = True) -> None:
        self.returncode = returncode
        self.make_pdf = make_pdf
        self.calls: list[tuple[list[str], Path]] = []

    def __call__(self, command, *, cwd, **_kwargs):  # noqa: ANN001 - subprocess shim
        self.calls.append((list(command), Path(cwd)))
        if self.make_pdf and self.returncode == 0:
            (Path(cwd) / Path(command[-1]).with_suffix(".pdf").name).write_bytes(b"%PDF-1.5")
        return subprocess.CompletedProcess(command, self.returncode, "", "")


def _book(tmp_path: Path) -> Path:
    tex_path = tmp_path / "translated_book.tex"
    tex_path.write_text(
        "\\documentclass{book}\n\\begin{document}\nx\n\\end{document}\n", encoding="utf-8"
    )
    return tex_path


def _patch(monkeypatch: pytest.MonkeyPatch, runner: FakeRun) -> None:
    monkeypatch.setattr("src.build.shutil.which", lambda engine: f"/usr/bin/{engine}")
    monkeypatch.setattr("src.build.subprocess.run", runner)


def test_the_default_engine_is_pdflatex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The preamble uses inputenc, and a finished book has no CJK left."""
    runner = FakeRun()
    _patch(monkeypatch, runner)

    build_pdf(_book(tmp_path))

    assert DEFAULT_ENGINE == "pdflatex"
    assert all(command[0] == "pdflatex" for command, _ in runner.calls)


def test_it_runs_twice_so_the_table_of_contents_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One pass writes the .toc; the second typesets it. A single pass gives
    a book with an empty contents page and no error."""
    runner = FakeRun()
    _patch(monkeypatch, runner)

    succeeded, message = build_pdf(_book(tmp_path))

    assert succeeded, message
    assert len(runner.calls) == 2


def test_it_never_waits_for_keyboard_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A macro the model invented would otherwise hang the build forever."""
    runner = FakeRun()
    _patch(monkeypatch, runner)

    build_pdf(_book(tmp_path))

    assert "-interaction=nonstopmode" in runner.calls[0][0]


def test_it_compiles_in_the_directory_holding_the_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """\\graphicspath{{images/}} is relative to the .tex file."""
    runner = FakeRun()
    _patch(monkeypatch, runner)
    tex_path = _book(tmp_path)

    build_pdf(tex_path)

    assert runner.calls[0][1] == tex_path.parent


def test_a_different_engine_can_be_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """assets/preamble.tex documents a two-line swap to XeLaTeX for a book
    that does end up with residual CJK."""
    runner = FakeRun()
    _patch(monkeypatch, runner)

    build_pdf(_book(tmp_path), engine="xelatex")

    assert runner.calls[0][0][0] == "xelatex"


def test_a_missing_source_file_is_reported_not_raised(tmp_path: Path) -> None:
    succeeded, message = build_pdf(tmp_path / "nope.tex")

    assert succeeded is False
    assert "not found" in message


def test_a_missing_engine_is_an_actionable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.build.shutil.which", lambda _engine: None)

    succeeded, message = build_pdf(_book(tmp_path))

    assert succeeded is False
    assert "not on PATH" in message
    assert "TeX Live" in message


def test_a_compile_failure_returns_the_error_lines_from_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log is tens of thousands of lines; about six of them matter."""
    tex_path = _book(tmp_path)
    tex_path.with_suffix(".log").write_text(
        "noise\n" * 5000
        + "! Undefined control sequence.\nl.4812 \\frobnicate\n"
        + "more noise\n" * 500,
        encoding="utf-8",
    )
    _patch(monkeypatch, FakeRun(returncode=1))

    succeeded, message = build_pdf(tex_path)

    assert succeeded is False
    assert "! Undefined control sequence." in message
    assert "l.4812 \\frobnicate" in message
    assert "noise" not in message


def test_a_timeout_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("pdflatex", 1800)

    monkeypatch.setattr("src.build.shutil.which", lambda engine: f"/usr/bin/{engine}")
    monkeypatch.setattr("src.build.subprocess.run", timeout)

    succeeded, message = build_pdf(_book(tmp_path))

    assert succeeded is False
    assert "timed out" in message


def test_a_silent_failure_to_produce_a_pdf_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, FakeRun(make_pdf=False))

    succeeded, message = build_pdf(_book(tmp_path))

    assert succeeded is False
    assert "no PDF was produced" in message


ASSEMBLED = (
    "\\documentclass{book}\n"  # 1
    "\\begin{document}\n"  # 2
    "% >>> translated/002.tex:3\n"  # 3 -- the fragment's body starts on its line 3
    "\\section{One}\n"  # 4 -> translated/002.tex:3
    "Text.\n"  # 5 -> translated/002.tex:4
    "\\sectionsection{Two}\n"  # 6 -> translated/002.tex:5
    "\\end{document}\n"
)

TWO_ERROR_LOG = (
    "! Undefined control sequence.\n"
    "<recently read> \\sectionsection\n"
    "\n"
    "l.6 \\sectionsection\n"
    "                   {Two}\n"
    "! Undefined control sequence.\n"  # the same error reported twice
    "l.6 \\sectionsection\n"
    "! LaTeX Error: Unicode character ① (U+2460)\n"
    "l.1 \\documentclass{book}①\n"
    "!  ==> Fatal error occurred, no output PDF file produced!\n"
)


def test_log_errors_are_mapped_back_to_the_fragment_they_came_from(tmp_path: Path) -> None:
    tex_path = tmp_path / "translated_book.tex"
    tex_path.write_text(ASSEMBLED, encoding="utf-8")

    errors = parse_log_errors(TWO_ERROR_LOG, tex_path)

    assert [(e.location, e.message) for e in errors] == [
        ("translated/002.tex:5", "Undefined control sequence."),
        ("translated_book.tex:1", "LaTeX Error: Unicode character ① (U+2460)"),
    ]
    assert errors[0].context == "\\sectionsection"


def test_a_failed_build_lists_every_error_from_one_more_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """-halt-on-error stops at the first error: four defects cost four builds.
    A failure triggers one run without it, and every error is reported."""
    tex_path = tmp_path / "translated_book.tex"
    tex_path.write_text(ASSEMBLED, encoding="utf-8")
    runner = FakeRun(returncode=1)

    def run(command, *, cwd, **kwargs):  # noqa: ANN001 - subprocess shim
        for arg in command:
            if arg.startswith("-output-directory="):
                out_dir = Path(arg.split("=", 1)[1])
                (out_dir / "translated_book.log").write_text(TWO_ERROR_LOG, encoding="utf-8")
        return runner(command, cwd=cwd, **kwargs)

    monkeypatch.setattr("src.build.shutil.which", lambda engine: f"/usr/bin/{engine}")
    monkeypatch.setattr("src.build.subprocess.run", run)

    succeeded, message = build_pdf(tex_path)

    assert succeeded is False
    assert "translated/002.tex:5: Undefined control sequence." in message
    assert "translated_book.tex:1: LaTeX Error: Unicode character" in message
    assert "(2)" in message
    probe = runner.calls[-1][0]
    assert "-halt-on-error" not in probe
    assert "-interaction=nonstopmode" in probe
    assert runner.calls[-1][1] == tmp_path  # still finds images/ relative to the .tex


def test_log_excerpt_falls_back_to_the_tail_when_nothing_matches(tmp_path: Path) -> None:
    log_path = tmp_path / "book.log"
    log_path.write_text("\n".join(f"line {i}" for i in range(200)), encoding="utf-8")

    assert "line 199" in log_excerpt(log_path)


def test_log_excerpt_survives_a_missing_log(tmp_path: Path) -> None:
    assert "no readable log" in log_excerpt(tmp_path / "gone.log")
