# textbook-translator

Scanned Chinese textbooks -> typeset English PDFs. The repo is the tool; each
book is data in `data/work/<book>/`, deliverables in `data/output/<book>/`.

- `src/` pipeline modules (parse via MinerU, prepare, lint + fidelity, build)
- `scripts/` the commands (`parse_book.py`, `prepare_kit.py`,
  `lint_translation.py`, `build_pdf.py`, `publish.py`, ...)
- `assets/` hand-maintained preamble and translation prompt
- `tests/` run with `uv run python -m pytest tests/ -q` -- no real API or
  MinerU calls, ever
- `docs/DECISIONS.md` why things are the way they are

**To translate a book, use `/translate-book`;** see
`.claude/skills/translate-book/SKILL.md`. Otherwise this is an ordinary
development session: do not start translating, parsing or calling MinerU
unless asked.

**MinerU quota: 1000 pages/day.** Never run `scripts/parse_book.py` (and
never pass it `--force`) without a reason; re-running resumes an in-flight
batch instead of re-uploading.
