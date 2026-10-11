# translation_project

Translates scanned textbooks (Chinese → English) into a typeset PDF, with
Claude as the only LLM.

The pipeline is **page range → parse → prepare → translate → lint → build →
publish**. Parsing is an HTTP call to the MinerU cloud API (OCR, not an LLM);
everything else is local CPU work plus Claude. There is no GPU step and no
notebook.

```
PDF ─page range─► source_body.pdf ─parse─► merged.md ─prepare─► kit/chunks/*.md
                                                                      │
                                                         (Claude translates, to LaTeX)
                                                                      │
                                                                      ▼
data/output/<book>/ ◄─publish─ PDF ◄─build─ translated_book.tex ◄─lint─ kit/translated/*.tex
```

The source chunks are Markdown, because that is what the parser produces. The
replies are **LaTeX body fragments**, which the build wraps in
[assets/preamble.tex](assets/preamble.tex) and compiles directly.

---

## Agent mode: one command, PDF in, PDF out

A whole book runs **unattended**: no step waits for a person. The
orchestrator — Claude Opus 5.5 at high effort — makes every decision this
README used to leave to a human (the body's page range, the parser backend,
`<-- CHECK` rows in the outline, flagged figures, lint findings, build
errors), records each one with its evidence as `"by": "agent"`, and writes
all of it to `run_log.md`: decisions, assumptions, uncertainties, known
limitations and a spot-check list for a human.

- The orchestrator's instructions are in
  [.claude/skills/translate-book/SKILL.md](.claude/skills/translate-book/SKILL.md).
- The subagents are in [.claude/agents/](.claude/agents/).
- Every design choice the brief left open is in [docs/DECISIONS.md](docs/DECISIONS.md).

**Launch it** in either mode:

| Mode | Needs | Command |
|---|---|---|
| No-API (Claude Code subscription) | Claude Code logged in; no API key | `scripts/translate_unattended.sh data/input/physics.pdf [book_name]` |
| API (Claude Agent SDK) | `ANTHROPIC_API_KEY` | `uv sync --group agent` then `uv run --group agent python scripts/run_agent_sdk.py data/input/physics.pdf [book_name]` |
| Interactive | Claude Code | `/model` → Opus 5.5 at high effort, then `/translate-book data/input/physics.pdf` |

`book_name` defaults to the PDF's stem, lowercased, spaces as `_`.

**Output lands in `data/output/<book>/`**: `<book>_translated.pdf`,
`<book>_translated.tex` (small, and enough to fix a typo and rebuild) and
`run_log.md`. A failed build publishes the run log alone, never a PDF. The
headless session transcript goes to `logs/`.

**Who does what:**

| Role | Model | Effort | Runs as |
|---|---|---|---|
| Orchestrator (whole run) | `claude-opus-5-5` | `high` | main session |
| Page-range finder (before parsing) | `claude-sonnet-5-5` | `medium` | subagent |
| Stage 2 runner (prepare kit) | `claude-sonnet-5-5` | `medium` | subagent |
| Book profile (glossary + heading levels) | `claude-sonnet-5-5` | `medium` | LLM call |
| Structure labelling + follow-ups | `claude-sonnet-5-5` | `medium` | LLM call |
| Figure classification (vision) | `claude-haiku-5-5` | `low` | LLM call |
| Chunk translation | `claude-haiku-5-5` | `medium` | LLM call |
| Digesting long output (lint, build logs, parse logs) | `claude-sonnet-5-5` | `medium` | subagent |

An "LLM call" is made by Python through the Messages API in API mode, and by
a subagent through the exchange (below) in no-API mode. Translation in no-API
mode is one `chunk-translator` subagent per chunk; in API mode, one Message
Batch (`scripts/translate_kit.py`).

**What the run is not allowed to do:** edit `src/`, `assets/`, `tests/`,
`scripts/` or `.claude/` (the launchers' settings overlay,
[.claude/run-settings.json](.claude/run-settings.json), denies it), or spend
MinerU quota twice. Per-book adjustments live in the work dir.

The rest of this README is the manual reference: what each stage does, what
it decides, and how to run it by hand.

---

## Setup

**1. Python dependencies** (Python 3.12+):

```bash
uv sync
```

For API mode's launcher, also `uv sync --group agent` (the Claude Agent SDK
is an optional group, so no-API mode does not need it).

**2. A LaTeX distribution** — TeX Live or MiKTeX, providing `pdflatex`. The
one dependency not installed through Python.

```bash
pdflatex --version
```

**3. `.env`** in the project root:

```bash
MINERU_TOKEN=your-mineru-token
# API mode only:
ANTHROPIC_API_KEY=your-anthropic-key
```

Get a MinerU token at <https://mineru.net/apiManage/token>. **No-API mode
needs only `MINERU_TOKEN`**: Claude is your Claude Code subscription. With
no `ANTHROPIC_API_KEY`, the pipeline's own LLM calls go through the exchange
automatically.

---

## Data layout: one repo, many books

**The repo is the tool; books are the data. Never copy the repo per book.**

```
data/work/<book_name>/
├── page_range.json     the body's physical pages, decided before parsing (hand-editable)
├── page_images/        thumbnails rendered for that decision (p0001.png = physical page 1)
├── source_body.pdf     pages start..end, the only part that is parsed
├── parse_batch.json    the MinerU batch in flight, for resuming
├── parts/              part PDFs sent to the API (+ manifest.json with the page offset)
├── images/             every figure, flat, named by content hash
├── merged.md           raw parser output          ← stage 1
├── content_list.json   MinerU's layout: every block's bbox and page
├── normalized.md       after split-heading repair ← stage 2
├── prepared.md         after re-levelling, structure, tables + figures
├── book_profile.json   cached LLM call: glossary + heading levels (hand-editable)
├── structure.json      part / chapter / reading per top-level heading (hand-editable)
├── llm/                the exchange: requests/, responses/, images/, pending.json
├── run_state.json      where an unattended run is, so it can resume
├── run_log.md          the orchestrator's log of every decision
├── logs/               full reports behind the --json summaries
└── kit/
    ├── chunks/001.md…       source chunks, image paths tokenised
    ├── translated/001.tex…  the translations, each ending with % END-OF-CHUNK
    ├── images/
    ├── image_map.json       IMG_nnnn → real filename
    ├── figures.json         number / width / keep per image (hand-editable)
    ├── figures_review.html  dropped, flagged and default-width figures, as thumbnails
    ├── known_commands.txt   this book's extra LaTeX commands, each with a reason
    ├── lint_waivers.json    findings judged false positives, each with a reason
    ├── preamble.tex         only if the book starts at another chapter than 1
    ├── system_prompt.txt
    ├── build_result.json    the last build attempt's outcome
    ├── translated_book.tex  ← stage 4 assembles this
    └── translated_book.pdf

data/output/<book_name>/
├── <book_name>_translated.pdf
├── <book_name>_translated.tex
└── run_log.md
```

`translated_book.tex` has to sit beside `images/`: the preamble's
`\graphicspath` resolves figures relative to the `.tex`.

`data/work/` and `data/output/` are gitignored.

**Provenance.** Every hand-editable decision file records who made each
decision in `"by"`. `"manual"` (a person) and `"agent"` (the orchestrator)
are never overwritten by a re-run or a `--force-*` flag; a person's
`"manual"` outranks the agent's.

---

## Stage 1a — Page range

Front matter (cover, title and copyright pages, preface, table of contents)
and non-content back matter are where parsing goes wrong: a TOC parses into
dozens of fake headings, an index into pages of numbers. They are cut
**before** parsing, so parsing starts at the first chapter and no quota is
spent on them. The PDF's table of contents is LaTeX's own, generated from
the chapters.

```bash
uv run python scripts/render_pages.py data/input/physics.pdf data/work/physics/page_images --pages 1-40,-40 --dpi 60
```

renders thumbnails named by physical page (`p0001.png`), plus `pages.json`
with the page count and which pages have a text layer. The orchestrator's
`page-range-finder` looks at them and writes `data/work/<book>/page_range.json`:

```json
{
  "start": 15, "end": 402,
  "first_chapter_number": 12,
  "starts_with_part_title": true,
  "excluded": [{"pages": "1-14", "kind": "toc", "note": "..."},
               {"pages": "403-420", "kind": "index", "note": "..."}],
  "confidence": "high",
  "evidence": "p15 has the 第三篇 电磁学 divider, p16 starts 第12章 静电场; p403 starts 索引",
  "by": "agent"
}
```

The policy:

- **Start** on the first page of the first chapter — or on a part-title page
  (`第N篇`) directly before it, which becomes `\part`. Preface, foreword,
  how-to-use pages and the TOC are always cut.
- **Keep** content-bearing back matter: appendices (constants, formulas, unit
  tables) and answers to exercises.
- **Cut** the index (its page numbers would be wrong anyway), bilingual term
  lists, references not cited inline by number, colophon, ads and trailing
  blank pages.
- **Never cut mid-book.** The range is one contiguous block.

`first_chapter_number` is checked in stage 2 against the preamble's
`\setcounter{chapter}`.

## Stage 1b — Parse

```bash
uv run python scripts/parse_book.py data/input/physics.pdf data/work/physics
```

With `page_range.json` present (or `--page-range <file>`), the PDF is first
cut to `source_body.pdf`; the page offset is recorded in
`parts/manifest.json`, so a body page maps back to the original
(`src.page_range.original_page`). Then the body is split to fit the API's
hard limits (200 pages / 200 MB per file), every part is uploaded in one
batch, polled with per-file page progress, and merged into `merged.md` plus
a flat `images/`. Each part's layout data is merged into
`content_list.json`; it is what sizes the figures in stage 2.

| Flag | Purpose |
|---|---|
| `--page-range FILE` | Parse only these pages (default: `page_range.json` if present) |
| `--model {vlm,pipeline}` | Parser backend (default `vlm`) |
| `--language` | OCR language hint (default `ch`) |
| `--wait-minutes N` | Poll at most this long, then exit 4 if still running |
| `--force` | Re-parse even if `merged.md` is current; upload again even with a batch in flight |
| `--json` | Print a summary: state, trimmed, page offset, first-heading warning, failed parts |

**Resumable.** The batch id is saved in `parse_batch.json` as soon as the
upload finishes. Re-running while the batch is in flight resumes polling it —
no re-upload, no quota spent twice. Exit code 4 means "still running; run the
same command again". That is how an unattended run polls a two-hour parse
from shell calls capped at ten minutes.

**Watch the page count.** The quota is 1000 pages/day at top priority. A book
whose `merged.md` is newer than the PDF is never parsed again, even if a
page range appears later; the summary says it was parsed untrimmed.

**The first heading is cross-checked**: `merged.md` should begin with a
chapter, or a part followed by a chapter. If it starts with `目录` or anything
that is not a chapter, the start page may be wrong, and a warning says so.

**`vlm` vs `pipeline` is a real per-book decision.** `vlm` is dramatically
cleaner on formula-dense material — `$10^{-20}$` where `pipeline` gives
`$1 0 ^ { - 2 0 }$` — and it preserves merged table cells. But VLM backends can
hallucinate, which for a physics textbook means a plausible equation with a
wrong exponent that **no automated check will catch** (see fidelity lint,
below). The orchestrator parses ~10 formula-dense pages each way and compares
before committing to `vlm`, and logs the evidence.

If a part fails server-side, the others still merge and the script exits
non-zero naming the failure — a partial book is never silently mistaken for a
complete one.

## Stage 2 — Prepare the translation kit

```bash
uv run python scripts/prepare_kit.py data/work/physics
```

Eight steps, in this order:

1. **normalize** — merge split headings; report anything else short
2. **profile** — one cached LLM call: glossary + a level per heading
3. **re-level** — apply that classification (pure, no LLM)
4. **structure** — decide which top-level headings are parts, chapters and
   supplementary readings, and write them as final `\part` / `\chapter` /
   `\chapter*` (see below)
5. **tables** — HTML → markdown pipe tables
6. **figures** — `kit/figures.json`: each image's source number and width;
   uncaptioned images are classified, and a confident decoration is removed
   before the translator sees it (see below)
7. **tokenize** — image paths → `IMG_nnnn`
8. **chunk** — ~30,000 chars, on paragraph boundaries only

| Flag | Purpose |
|---|---|
| `--no-profile` | Skip the LLM calls; headings keep their parsed levels, and structure uses the numbering rules only |
| `--force-profile` | Ignore the cached profile (a `"by": "manual"`/`"agent"` profile is kept) |
| `--force-structure` | Ask Claude again about headings it already decided |
| `--no-classify` | No vision calls: keep every uncaptioned image and list it for review |
| `--classify-dry-run` | Classify and fill in the review sheet, but drop nothing |
| `--reclassify` | Ignore the labels cached in `figures.json` and ask the vision model again |
| `--chunk-chars N` | Override the chunk budget |
| `--fix-math-spacing` | Trim whitespace inside inline `$` |
| `--json` | Print a digest instead of the report; the report goes to `logs/prepare_kit.txt` |
| `-v` | Also list every short heading left unchanged |

**What used to want a human's eyes, the orchestrator now decides** — and
records each decision as `"by": "agent"`:

- **Merged headings** are listed individually. On the reference book this is 5
  chapter merges (`第13章` + `电势` → `第13章 电势`) and 3 part-title merges.
- **`part-ordinal-missing`** means a part title was recovered but its number
  was not in the parsed text at all (`第篇 电磁学`). The structure step
  recognises `第篇` as a part and LaTeX numbers parts itself.
- **Tables left as HTML** are reported with a reason. Merged cells
  (`colspan`/`rowspan`) cannot be expressed as a pipe table without
  duplicating content; on the reference book, 19 of 24 convert and the 5
  skipped are legitimate merged header cells.

`book_profile.json` is plain JSON. A heading at the wrong level is fixed by
editing that line (and setting `"by"`) and re-running — the cached profile is
reused, so no LLM call is repeated.

**Structure.** The parser drops `第N章` from most chapter titles, so a chapter
(`静电场`) and a supplementary reading (`大气电学`) look alike by title. What
tells them apart is the numbering under them compared with their siblings: on
the reference book, the chapters' sections and captions carry the chapter
number (`12.1`, `图12.3`) and the readings' are lettered (`G.1`, `图G.3`).

- Code collects the evidence per top-level heading — sub-heading, caption and
  equation-tag numbers — without assuming any caption word or title format.
- **One Claude call labels the whole outline** (part / chapter / reading /
  front-back matter), so it can compare siblings.
- Code cross-checks the labels: chapter numbers must run on, a chapter's
  number must match the numbering under it, a part must have chapters after
  it, and this corpus's numbering conventions give a second opinion. A
  conflicting row gets one follow-up call, kept only if it resolves the
  conflict without creating another.
- Whatever still conflicts is marked `<-- CHECK`. **The orchestrator decides
  each one** from the evidence, sets its `"kind"` (and `"number"`) in
  `structure.json` with `"by": "agent"`, and logs why.

The step also turns numbered items under a chapter summary (提要) back into
paragraphs, stars the sections of a reading, and warns if the book's first
chapter — from the outline, and from `page_range.json`'s
`first_chapter_number` — does not match `\setcounter{chapter}` in the book's
preamble. The orchestrator then creates `kit/preamble.tex` with the right
counter.

**Figures.** Each image's number, width and whether it is a figure at all are
decided here, outside the translating model, and stored per image filename in
`kit/figures.json`. The model only copies `IMG_nnnn` tokens; assembly applies
the records at build time, so a wrong width, number or drop is fixed by
editing one line and rebuilding — no re-translation.

- **Number** — from MinerU's `image_caption`, else the caption line under the
  image (`图 18.3`, `图18.3`, `图 G.2`). The PDF prints exactly that number;
  LaTeX's own figure counter is never shown. The panels of a figure — `(a)` /
  `(b)` images under one caption — share its number.
- **Width** — the image's bbox width over the median width of the book's long
  body-text blocks, clamped to 0.20–0.90 of `\textwidth`, and every figure is
  capped at 0.45 `\textheight`. No bbox → 0.6, listed for review.
- **Keep** — only uncaptioned images are candidates. Each one goes to the
  classifier (Claude Haiku 5.5) with the text around it, and is dropped only
  if it comes back as a `decoration` with confidence ≥ 0.9 **and** it does not
  sit in a gap in the caption numbering. Everything else — a low confidence,
  another label, a malformed reply, an API failure — keeps the image and
  flags it. A kept decoration costs one wasted figure; a dropped figure is
  missing content.

The orchestrator runs stage 2 first with `--classify-dry-run`, looks at
`kit/figures_review.html` (dropped, flagged and default-width images, as
thumbnails), overrules what needs it in `figures.json` with `"by": "agent"`,
then runs it for real — labels are cached, so no new calls.

**A book prepared before the figures step** needs no re-translation:
`scripts/prepare_figures.py <work-dir>` builds `figures.json` from the
existing chunks; `scripts/link_figures.py` borrows layout data from a re-parse
into a separate work dir.

### The exchange (no-API mode)

Without an API key, Python cannot call Claude. Every LLM call inside the
pipeline (profile, structure, classify) goes through
[src/llm/exchange.py](src/llm/exchange.py) instead:

```
data/work/<book>/llm/
├── requests/<id>.json     written by Python: role, model, effort, system, user, images
├── responses/<id>.txt     written by an agent: the model's raw reply, nothing else
├── images/                copies of the images a classify request shows
└── pending.json           unanswered request ids, grouped by role
```

A request is exactly what the API would have been sent; an answer is used
exactly as an API reply would be. With a missing answer, the request is
written and recorded, the run carries on as far as it sensibly can (all
figure requests are emitted in one pass), and the script exits with **code
3**. Answer the requests, re-run: answered calls are free, so each run is one
round. `prepare_kit.py` stops after the first stage with pending requests,
because each stage's prompts depend on the decisions before it. A book
typically takes four rounds.

The `stage2-preparer` subagent drives this loop, spawning
`llm-answerer-sonnet` and `figure-classifier` agents. **By hand**, if ever
needed: read each `llm/requests/<id>.json`, treat its `system` field as the
instructions and `user` (plus `images`) as the input, write the reply alone
to `llm/responses/<id>.txt`, re-run. To re-ask, delete the response file.

## Translate

`kit/system_prompt.txt` is
[assets/system_prompt_latex.txt](assets/system_prompt_latex.txt) with the
book's context and glossary substituted in. Each chunk `kit/chunks/NNN.md` is
translated with it into `kit/translated/NNN.tex`:

- **No-API mode:** the orchestrator spawns one `chunk-translator` agent per
  chunk (at most 10 at a time).
- **API mode:** `uv run python scripts/translate_kit.py <kit> --json` sends
  every untranslated chunk as one Message Batch, with the system prompt
  cached across the batch. Resumable like the parse (exit 4 while running).
- **By hand, as a fallback:** paste `system_prompt.txt` and one chunk into any
  Claude conversation and save the reply as `translated/NNN.tex` — the kit
  format is unchanged.

Every reply ends with the line `% END-OF-CHUNK`. A reply without it was cut
off — that is the deterministic truncation check when the model's stop
reason is invisible. Assembly strips it.

Two constraints worth knowing:

- Chunk size is bound by the model's **output** limit, not its context
  window. Measured on the reference book, output ran 1.33×–2.87× the input
  length, and the longest reply reached ~36,000 tokens. The budget is 30,000
  source characters: worst case ≈ 22,000 tokens of LaTeX.
- **The model writes body fragments only.** No `\documentclass`, no
  `\usepackage`, no `\begin{document}` — the preamble is added once, at
  assembly.

## Stage 3 — Lint

```bash
uv run python scripts/lint_translation.py data/work/physics/kit
```

**This is the stage that ends the debugging loop.** LaTeX reports one error
per compile; every check here runs at once, located as `translated/NNN.tex:L`.

- **review** — someone must look (in an unattended run, the orchestrator).
  Exits non-zero, which gates the build.
- **auto-fix** — provably safe. Apply with `--fix`, which repairs the saved
  replies in `kit/translated/` in place.
- **info** — context, not a problem.

Per chunk pair: CJK residue, truncation (the sentinel, an absolute floor, and
an outlier against the book's own median expansion of *prose*), image-token
integrity, and **fidelity** (below).

On the translated LaTeX: unbalanced and crossed environments, undefined
environments, preamble leakage, characters pdflatex cannot typeset, display
delimiters that lost their backslash, `\tag` outside an equation, brace
balance, inline `$` parity per paragraph, duplicated numbering, Markdown
headings, dangling argument commands, doubled commands, unknown command
tokens, text-mode commands inside math, commands whose package the preamble
does not load, chapter numbers that drift from the book's own equation tags,
and figure paths that do not resolve. Against `figures.json`: figures with no
record, dropped images still referenced, and numbering gaps (info).

`--json` prints a summary (counts, every review and auto-fix finding with its
file and line) and writes the full report to `kit/logs/lint.txt`.

### Fidelity lint: does the translation say the same mathematics?

The checks above prove the LaTeX *compiles*. [src/fidelity.py](src/fidelity.py)
checks that it still says what the source says, deterministically and
without an LLM. Both sides are first reduced to a canonical token stream:
delimiters unified (`$$`, `\[ \]`, `equation`, `align` …), the parser's
character spacing removed, Unicode `−` `×` `·` mapped to LaTeX,
presentation-only commands dropped (`\mathrm`, `\,`, `\quad`, `\left`/`\right`
…), and translated prose (`\text{}` contents) ignored. Then:

| Check | Rule | Severity |
|---|---|---|
| `fidelity-tags` | equation tags (`\tag{12.3}`, or `(12.3)` closing a display) differ | review |
| `fidelity-display-count` | a display block was lost or added (one that became inline math, or vice versa, is info) | review |
| `fidelity-numbers` | a numeric literal in math — signed exponents included — is missing or invented | review (a value on both sides with a different count: info) |
| `fidelity-blocks` | a matched display block's token similarity is below 0.85, or its sign/relation operators differ | review |
| `fidelity-prose-numbers` | Arabic numerals in source prose absent from the translation's prose | info |

Calibrated on 89 real chunk pairs from five books so that formatting-only
differences stay quiet (details in [docs/DECISIONS.md](docs/DECISIONS.md)).

**What it cannot catch: a parse hallucination.** If the VLM parser reads an
exponent wrong off the page, the wrong value is in the source chunk too, both
sides agree, and every check passes. Fidelity lint catches *translation
drift only*. Every run log lists this as a known limitation.

### Per-book additions

- **`kit/known_commands.txt`** — commands that are real LaTeX this book uses,
  one per line with a `#` comment saying why; merged into the vocabulary at
  lint time. Unknown commands are still reported, never auto-fixed: on the
  reference book 40 of 41 flagged tokens were legitimate and one was the
  defect. Nobody edits `KNOWN_COMMANDS` in `src/lint.py` during a run.
- **`kit/lint_waivers.json`** — a list of `{check, file, line_excerpt,
  reason, by}`. A matching finding (of any check) is downgraded to info with
  its reason in the message, so it stays visible. A waiver without a reason
  is refused; one that matches nothing is reported.

Three behaviours are deliberate:

- **A doubled known command is auto-fixed** (`\mathrmmathrm` → `\mathrm`); the
  second spelling only for commands never repeated on purpose.
- **So is a duplicated number** (`\section{12.1 Electric Charge}` →
  `\section{Electric Charge}`).
- **So is a bare `[` on its own line** (→ `\[`), only when the brackets
  strictly alternate.

Lint one file with `--tex one_chunk.tex` or `--markdown merged.md`.

## Stage 4 — Build and publish

```bash
uv run python scripts/build_pdf.py data/work/physics/kit --publish
```

Lints the kit, concatenates the replies, rewrites every `\bookfig` /
`\bookfigtwo` from `figures.json`, restores the real image filenames, wraps
the result in the preamble, and runs `pdflatex` twice (the second pass fills
in the table of contents).

| Flag | Purpose |
|---|---|
| `--no-lint` | Build even when findings need review |
| `--fix` | Apply the safe repairs to the assembled `.tex` first |
| `--engine` | Default `pdflatex` |
| `--json` | Print a summary: lint counts, assembly problems, every LaTeX error located as `translated/NNN.tex:L` |
| `--publish` | Afterwards, copy the deliverables to `data/output/<book>/` |

When the build fails, it compiles once more without `-halt-on-error` and lists
**every** error in the book, each located in the reply it came from. Every
attempt is recorded in `kit/build_result.json`.

**Publishing** (`scripts/publish.py data/work/<book>`, or `--publish`) copies
— never moves — `translated_book.pdf`, `translated_book.tex` and the run log
to `data/output/<book>/`. If the last build failed, only the run log is
published, and a PDF from an earlier run is renamed `*.stale.pdf`.

The preamble is a verified configuration and the specification:
`openany` removes the blank verso before every chapter, `\bookfigx` /
`\bookfigtwox` size each figure from `figures.json` and cap its height, and
`\setcounter{chapter}{N}` makes a volume starting at chapter N+1 number its
sections correctly. A book that starts elsewhere gets its own copy as
`kit/preamble.tex` with only the counter changed. If a finished book still has
Chinese in it, the preamble carries a commented swap to `fontspec` + `xeCJK`;
uncomment it and pass `--engine xelatex`.

---

## Configuration

All settings are environment variables, read in [src/config.py](src/config.py).

| Variable | Default | Purpose |
|---|---|---|
| `MINERU_TOKEN` | — | MinerU cloud API token (required) |
| `ANTHROPIC_API_KEY` | — | API mode only |
| `LLM_BACKEND` | `anthropic` if a key is set, else `exchange` | Who answers the pipeline's LLM calls |
| `CLAUDE_MODEL_<ROLE>` | see the role table above | Model per role: `PROFILE`, `STRUCTURE`, `CLASSIFY`, `TRANSLATE` |
| `CLAUDE_EFFORT_<ROLE>` | see the role table above | `low` / `medium` / `high` / `xhigh` / `max` |
| `CLAUDE_MAX_TOKENS_<ROLE>` | 32000 / 32000 / 4000 / 96000 | Output cap; thinking counts toward it, and hitting it is a failure |
| `CLAUDE_REFUSAL_FALLBACK` | `default` | Server-side refusal fallback on the Sonnet/Opus roles; `off` disables it |
| `ANTHROPIC_MAX_RETRIES` | `2` | The SDK's own retries |
| `SOURCE_LANG` / `TARGET_LANG` | `Chinese` / `English` | |
| `MINERU_MODEL_VERSION` | `vlm` | `vlm` or `pipeline` |
| `MINERU_LANGUAGE` | `ch` | OCR language hint |
| `MINERU_TIMEOUT_MINUTES` | `120` | Default poll limit for one parse run |
| `KIT_CHUNK_CHARS` | `30000` | Chunk budget, in source characters |
| `SOURCE_RESIDUE_THRESHOLD` | `0.04` | CJK fraction above which a chunk reads as untranslated |
| `SPLIT_MAX_PAGES` | `190` | Per-part page target (API hard limit 200) |
| `SPLIT_MAX_SIZE_MB` | `180` | Per-part size target (API hard limit 200) |
| `FIGURE_DROP_CONFIDENCE` | `0.9` | A decoration is dropped only at or above this confidence |
| `VISION_MIN_INTERVAL_SECONDS` | `0` | Pause between vision calls, for a low rate limit |

No temperature is configurable: the current Claude models reject non-default
sampling parameters, so determinism comes from the on-disk caches — each
answer is asked for once and reused.

`HTTP_PROXY` / `HTTPS_PROXY` are honoured if set — MinerU's storage sits behind
Aliyun OSS, which is slow from some networks.

**Exit codes** across the scripts: `0` done, `1` failure (or a lint/build
gate), `2` usage error, `3` exchange requests pending, `4` a remote job (parse,
batch) still running.

---

## Why the stages exist

The cloud VLM parser fixed most of what used to make builds painful. Measured
across the full reference book, it produces **zero** space-mangled LaTeX, zero
whitespace inside inline delimiters (of 6,641 spans), zero escaped `\$`, zero
welded `$$$$`, even `$$` parity, and 570 equation tags. Those checks are still
in `lint.py` as diagnostics, with deliberately no repair passes behind them.

| Issue | Stage |
|---|---|
| Front matter and indexes parsed into fake headings, quota spent on them | page range |
| Headings flat below level 2 (422 of 442 at `##`) | prepare → profile + re-level |
| Parts and readings mistaken for chapters, shifting every chapter number | prepare → structure |
| Split headings (`## 第13章` / `## 电势`) | prepare → normalize |
| Tables emitted as raw HTML | prepare → tables |
| Model corrupting 64-char image hashes (~7.7%) | prepare → tokenize |
| Every figure the same width; tall ones overflowing the page | prepare → figures, build |
| Figure numbers shifted by uncaptioned images | prepare → figures, build |
| Decorations and text-as-image extracted as figures | prepare → figures |
| A translation that compiles but changed a number, sign or formula | lint → fidelity |
| A reply cut off mid-chunk | translate → sentinel, lint |

---

## Troubleshooting

**`MINERU_TOKEN is missing`** — no token in `.env`. Documented API error codes
are translated into actionable messages: `A0202` bad token, `A0211` expired,
`-60005` file too large, `-60006` too many pages, `-60018` daily limit reached.

**A part failed to parse** — the script names the file and the server's
`err_msg`, and the other parts still merge. Re-run with `--force` after fixing.

**`parse_book.py` exits 4** — the batch is still running server-side. Run the
same command again; it resumes polling and never re-uploads.

**`prepare_kit.py` exits 3** — no-API mode is waiting for answers in
`llm/pending.json` (see The exchange).

**Upload times out** — set `HTTP_PROXY` / `HTTPS_PROXY`, or raise
`MINERU_UPLOAD_TIMEOUT_SECONDS`.

**`truncated-sentinel`** — the reply does not end with `% END-OF-CHUNK`, so it
was cut off. Re-translate the chunk.

**A fidelity finding that is not a change** — add a waiver to
`kit/lint_waivers.json` with the concrete reason.

**`pdflatex failed on pass 1/2`** — every error is listed, located in its
reply. Run the lint stage first; it finds nearly all of these at once.

**`Environment solution undefined`** — the model invented an environment; only
those in [src/latex.py](src/latex.py)`::DEFINED_ENVIRONMENTS` exist.

**A chunk came back untranslated** — lint catches it by CJK density.
Re-translate it.

---

## Development

```bash
uv run python -m pytest tests/ -q
```

The suite makes **no real API calls and no MinerU calls**: LLMs are fakes
implementing `BaseLLM` / `VisionLLM`, the Anthropic provider and the batch
translator are tested against an `httpx2.MockTransport` serving the real wire
format, and MinerU against an `httpx.MockTransport`. If `pdflatex` is
installed, one test compiles the real preamble's figure macros.
`scripts/test_api.py` is the separate smoke test against the real Claude API
(API mode).

| Module | Role |
|---|---|
| [src/mineru_api.py](src/mineru_api.py) | Cloud API: split, upload, poll (resumable), merge |
| [src/page_range.py](src/page_range.py) | Body page range, trimming, first-heading check |
| [src/normalize.py](src/normalize.py) | Split-heading repair + diagnostics |
| [src/tables.py](src/tables.py) | HTML tables → pipe tables |
| [src/latex.py](src/latex.py) | Preamble + prompt assets, fragment scanning, assembly |
| [src/profiler.py](src/profiler.py) | Cached per-book LLM call: glossary + heading levels |
| [src/structure.py](src/structure.py) | Parts / chapters / readings: Claude labels, code checks |
| [src/figures.py](src/figures.py) | Figure numbers, widths and drops (`figures.json`), vision classifier |
| [src/kit.py](src/kit.py) | Tokenising, chunking, assembly |
| [src/lint.py](src/lint.py) | Every check, in one pass; waivers; per-book allowlist |
| [src/fidelity.py](src/fidelity.py) | Source-vs-translation mathematics |
| [src/build.py](src/build.py) | `pdflatex` invocation |
| [src/publish.py](src/publish.py) | `data/output/<book>/` |
| [src/run_state.py](src/run_state.py) | Resumable state of an unattended run |
| [src/batch_translate.py](src/batch_translate.py) | API mode: translation as one Message Batch |
| [src/llm/providers/anthropic.py](src/llm/providers/anthropic.py) | Claude via the Messages API, per role |
| [src/llm/exchange.py](src/llm/exchange.py) | Claude via request/response files (no-API mode) |
| [src/splitter.py](src/splitter.py) | PDF page-range splitting |

There is also a fully automated Python-only path, `uv run python -m src.main
<pdf>` (API mode), which parses and translates without the review stages.
Prefer the agent mode for anything book-length.
