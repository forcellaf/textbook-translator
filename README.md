# translation_project

Translates scanned textbooks (Chinese → English) into a typeset PDF.

The pipeline is four stages: **parse → prepare → lint → build**. Parsing is an
HTTP call to the MinerU cloud API; everything else is local CPU work plus one
cached LLM call per book. There is no GPU step and no notebook.

```
PDF ──parse──► merged.md ──prepare──► kit/chunks/*.md
                                           │
                                (you translate, to LaTeX)
                                           │
                                           ▼
PDF ◄──build── translated_book.tex ◄─lint─ kit/translated/*.tex
```

The source chunks are Markdown, because that is what the parser produces. The
replies are **LaTeX body fragments**, which the build wraps in
[assets/preamble.tex](assets/preamble.tex) and compiles directly.

---

## Setup

**1. Install Python dependencies** (Python 3.12+):

```bash
uv sync
```

**2. Install a LaTeX distribution.** This is the one dependency not installed
through Python: TeX Live or MiKTeX, providing `pdflatex`.

```bash
pdflatex --version
```

**3. Create `.env`** in the project root:

```bash
MINERU_TOKEN=your-mineru-token
DEEPSEEK_API_KEY=your-deepseek-key
LLM_PROVIDER=deepseek
SOURCE_LANG=Chinese
TARGET_LANG=English
```

Get a MinerU token at <https://mineru.net/apiManage/token>. Only the active
`LLM_PROVIDER`'s key is required — a DeepSeek-only setup is not blocked by a
missing Gemini key.

---

## Data layout: one repo, many books

**The repo is the tool; books are the data. Never copy the repo per book.**
Each book gets its own work directory, so several coexist in one checkout:

```
data/work/<book_name>/
├── parts/              part PDFs sent to the API (+ manifest.json)
├── images/             every figure, flat, named by content hash
├── merged.md           raw parser output          ← stage 1
├── normalized.md       after split-heading repair ← stage 2
├── prepared.md         after re-levelling + tables
├── book_profile.json   cached LLM call: glossary + heading levels
└── kit/
    ├── chunks/001.md…       source chunks, image paths tokenised
    ├── translated/001.tex…  your saved replies (you fill this in)
    ├── images/
    ├── image_map.json       IMG_nnnn → real filename
    ├── source_clean.md
    ├── system_prompt.txt
    ├── translated_book.tex  ← stage 4 assembles this
    └── translated_book.pdf  ← the deliverable
```

`translated_book.tex` has to sit beside `images/`: the preamble's
`\graphicspath` resolves figures relative to the `.tex`.

`data/work/` is gitignored.

---

## Stage 1 — Parse

```bash
uv run python scripts/parse_book.py data/input/physics.pdf data/work/physics
```

Splits the PDF to fit the API's hard limits (200 pages / 200 MB per file),
uploads every part in one batch, polls with per-file page progress, and merges
the results into `merged.md` plus a flat `images/`.

| Flag | Purpose |
|---|---|
| `--model {vlm,pipeline}` | Parser backend (default `vlm`) |
| `--language` | OCR language hint (default `ch`) |
| `--force` | Re-parse even if `merged.md` is current |

**Watch the page count it logs.** The quota is 1000 pages/day at top priority,
and a 500-page textbook is half of it.

**`vlm` vs `pipeline` is a real per-book decision.** `vlm` is dramatically
cleaner on formula-dense material — `$10^{-20}$` where `pipeline` gives
`$1 0 ^ { - 2 0 }$` — and it preserves merged table cells. But VLM backends can
hallucinate, which for a physics textbook means a plausible equation with a
wrong exponent that no automated check will catch. For a new book, parse one
chapter each way and diff the formulas before committing to `vlm`.

Re-running is cheap: parsing is skipped when `merged.md` is newer than the PDF.
If a part fails server-side, the others still merge and the script exits
non-zero naming the failure — a partial book is never silently mistaken for a
complete one.

## Stage 2 — Prepare the translation kit

```bash
uv run python scripts/prepare_kit.py data/work/physics
```

Six steps, in this order:

1. **normalize** — merge split headings; report anything else short
2. **profile** — one cached LLM call: glossary + a level per heading
3. **re-level** — apply that classification (pure, no LLM)
4. **tables** — HTML → markdown pipe tables
5. **tokenize** — image paths → `IMG_nnnn`
6. **chunk** — ~30,000 chars, on paragraph boundaries only

| Flag | Purpose |
|---|---|
| `--no-profile` | Skip the LLM call; headings keep their parsed levels |
| `--force-profile` | Ignore the cached profile |
| `--chunk-chars N` | Override the chunk budget |
| `--fix-math-spacing` | Trim whitespace inside inline `$` (see below) |
| `-v` | Also list every short heading left unchanged |

**Read the output.** Three things want your eyes:

- **Merged headings** are listed individually. On the reference book this is 5
  chapter merges (`第13章` + `电势` → `第13章 电势`) and 3 part-title merges.
- **`part-ordinal-missing`** means a part title was recovered but its number
  was not in the parsed text at all (`第篇 电磁学`). The pipeline will not
  invent it — insert the digit by hand in `normalized.md` and re-run.
- **Tables left as HTML** are reported with a reason. Merged cells (`colspan`/
  `rowspan`) cannot be expressed as a pipe table without duplicating content,
  so they stay as HTML rather than being guessed at. On the reference book,
  19 of 24 convert and the 5 skipped are legitimate merged header cells, not
  damage.

`book_profile.json` is plain JSON and meant to be hand-edited. If the
classifier puts one heading at the wrong level, fix that line and re-run —
the cached profile is reused, so no LLM call is repeated.

**Then translate.** Paste `kit/system_prompt.txt` as the system instruction,
send `kit/chunks/001.md` … one at a time in a fresh conversation each, and save
each reply as `kit/translated/<same-number>.tex`.

`system_prompt.txt` is [assets/system_prompt_latex.txt](assets/system_prompt_latex.txt)
with the book's context and glossary substituted into it. That file is meant to
be edited: when the model gets something wrong repeatedly, tighten the rule it
broke and re-run stage 2.

Two constraints worth knowing:

- Chunk size is bound by the model's **output** limit, not its context window —
  it has to emit a full translation of everything you give it. Measured on the
  reference book, output ran 1.33×–2.87× the input length (mean 2.29×), and the
  longest reply hit 143,000 characters ≈ 36,000 tokens, occasionally coming back
  truncated. LaTeX is more verbose again, so the budget is 30,000 source
  characters: worst case ≈ 87,000 characters ≈ 22,000 tokens.
- **The model writes body fragments only.** No `\documentclass`, no
  `\usepackage`, no `\begin{document}` — the preamble is added once, at
  assembly, because a preamble re-invented per conversation comes back
  different every time and the book stops compiling.

## Stage 3 — Lint

```bash
uv run python scripts/lint_translation.py data/work/physics/kit
```

**This is the stage that ends the debugging loop.** LaTeX reports one error per
compile — so four defects used to cost four full build cycles. Every check runs
at once here, with line numbers and context.

Findings are split three ways:

- **review** — a human must look. Exits non-zero, which gates the build.
- **auto-fix** — provably safe. Apply with `--fix`.
- **info** — context, not a problem.

Per chunk pair: CJK residue, truncation (both an absolute floor and an outlier
against the book's own median expansion), and image-token integrity — every
`IMG_nnnn` present, none invented.

On the translated LaTeX: unbalanced and crossed environments, undefined
environments, preamble leakage, brace balance, inline `$` parity per paragraph,
duplicated numbering, dangling argument commands (`\mathrm$`), unknown command
tokens, and figure paths that do not resolve on disk.

**Every one of those LaTeX checks was a real failure during testing, and none
of them is visible by reading the file** — the output looks correct and only
fails at compile time, one error per compile. The undefined-environment check
exists because the model invented `solution` and `theorem`; the numbering check
because "12.1 12.1 Electric Charge" reads fine to a compiler.

Three behaviours are deliberate:

- **Unknown commands are reported, never auto-fixed.** On the reference book
  this flagged 41 tokens, of which 40 were real LaTeX simply missing from the
  known-commands list and 1 was the genuine defect. A checker with that hit
  rate must not edit. If a flagged command is real, add it to `KNOWN_COMMANDS`
  in [src/lint.py](src/lint.py).
- **A doubled known command is auto-fixed** (`\mathrmmathrm` → `\mathrm`): it
  cannot be anything but damage.
- **So is a duplicated number** (`\section{12.1 Electric Charge}` →
  `\section{Electric Charge}`): the pattern is unambiguous, and a title that
  merely starts with a digit (`3D Charge Distributions`) is left alone.

Lint one file instead of a kit with `--tex one_chunk.tex` (a translated
fragment) or `--markdown merged.md` (the parsed source).

## Stage 4 — Build

```bash
uv run python scripts/build_pdf.py data/work/physics/kit
```

Lints the kit, concatenates the saved LaTeX replies, restores the real image
filenames from `image_map.json`, wraps the result in
[assets/preamble.tex](assets/preamble.tex), and runs `pdflatex` twice — the
second pass is what fills in the table of contents.

| Flag | Purpose |
|---|---|
| `--no-lint` | Build even when findings need review |
| `--fix` | Apply the safe repairs to the assembled `.tex` first |
| `--engine` | Default `pdflatex` (see below) |

The lint gate is on by default. A broken figure or an untranslated chunk is far
cheaper to fix now than to find in a 700-page PDF.

`pdflatex` is enough: the preamble uses `inputenc`, and a finished book has no
Chinese left in it. If one does, `assets/preamble.tex` carries a commented
two-line swap to `fontspec` + `xeCJK` — uncomment it and pass
`--engine xelatex`.

The preamble is a verified configuration, hand-maintained, and the
specification rather than an output of the code: `openany` removes the blank
verso before every chapter, `\bookfig` caps figure width (MinerU images carry
no usable intrinsic size, so without it every figure renders wider than the
text block), and `\setcounter{chapter}{11}` is what makes a volume starting at
chapter 12 number its sections `12.1`, `12.2` … instead of `1.1`, `1.2`. Set
that to (first chapter number − 1) for a different volume.

A compile failure is printed as the error lines pulled out of the log, not the
log itself, and the script exits non-zero.

---

## Configuration

All settings are environment variables, read in [src/config.py](src/config.py).

**Required**

| Variable | Purpose |
|---|---|
| `MINERU_TOKEN` | MinerU cloud API token |
| `DEEPSEEK_API_KEY` | Or the key for whichever `LLM_PROVIDER` you set |

**Common**

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `deepseek` | `deepseek` or `gemini` |
| `SOURCE_LANG` / `TARGET_LANG` | `Chinese` / `English` | |
| `MINERU_MODEL_VERSION` | `vlm` | `vlm` or `pipeline` |
| `MINERU_LANGUAGE` | `ch` | OCR language hint |
| `MINERU_TIMEOUT_MINUTES` | `120` | Poll timeout for a whole batch |
| `KIT_CHUNK_CHARS` | `30000` | Chunk budget, in source characters |
| `SOURCE_RESIDUE_THRESHOLD` | `0.04` | CJK fraction above which a chunk reads as untranslated |
| `SPLIT_MAX_PAGES` | `190` | Per-part page target (API hard limit 200) |
| `SPLIT_MAX_SIZE_MB` | `180` | Per-part size target (API hard limit 200) |

`HTTP_PROXY` / `HTTPS_PROXY` are honoured if set — MinerU's storage sits behind
Aliyun OSS, which is slow from some networks.

---

## Why the stages exist

The cloud VLM parser fixed most of what used to make builds painful. Measured
across the full reference book, it produces **zero** space-mangled LaTeX, zero
whitespace inside inline delimiters (of 6,641 spans), zero escaped `\$`, zero
welded `$$$$`, even `$$` parity, and 570 equation tags. Merged table cells
survive instead of being collapsed.

Those checks are still in `lint.py` as **diagnostics** — they are cheap, and a
future parser change could reintroduce any of them — but there are deliberately
no repair passes behind them. `--fix-math-spacing` exists only as insurance if
the parser regresses.

What genuinely still needs handling, and which stage handles it:

| Issue | Stage |
|---|---|
| Headings flat below level 2 (422 of 442 at `##`) | prepare → profile + re-level |
| Split headings (`## 第13章` / `## 电势`) | prepare → normalize |
| Tables emitted as raw HTML, which the model reproduces as HTML | prepare → tables |
| Model corrupting 64-char image hashes (~7.7%) | prepare → tokenize |

Display math is still character-spaced. That is **cosmetic only** — with no
delimiter-adjacent whitespace it compiles correctly. Don't "fix" it.

---

## Troubleshooting

**`MINERU_TOKEN is missing`** — no token in `.env`. Documented API error codes
are translated into actionable messages: `A0202` bad token, `A0211` expired,
`-60005` file too large, `-60006` too many pages, `-60018` daily limit reached.

**A part failed to parse** — the script names the file and the server's
`err_msg`, and the other parts still merge. Re-run with `--force` after fixing.

**Upload times out** — MinerU's storage is Aliyun OSS. Set `HTTP_PROXY` /
`HTTPS_PROXY`, or raise `MINERU_UPLOAD_TIMEOUT_SECONDS`.

**`pdflatex failed on pass 1/2`** — the error lines from the log are printed,
which is where LaTeX names the actual problem. Run the lint stage first; it
finds nearly all of these, all at once, before the first compile.

**`Environment solution undefined`** — the model invented an environment. Lint
reports these by name and line; only the list in
[src/latex.py](src/latex.py)`::DEFINED_ENVIRONMENTS` exists.

**A number appears twice** ("12.1 12.1 Electric Charge") — the model kept a
source number LaTeX generates itself. `--fix` strips it.

**Tables missing or mangled in the PDF** — check the prepare output for tables
left as HTML; merged cells are reported, not guessed at.

**A chunk came back untranslated** — lint catches it by CJK density, because an
untranslated chunk is *full length* and correctly formatted. Re-send that chunk.

---

## Development

```bash
DEEPSEEK_API_KEY=fake uv run python -m pytest tests/ -q
```

The suite makes **no real API calls**: LLMs are fakes implementing `BaseLLM`,
and MinerU is served by an `httpx.MockTransport`. Fixtures are built from real
defect signatures, so each fails without its fix — including negative fixtures
proving that `提要` and `习题` (19 legitimate occurrences each) are never
merged, and that the historical defects are still *detected* even though
nothing repairs them any more.

| Module | Role |
|---|---|
| [src/mineru_api.py](src/mineru_api.py) | Cloud API: split, upload, poll, merge |
| [src/normalize.py](src/normalize.py) | Split-heading repair + diagnostics |
| [src/tables.py](src/tables.py) | HTML tables → pipe tables |
| [src/latex.py](src/latex.py) | Preamble + prompt assets, fragment scanning, assembly |
| [src/profiler.py](src/profiler.py) | Cached per-book LLM call: glossary + heading levels |
| [src/kit.py](src/kit.py) | Tokenising, chunking, assembly, packaging |
| [src/lint.py](src/lint.py) | Every check, in one pass |
| [src/build.py](src/build.py) | `pdflatex` invocation |
| [src/splitter.py](src/splitter.py) | PDF page-range splitting |

There is also a fully automated path, `uv run python -m src.main <pdf>`, which
parses and translates without the review stages. Prefer the four scripts for
anything book-length.
