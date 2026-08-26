# Task: implement the translation stage, book profiler, and LaTeX output

## Context

This repo parses scanned Chinese textbooks into Markdown (MinerU) and is supposed to
translate them to English via the Gemini API. **The parsing half works. The translation
half was never written.**

Specifically: `src/main.py` does `from src.translator import translate_markdown` inside a
try/except, but `src/translator.py` does not exist. The import silently fails and the
pipeline falls back to writing out the *untranslated* Chinese Markdown with only a logged
warning.

The supporting pieces already exist and work — **use them, do not rewrite them**:

- `src/llm/base.py` — `BaseLLM` ABC with `generate(system_prompt, user_text, temperature) -> str`
- `src/llm/factory.py` — `get_llm()` returns the configured provider
- `src/llm/providers/gemini.py` — `GeminiProvider`, raises `RuntimeError` on API failure
- `src/chapter_splitter.py` — `split_into_chapters(md_path, out_dir) -> ChapterResult`
- `src/parser.py` — `parse_book(pdf_filename, mode=...) -> Path` (returns `merged.md`)
- `src/merger.py`, `src/splitter.py`, `src/models.py`, `src/config.py`

## Critical constraint: the public API is fixed

A Jupyter notebook (not in this repo) already calls these functions. **The signatures
below are a contract. Do not rename, reorder, or change defaults.** If you think a
signature is wrong, implement it as specified anyway and note your concern separately.

### `src/profiler.py` (new file)

```python
PROFILE_FILENAME = "book_profile.json"

@dataclass(frozen=True)
class BookProfile:
    subject: str = "general"
    subfield: str = ""
    education_level: str = "university"
    audience: str = "students"
    register: str = "formal academic prose"
    notation_notes: str = ""
    structural_elements: tuple[str, ...] = ()
    glossary: tuple[tuple[str, str], ...] = ()   # (source_term, target_term)
    latex_documentclass: str = "book"
    latex_packages: tuple[str, ...] = ()
    summary: str = ""

    @classmethod
    def generic(cls) -> "BookProfile": ...
    def to_json(self) -> str: ...
    @classmethod
    def from_dict(cls, data: dict) -> "BookProfile": ...

def profile_book(
    merged_md_path: Path,
    work_dir: Path,
    *,
    llm: BaseLLM | None = None,
    source_lang: str = SOURCE_LANG,
    target_lang: str = TARGET_LANG,
    force: bool = False,
) -> BookProfile: ...

def profile_to_prompt_block(profile: BookProfile, *, max_glossary: int = 40) -> str: ...
```

### `src/latex.py` (new file)

```python
def build_preamble(profile: BookProfile, title: str, *, target_lang: str = "English") -> str: ...
def escape_latex(text: str) -> str: ...
def validate_fragment(tex: str) -> list[str]: ...   # [] means valid
def assemble_document(
    body_parts: list[str],
    profile: BookProfile,
    title: str,
    output_path: Path,
    *,
    target_lang: str = "English",
) -> Path: ...
def compile_pdf(
    tex_path: Path,
    *,
    engine: str = "xelatex",
    passes: int = 2,
    timeout_seconds: int = 1800,
) -> tuple[bool, str]: ...   # (succeeded, log_excerpt)
```

### `src/translator.py` (new file)

```python
class TranslationError(RuntimeError): ...

def chunk_markdown(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[str]: ...

def translate_chunk(
    chunk: str,
    *,
    llm: BaseLLM,
    source_lang: str = SOURCE_LANG,
    target_lang: str = TARGET_LANG,
    profile: BookProfile | None = None,
    output_format: str = "markdown",
) -> str: ...

def translate_markdown(
    markdown_text: str,
    target_lang: str = TARGET_LANG,
    *,
    source_lang: str = SOURCE_LANG,
    llm: BaseLLM | None = None,
    on_chunk_done: Callable[[int, int, str], None] | None = None,
    profile: BookProfile | None = None,
    output_format: str = "markdown",
) -> str: ...

def translate_book(
    merged_md_path: Path,
    work_dir: Path,
    *,
    target_lang: str = TARGET_LANG,
    source_lang: str = SOURCE_LANG,
    llm: BaseLLM | None = None,
    resume: bool = True,
    output_format: str = "markdown",
    profile: BookProfile | None = None,
    use_profile: bool = True,
    title: str | None = None,
) -> Path: ...
```

`translate_markdown`'s first two positional parameters must stay positional — `src/main.py`
calls `translate_markdown(md, lang)`.

---

## Behavioral requirements

### `chunk_markdown`

Split on blank-line paragraph boundaries only, never mid-paragraph, so Markdown tables and
list items stay intact. Budget is measured in *estimated* tokens — there is no tokenizer
available, so approximate from character count using a constant that skews toward the CJK
end (CJK runs ~1 token/char, Markdown/English ~1 token per 4 chars). A single paragraph
that alone exceeds the budget must be kept **whole**, not truncated: truncating mid-table
or mid-formula corrupts content unrecoverably, while one oversized request is a safe
failure. Empty/whitespace input returns `[]`.

### `translate_chunk` — two resilience layers that must NOT compound

This is the subtlest requirement. Get it wrong and a dead API key takes ~4x longer to
report than it should.

1. **Transient failures** (network, rate limit): retry inside a *single* call using
   `tenacity` with `stop_after_attempt(API_MAX_RETRIES)` and exponential backoff, retrying
   on `RuntimeError` (what `GeminiProvider` raises). If it still fails after that full
   backoff schedule, raise `TranslationError` **immediately**.

2. **Bad-but-successful output** (empty, implausibly short, or structurally invalid
   LaTeX): re-ask up to `MAX_HEAL_ATTEMPTS` times with the *specific* problem quoted back
   in the prompt.

**Do not put the tenacity-wrapped call inside the heal loop such that a persistent hard
failure re-runs the whole backoff schedule once per heal attempt.** A bad API key must
surface after one backoff schedule, not `API_MAX_RETRIES × (MAX_HEAL_ATTEMPTS + 1)`
attempts.

The "implausibly short" heuristic: flag empty output, or output under 30% of source length
when the source is longer than 200 chars. In LaTeX mode, additionally run
`validate_fragment()` and treat any issues as a heal trigger. The heal prompt must name the
actual problem (e.g. "output was empty or implausibly short (22 chars for 600 chars of
input)", or "invalid LaTeX: \\begin{equation} is never closed") — not a generic "try again".

After exhausting heals, log an error and return the last result rather than raising.

### `profile_book` — must never be fatal

One LLM call per book. Sample ~24,000 chars: the first ~6,000 (front matter/TOC/preface,
which usually states level and audience outright) plus ~5 interior slices of ~3,000 chars
each, anchored at Markdown headings where possible so samples start at section boundaries.

Prompt the model to return **only** a JSON object with the `BookProfile` fields. Ask for
15–40 glossary entries as `{"source": ..., "target": ...}` objects, prioritizing terms with
several plausible translations. Tell it the glossary is the most important field.

Parsing must tolerate reality: strip ``` fences, fall back to slicing between the first `{`
and last `}`. `from_dict` must accept both object-style and 2-element-list-style glossary
entries, skip malformed ones, coerce a bare string to a 1-tuple for list fields, and
tolerate missing keys.

**On any failure — unreachable LLM, unparseable JSON — log a warning and return
`BookProfile.generic()`.** Never raise. A profiling hiccup must degrade quality, not kill a
multi-hour run. Do **not** cache a failed profile (the failure would become sticky).

Cache successes to `work_dir/book_profile.json`; return the cached profile on later calls
unless `force=True`.

### `profile_to_prompt_block`

Renders the profile into a prompt fragment. Keep it **compact** — this text is prepended to
every chunk's system prompt, so its length is multiplied by chunk count. Include a
"Required terminology" section listing `source -> target` pairs with an instruction to use
them exactly and consistently. This is the mechanism that keeps independently-translated
chunks consistent; without it the same term comes back three different ways across chapters.

### LaTeX: the model writes body fragments only

The system prompt for `output_format="latex"` must forbid `\documentclass`, `\usepackage`,
`\begin{document}`, `\end{document}`, code fences, and commentary. Preamble content must be
identical book-wide, and it is exactly what hundreds of independent LLM calls would render
inconsistently — so `src/latex.py` owns the preamble and the model owns only chunk-local
content.

Structure mapping to specify in the prompt: `#`→`\chapter`, `##`→`\section`,
`###`→`\subsection`; lists→itemize/enumerate; tables→`tabular` in a `table` float with
booktabs rules; `![alt](path)`→`figure` with `\includegraphics` **keeping the path exactly
as given**; inline `$...$` stays inline; display math→`equation` or `\[...\]`. Preserve all
math verbatim — never "fix" or re-derive formulas. Escape `& % # _` and literal `$` in prose.

`validate_fragment` must detect: preamble-level commands leaking in; `\begin`/`\end`
mismatches including *crossed* pairs (`\begin{itemize}...\end{enumerate}`); unbalanced
braces ignoring escaped `\{` `\}`; odd `$` count ignoring escaped `\$` and `$$`. Return
human-readable strings.

`build_preamble` must use XeLaTeX + `fontspec` with a Noto CJK fallback font guarded by
`\IfFontExistsTF` — one OCR'd Chinese figure label surviving translation would otherwise
fail the entire build. Merge profile-suggested packages with a baseline (amsmath, amssymb,
graphicx, booktabs, longtable, array, hyperref). **Treat profile package names as untrusted
LLM output**: drop anything not matching `^[A-Za-z][A-Za-z0-9\-]*$`, and denylist
`inputenc, fontenc, ctex, xeCJK, fontspec, geometry, babel` (they conflict with the fixed
preamble). Clamp `latex_documentclass` to `book|report|article`, defaulting to `book`.

`compile_pdf` must run `-interaction=nonstopmode` (so a bad macro can't hang waiting for
input), run `passes` times so the TOC resolves, return `(False, message)` rather than
raising on failure, extract error lines from the log rather than dumping the whole thing,
and return a clear message if the engine binary isn't on PATH.

### `translate_book` — checkpointing and lazy profiling

Per-chapter checkpoints under `work_dir/translated_chapters/`. Skip a chapter when
`resume=True` and its checkpoint exists and is newer than the source chapter file, so
re-running after an interrupted session is always safe.

**Checkpoint filenames must carry a format-specific suffix** (`.md` vs `.tex`) so switching
`output_format` between runs re-translates instead of silently mixing formats into one
document.

**Profile lazily**: determine which chapters actually need work *first*, and only call
`profile_book` if that list is non-empty. Re-running a fully-translated book must cost
**zero** LLM calls.

Markdown mode → concatenate to `work_dir/translated_merged.md`. LaTeX mode → pass fragments
to `assemble_document` → `work_dir/translated_book.tex`. Raise `ValueError` on an unknown
`output_format` (check this before doing any work).

### `scripts/test_api.py` (new file)

A standalone smoke test that calls the **real** API (everything in `tests/` uses fakes).
Five stages, cheapest first, stopping at the first failure so output points at the specific
broken layer:

1. Config/credentials load
2. One raw `get_llm().generate()` call
3. `profile_book` on a small sample — **treat a fallback to the generic profile as a
   FAILURE here**, which is deliberately the opposite of production behavior: in production
   the fallback protects a long run, but in a smoke test the silent degradation is exactly
   what you need to know about
4. `translate_markdown` in Markdown mode + structural spot-checks (heading, math, table
   preserved; no CJK remaining)
5. `translate_markdown` in LaTeX mode + `validate_fragment` + no-preamble-leakage check

Support `--quick` for stages 1–2 only. Failures must print a clear message and remediation
hint, not a raw traceback. Add `sys.path.insert(0, <repo root>)` so it runs from anywhere.

---

## Also do

Update the stale note in `src/main.py`'s module docstring that says translation "is not
wired up yet" / "still a stub" — it will be wrong once this lands.

## Do NOT

- Modify `src/parser.py`, `src/splitter.py`, `src/merger.py`, `src/chapter_splitter.py`,
  `src/models.py`, `src/llm/**`, or `pyproject.toml`.
- Add new dependencies. `tenacity`, `google-genai`, and `python-dotenv` are already declared.
- Fill in the empty `src/prompts/` or `src/translation_project/` directories, or implement a
  DeepSeek provider (`config.py` has unused DeepSeek constants). Those are scaffolding for
  unstarted work — leave them alone.
- Touch `.gitignore`.

## Tests

Add `tests/test_translator.py` and `tests/test_profiler_latex.py`, matching the existing
style in `tests/`. Use a fake `BaseLLM` implementation that records calls and returns
scripted responses — **no real API calls in the test suite**.

Cover at minimum:

- chunking: empty input, single chunk, multi-chunk paragraph boundaries, oversized single
  paragraph kept whole
- heal on empty response; `TranslationError` after retries are exhausted
- heal on invalid LaTeX — **make the fake's fragments long enough to pass the length
  heuristic**, otherwise the length check fires first and the LaTeX validation path is never
  actually exercised (this is an easy trap to fall into)
- `from_dict` with object-style glossary, list-style glossary, junk entries, missing keys
- profile caching (second call must not hit the LLM); generic fallback on bad JSON and on a
  raising LLM; failed profile not cached
- `validate_fragment` flagging each failure class; sound LaTeX passing
- preamble denylist and documentclass clamping
- `assemble_document` producing exactly one `\begin{document}`/`\end{document}` in order
- `translate_book` LaTeX mode producing `translated_book.tex` with `.tex` chapter
  checkpoints
- resume skipping already-translated chapters
- a fully-resumed book making **zero** LLM calls (locks in lazy profiling)

## Verify before you finish

```bash
GEMINI_API_KEY=fake python -m pytest tests/ -q
```

All tests must pass, including the ~25 that already exist. Then confirm there is no
circular import between the three new modules (`translator` imports from both `profiler`
and `latex`):

```bash
GEMINI_API_KEY=fake python -c "import src.translator, src.profiler, src.latex; print('ok')"
```

Do not commit or push. I'll review and push myself.
