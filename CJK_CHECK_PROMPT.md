# Task: detect untranslated output and trigger the existing heal retry

## The bug

A real run just produced a `.tex` where **6.3% of the document was still Chinese** — 79 of
1,140 lines, including the chapter title (`\chapter{静电场}`) and several figure captions.
The model converted Markdown structure to LaTeX correctly but did not translate the text
in those chunks; it passed the source through verbatim.

The clearest symptom, from the real output:

```latex
\section{Electric Field and Electric Field Strength}
\section{电场 电场强度}
```

It emitted the heading translated, then emitted it again untranslated and continued in
Chinese for the rest of that chunk.

## Why nothing caught it

`translate_chunk` already has a heal loop driven by `_diagnose_output` (see
`src/translator.py`). Neither existing check fires here:

- **`_is_suspect_output`** measures output *length*. An untranslated chunk is full-length,
  so it passes.
- **`validate_fragment`** (`src/latex.py`) measures LaTeX *structure*. The Chinese sits
  inside perfectly well-formed `\section{}` blocks, so it passes.

There is no check for *"did it actually translate?"* — that is the entire gap. The heal
machinery to fix this already exists and works; it just needs a third detector wired into
it.

## What to build

### A source-language-residue check in `src/translator.py`

Add a detector that measures how much source-language script remains in the output, and
wire it into `_diagnose_output` alongside the existing checks so a failure triggers the
normal heal retry (with the problem quoted back to the model, as the other detectors do).

Requirements:

**Only run it when it makes sense.** The check must be a no-op unless the source language
uses a script the target does not. Translating Chinese → English, CJK residue is a
reliable signal. Translating English → French it is meaningless, and Chinese → Japanese it
would be actively wrong. Gate it on the actual `source_lang`/`target_lang` pair rather than
assuming Chinese → English. If you cannot determine the scripts, skip the check rather
than guessing — a false positive here burns an extra API call on every chunk.

**Exclude math and image paths before measuring.** Strip `$...$`, `$$...$$`, `\[...\]`,
`equation`/`align` environments, and `\includegraphics{...}` arguments from the text first.
A formula legitimately containing a CJK subscript, or a filename, must not count as
untranslated prose.

**Pick the threshold from real data, and justify it in a comment.** The source markdown for
this book is ~35% CJK by character. A correct translation should be near 0%. Some residue
is legitimate — an untranslatable proper noun, a term the glossary says to leave. Use a
threshold well above zero but far below source density; something in the 3–5% range of
non-math characters is defensible. Make it a module-level named constant, not a magic
number inline.

**Report it usefully.** The message fed into the heal prompt should say what was found,
e.g. `"output is still 31% Chinese - the text was not translated"`. The existing heal
prompt quotes the problem back to the model, and a specific message is what makes the
retry likely to succeed.

**Also catch the duplicated-heading symptom if it is cheap to do so.** The example above
shows the same heading emitted twice, once per language. If that falls out of the residue
check naturally, fine — do not build a separate special case for it unless it is simple.

### Configuration

Expose the threshold via `src/config.py` following the existing style there (env-var with a
default), so it can be tuned per book without a code change. Do not add new dependencies.

## Tests

Add tests to `tests/test_translator.py` in the existing style, using the existing fake
`BaseLLM`. Cover at minimum:

- An all-Chinese output for a Chinese→English chunk triggers a heal, and the heal prompt
  mentions the residue (this is the regression test for the actual bug)
- A properly translated English output does **not** trigger a heal
- A translation containing a CJK character inside `$...$` math does **not** trigger a heal
  (the math-stripping requirement)
- A small amount of residual CJK below the threshold does **not** trigger a heal
- The check is skipped entirely for a language pair where it does not apply (e.g.
  English→French), with no extra LLM call
- After `MAX_HEAL_ATTEMPTS` exhausted, the last output is still returned rather than
  raising — matching current behaviour

**Important trap:** make the fake LLM's responses long enough to clear
`_is_suspect_output`'s length heuristic. If the fixture text is short, the length check
fires first and your new check is never reached, so the test passes for the wrong reason.
This exact mistake was made when the LaTeX-validation tests were first written.

## Verify

```bash
DEEPSEEK_API_KEY=fake python -m pytest tests/ -q
```

All existing tests must still pass.

## Do NOT

- Modify `src/parser.py`, `src/splitter.py`, `src/merger.py`, `src/chapter_splitter.py`,
  `src/models.py`, `src/profiler.py`, or `src/llm/**`.
- Restructure `translate_chunk`'s retry/heal separation. The tenacity retry must stay
  outside the heal loop so a hard API failure doesn't re-run the full backoff schedule per
  heal attempt.
- Add a language-detection dependency. Unicode range checks are sufficient and have no
  install cost.
- Commit or push.
