# Decisions

Choices the Claude-only / agent-centric brief left open, with the reason for
each. Decisions the brief itself made (models per role, two run modes,
unattended runs, the output folder, fidelity lint, parsing only the body) are
not repeated here.

## Phase 1 — provider swap

**Temperature is no longer sent.** The current model docs say Claude Opus 5.5
rejects `temperature` outright, and Claude Sonnet 5.5 and Claude Haiku 5.5
reject any non-default value with HTTP 400; thinking is always on. The
`temperature` argument stays in `BaseLLM.generate` / `VisionLLM` so call sites
and test fakes are unchanged, but `AnthropicProvider` never sends it.
Determinism now rests on the on-disk caches (`book_profile.json`,
`structure.json`, `figures.json`, the exchange's `llm/responses/`): every answer
is asked for once and reused.

**Every call streams.** A translation can exceed the SDK's ten-minute guard for
non-streaming requests (thinking + ~36k output tokens), and one code path is
simpler than two. Tests serve real SSE through `httpx2.MockTransport`.

**`httpx2`, not `httpx`, in the provider tests.** `anthropic` 1.x runs on
`httpx2` (the maintained fork, published by Pydantic); a transport from the
`httpx` package is rejected. MinerU stays on `httpx`. `httpx2` is declared in
the dev group because the tests import it directly.

**A truncated reply is a `RuntimeError` subclass (`TruncatedReplyError`).**
The brief requires raising, not which class. Thinking length varies between
attempts, so the same request can fit next time: retryable is right. A
refusal (`RefusalError`) is retryable for the same reason.

**Server-side refusal fallback on the Sonnet/Opus roles** (`fallbacks:
"default"`, beta `server-side-fallback-2026-07-01`), switchable with
`CLAUDE_REFUSAL_FALLBACK=off`. Haiku has no server-side fallback and the Batch
API rejects the parameter, so translation and classification never send it.

**max_tokens per role**: profile and structure 32,000, classify 4,000,
translate 96,000. Thinking counts toward it; translation replies reach ~36k
tokens of LaTeX. All overridable with `CLAUDE_MAX_TOKENS_<ROLE>`.

**`scripts/test_api.py` was rewritten, not deleted**: it is now an Anthropic
smoke test for API mode (it forces `LLM_BACKEND=anthropic` and refuses to run
without `ANTHROPIC_API_KEY`). No-API mode has nothing to smoke-test.

**The `translate` role refuses the exchange backend** with directions to the
kit and the `chunk-translator` agent, because the brief keeps translation out
of the exchange.

**`translate_book` with an explicit `llm`** still uses it for profiling too
(that is how the tests drive it); without one, profiling and translation each
get their own role's model.

## Phase 2 — exchange backend

**Pending exit code is 3.** No script used it (0 success, 1 failure, 2 is
argparse's usage error). Code 4 is reserved for "a remote job is still
running" (resumable parse, batch translation).

**`ExchangePending` is a plain `Exception`,** not `RuntimeError`/`ValueError`,
so the API-era retry loops (tenacity in the translator and figure classifier)
never retry it and the fail-fast paths never treat it as fatal.

**`prepare_kit.py` stops after the first stage that left requests** (profile,
then structure, then figures). Each stage's prompts depend on the decisions
before it: heading levels change the outline table, and structure decisions
change the text around every image that the classifier sees. Asking a later
stage's questions on provisional text would produce requests that are never
reused. A typical book takes four rounds (profile + headings together,
outline, figures, done), plus one per wave of structure follow-ups; this fits
the stage2-preparer's six-round limit.

**Both profile calls are asked in the same round** (the profile and the heading
classification are independent), and every structure follow-up is asked even
after one is pending. A follow-up whose question changes because an earlier
follow-up's answer was kept costs one more round; that is rare (it needs two
conflicting rows) and keeps API-mode behaviour unchanged.

**Nothing provisional is cached.** A profile or outline built while answers are
pending is not written to `book_profile.json` / `structure.json`, so the next
round asks the outstanding questions instead of reusing a placeholder.

**Images for classify requests are copied to `llm/images/<sha16>.<ext>`.**
`VisionLLM` receives bytes, not filenames, so the exchange stores a copy the
answering agent can `Read`; the request's `images` paths are relative to the
work dir, as the schema says.

**A malformed answer is permanent until its response file is deleted.** The
same id finds the same file on every run, so a bad answer cannot loop. To
re-ask, delete `llm/responses/<id>.txt`. `--reclassify` / `--force-*` discard
cached *decisions* only; an unchanged prompt still finds its old answer.

## Phase 3 — fidelity lint, sentinel, waivers, allowlist, `by: agent`

**Calibrated on real data.** Thresholds and normalisation were tuned on the 89
translated chunk pairs of the five books in `data/work/` (eee_1, eee_2,
eee_practice, physics_2_pipeline, physics_2_study_guide). First pass: 541
number findings alone. Final: 52 review findings over 89 chunks (blocks 17,
display count 8, numbers 23, tags 4), nearly all real drift or the model
"correcting" OCR (`u_{01}` -> `u_{o1}`, `/ψ` -> `∠ψ`), plus one silent sign
change the old lint could not see (`= \frac` -> `= - \frac`, eee_practice/012).
`BLOCK_SIMILARITY = 0.85`; formatting-only pairs score above it once
normalised.

**Normalisation beyond the brief's list**, each driven by a real false
positive: bold (`\mathbf`, `\boldsymbol`) is presentation (the parser drops
vector bold, the model restores it); `array` is an alignment wrapper (added or
removed around the same lines); a word of 3+ letters in a font command is
prose (`P_{损}` -> `P_{\mathrm{loss}}`); `\mathrm{cos}` is `\cos`; `\bar` is
`\overline`; footnote markers (`^{①}`, `\textcircled{1}`) are dropped (the
prompt tells the model to delete them); sentence punctuation closing a display
is dropped; empty `( )` left by removed CJK is dropped; thin spaces inside a
number join it (`1\,000`).

**Numbers: a value that moved between prose and math is not a change.** The
source often writes `I=-60 mA` or `3.4×10⁻³` as plain text and the
translation, correctly, as math. Only the *exact* value (sign and `^`
exponent marker included, superscripts and HTML `<sup>` read as exponents)
gets that pass, so `10^{-20}` -> `10^{-2}` and a flipped sign still fire.

**Numbers: absent on one side is review; a different count is info.** The
brief asks for a multiset. On real data, nearly every count-only difference
was `0`, `2` or `^2` one apart (a formula rewritten or split). Those are still
reported, as one info finding per chunk; a value missing from one side
entirely is review. Subscript numbers are marked `_` (`u_{02}` -> `_02`) so a
label never masks a value.

**Display count: a display that is inline math on the other side was moved,
not lost.** The model promotes `$F = qE$` to a display and back; that is
reported as info. Only unexplained losses/additions are review.

**Tags:** only displays can end in a written-out `(12.3)`; inline `$(12.3)$`
is a reference. A tag alone on its line (`(3.5.1)` -- the parser lost the
formula and kept its number) counts as a source tag.

**Known limitation, kept explicit:** a swapped subscript in a long display may
score above the threshold; the numbers check does not see it (labels are the
same multiset). Short displays fire.

**Sentinel only where the kit asks for it.** `truncated-sentinel` applies when
`kit/system_prompt.txt` contains `% END-OF-CHUNK`; kits built before it (all
five existing books) are not held to it. A fenced reply keeps the sentinel
after its closing fence, so both fence strippers look past it.

**Lint findings now carry `file` and a per-file line.** LaTeX checks run on
the concatenated book; `lint_kit` maps each line back to
`translated/NNN.tex:L`. Waivers match on `check`, `file` and a whitespace-
collapsed `line_excerpt` substring of the message or context (empty excerpt =
every finding of that check in that file). A waiver without a reason is
refused (review); a waiver that matched nothing is reported (info) so stale
ones are noticed. `Finding`/`Severity`/`LintReport` moved to `src/findings.py`
(re-exported by `src.lint`) so `src.fidelity` does not import `src.lint`.

**`by: agent`** (`src/provenance.py`): protected like `manual` in
structure.json, figures.json and book_profile.json (a top-level `by` field,
honoured by `--force-profile`). Decided rows are no longer flagged or counted
as "flagged" figures. "Manual beats agent": both are immune to the pipeline;
between the two, the skill forbids the agent from editing a `manual` entry,
and `provenance.stronger()` ranks them wherever two could compete.
