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
