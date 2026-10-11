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
