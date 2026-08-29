# Task: add a DeepSeek provider and make it the default

## Context

This repo translates scanned Chinese textbooks to English. It currently has exactly one
LLM provider (Gemini). Gemini's free tier caps at 20 requests/day, which makes a 500-page
book take ~5 days. DeepSeek has effectively no rate limit and costs ~$0.60–1.80/book, so
we're switching.

The abstraction for this already exists — **use it, don't rebuild it**:

- `src/llm/base.py` — `BaseLLM` ABC: `generate(system_prompt, user_text, temperature) -> str`
- `src/llm/factory.py` — `get_llm()` dispatches on `config.LLM_PROVIDER`
- `src/llm/providers/gemini.py` — the existing provider; **read it first and mirror its
  structure, error handling, and docstring style**
- `src/config.py` — already declares unused `DEEPSEEK_API_KEY` and `DEEPSEEK_MODEL`
  constants. Use them; don't invent new names.

Nothing outside `src/llm/`, `src/config.py`, and `scripts/test_api.py` should change.
`translator.py`, `profiler.py`, and `latex.py` all depend only on `BaseLLM` and must not
be touched.

---

## API facts (verified against https://api-docs.deepseek.com/ — do not guess these)

- **OpenAI-compatible.** Use the `openai` SDK with `base_url="https://api.deepseek.com"`.
  Endpoint is `/chat/completions`; the SDK appends that.
- **Models:** `deepseek-v4-flash`, `deepseek-v4-pro`. Default to `deepseek-v4-pro`.
- **Thinking mode is ENABLED BY DEFAULT at `high` effort.** Two reasons we must disable it:
  1. Chain-of-thought tokens bill as output tokens — our most expensive line item, on a
     task that needs no reasoning (the model is transcribing and translating, not solving).
  2. **Thinking mode silently ignores `temperature`.** The docs state that `temperature`,
     `top_p`, `presence_penalty`, and `frequency_penalty` "will not trigger an error but
     will also have no effect" in thinking mode. Our pipeline passes `temperature=0.3` for
     translation and `0.2` for profiling and depends on that determinism. Leaving thinking
     on would silently break it with no error.

  Disable it by passing, via the OpenAI SDK:
  ```python
  extra_body={"thinking": {"type": "disabled"}}
  ```
  Do **not** pass `reasoning_effort` when thinking is disabled.
- **Response shape:** `response.choices[0].message.content`. (With thinking enabled there's
  also `reasoning_content`; we won't have it, but guard against `content` being `None`.)
- **Context caching is automatic** — disk-based prefix caching, no headers or flags. Our
  system prompt (profile + glossary) is a byte-identical prefix on every call, so it
  caches for free. Nothing to implement; just don't do anything that would vary the prefix.
- **Error codes:**
  | Code | Meaning | Retryable? |
  |---|---|---|
  | 400 | Invalid request format | No |
  | 401 | Bad API key | No |
  | 402 | Insufficient balance | No |
  | 422 | Invalid parameters | No |
  | 429 | Rate limit | **Yes** |
  | 500 | Server error | **Yes** |
  | 503 | Server overloaded | **Yes** |

---

## What to build

### 1. `src/llm/providers/deepseek.py`

A `DeepSeekProvider(BaseLLM)` mirroring `GeminiProvider`:

- Construct the `OpenAI` client once in `__init__` (not per call), reading
  `config.DEEPSEEK_API_KEY` and `config.DEEPSEEK_MODEL`.
- `generate()` sends `system_prompt` as a `{"role": "system"}` message and `user_text` as
  `{"role": "user"}`, passes `temperature`, and disables thinking as above.
- **Error contract — this matters.** `src/translator.py` retries on `RuntimeError` via
  tenacity and treats anything else as fatal. So:
  - Wrap **retryable** failures (429, 500, 503, connection/timeout errors) in
    `RuntimeError` so tenacity's backoff handles them.
  - Raise **non-retryable** failures (400, 401, 402, 422) as something that is *not* a
    `RuntimeError` — a `ValueError` is fine — with a clear message. Retrying a bad API key
    or an empty balance 5 times with exponential backoff wastes minutes and tells the user
    nothing. **402 in particular should fail fast and say the account is out of credit**,
    since that's the one people hit mid-run.
  - Read the status code off the `openai` exception types rather than string-matching the
    message.
- Treat an empty or `None` `content` as a `RuntimeError` (retryable) — that's a transient
  server-side oddity, not a config error.

### 2. `src/llm/factory.py`

Add a `deepseek` branch. Keep the existing `gemini` branch working — don't remove it.
Preserve the existing `ValueError` for unknown providers.

### 3. `src/config.py`

- Default `LLM_PROVIDER` to `"deepseek"`.
- Default `DEEPSEEK_MODEL` to `"deepseek-v4-pro"`.
- **Make credential validation conditional on the selected provider.** Right now
  `config.py` raises at import time if `GEMINI_API_KEY` is missing. After this change, a
  DeepSeek user with no Gemini key must not be blocked — validate only the key belonging to
  the active `LLM_PROVIDER`. This is the change most likely to break existing behaviour, so
  be careful: `main.py` and `scripts/test_api.py` both rely on config importing cleanly.
- Keep `.env` loading behaviour unchanged.

### 4. `pyproject.toml`

Add `openai>=1.0.0` to `dependencies`. This is the one file outside `src/` you may edit.

### 5. `scripts/test_api.py`

It currently prints Gemini-specific hints. Make stage 1 and stage 2 provider-aware:
report the active provider and model, and give remediation hints matching that provider
(e.g. for 402, point at the DeepSeek top-up page). Don't restructure the five stages.

---

## Tests

Add `tests/test_deepseek_provider.py` in the existing style. Mock the `openai` client —
**no real API calls in the test suite.** Cover:

- `generate()` sends system and user messages in the right roles, with the configured model
- `thinking` is disabled in the request, and `reasoning_effort` is not sent
- `temperature` is forwarded
- 429 / 500 / 503 raise `RuntimeError` (so tenacity retries them)
- 401 / 402 / 400 / 422 raise a non-`RuntimeError` (so they fail fast), and the 402 message
  mentions balance/credit
- empty or `None` content raises `RuntimeError`
- `factory.get_llm()` returns `DeepSeekProvider` when `LLM_PROVIDER=deepseek` and
  `GeminiProvider` when `gemini`
- config import succeeds with only `DEEPSEEK_API_KEY` set (no Gemini key present)

## Verify before finishing

```bash
DEEPSEEK_API_KEY=fake python -m pytest tests/ -q
```

All existing tests must still pass alongside the new ones. Also confirm config imports
cleanly with only a DeepSeek key:

```bash
env -u GEMINI_API_KEY DEEPSEEK_API_KEY=fake python -c "import src.config, src.llm.factory; print('ok')"
```

## Do NOT

- Modify `src/translator.py`, `src/profiler.py`, `src/latex.py`, `src/parser.py`,
  `src/splitter.py`, `src/merger.py`, `src/chapter_splitter.py`, or `src/models.py`.
- Remove or break the Gemini provider.
- Implement streaming, tool calls, vision, or the Anthropic-format endpoint. We need one
  synchronous text call.
- Add a retry loop inside the provider — `src/translator.py` already owns retry via
  tenacity, and a second layer would compound the backoff.
- Commit or push. I'll review first.
