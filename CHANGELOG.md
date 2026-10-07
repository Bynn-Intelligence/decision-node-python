# Changelog

All notable changes to this package are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First version. Implements version 1.0 of the specification shared by every official
DecisionNode SDK, exported as `SPEC_VERSION`.

### Added

- `DecisionNode` and `AsyncDecisionNode` clients with `decide`, `decide_many`, `check`,
  `classify`, `score`, `gate`, `models` and `decide_raw`. `check`, `classify`, `score`
  and `gate` take `images` and `model` like `decide`.
- `gate` with fixed wording, the same in every official SDK.
- `ModelInfo` for each entry of `models()`, and `DecisionResult` for each result of
  `decide_many`.
- Question builders `truth`, `choice` and `score`; wire-form mappings are accepted too.
- Typed answers: `TruthAnswer`, `ChoiceAnswer`, `ScoreAnswer`, and `UnknownAnswer` for
  answer types this version does not know.
- Images from a file path, bytes or an `https` URL. The media type of a file, and of
  bytes given without one, is read from the first bytes (JPEG, PNG or WebP); another
  format, or a file that cannot be read, raises `ValueError` when the `Image` is made.
- An answer of a known type with a missing or wrongly typed field comes back as
  `UnknownAnswer` with its raw fields; the rest of the decision parses as usual.
- Retries on 408, 429, 5xx, timeouts and connection errors, honouring `retry-after-ms`
  and `Retry-After` (seconds or an HTTP date), capped at 60 seconds.
- A 10 second timeout that covers the whole attempt, from sending the request to the
  last byte of the response body. `timeout` must be above 0 and `max_retries` 0 or
  more.
- Exception classes for the documented statuses, all derived from `DecisionNodeError`,
  including `PermissionDeniedError` for a 403 that is not a refusal. A redirect and any
  other status raise `DecisionNodeError` itself with its `status`; a 2xx whose body is
  not JSON raises `ServerError`. `str(error)` is the server's message, `""` when it
  sent none.
- Argument problems found before sending (an empty question map, a value that cannot be
  sent as JSON, an unusable image, a bad `base_url`, `timeout` or `max_retries`) raise
  Python's own `ValueError`, never a `DecisionNodeError`. `decide_many` checks every
  request before sending any.
- Redirects are never followed, also on an `http_client` passed in that is set to
  follow them.
- The base URL must be `https`; plain `http` is accepted only for `localhost`,
  `127.0.0.1` and `::1`.
- `Decision.safety`, parsed from the `x-decisionnode-safety` response header. The
  README shows how to send `x-decisionnode-safety-body: true` to also get the result
  in the response body.
- Settings from `DECISIONNODE_API_KEY`, `DECISIONNODE_BASE_URL` and `DECISIONNODE_MODEL`;
  an empty argument counts as not given.
- `decide_many` refuses a `concurrency` below 1 with `ValueError` before sending.
- `check`, `classify` and `score` raise `ServerError` (`status` 200, `reason` from
  `x-decisionnode-error`) when the answer to their question is missing or malformed;
  `decide` returns such an answer as `UnknownAnswer`.
- A caller's `http_client` is checked when the client is made: it must be an
  `httpx.Client` (`httpx.AsyncClient`) itself, with httpx's own transports, or it is
  refused with `ValueError`, since another type may follow redirects. For an `https`
  base URL it may not turn off certificate or hostname checks.
- `DecisionNodeError` is exported by that name and is the base of every exception.
- CI pins every action to a commit SHA, sets `persist-credentials: false` on every
  checkout, installs a pinned uv and checks out the conformance suite at a pinned
  commit (suite 1.3).
