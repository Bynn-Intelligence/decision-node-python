# Changelog

All notable changes to this package are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.3.0] - Unreleased

Implements version 1.3 of the SDK specification (`SPEC_VERSION` is `"1.3"`).

### Added

- Boosters: `boosters` on `decide`, `check`, `classify`, `score`, `number`, `gate`,
  `decide_many` requests, batch requests and `check_request`, as booster names,
  `Booster(name, images, require, options)` objects (sent as `{"name": ...}` with each
  setting only when given) or wire-form mappings, sent in the order given. Left out or empty, the request has no `boosters` field. The client checks
  nothing about boosters itself; the API answers. Sessions take no boosters.
- `Decision.evidence`: booster name to `Evidence` (`version`, `status`, `latency_ms`,
  `results`), with `AgeReading`, `AiGeneratedReading` (its `TopGenerator`, `None` unless
  the reading says likely generated, `c2pa_signer`, and `c2pa`, `untrusted` for Content
  Credentials that validate but are not trusted) and `UnknownReading` for a reading of a
  booster this version does not know or with a field missing or of the wrong type. `None`
  when the request named no boosters.
- `Decision.usage.booster_tokens` and `Decision.usage.boosters` (`None` when absent).
- `DecisionNodeError.boosters`: the `x-decisionnode-boosters` header as booster name to
  `BoosterStatus` (`status`, `latency_ms`), on every error; empty when the header is absent.
- Fine-tuned models: `ModelInfo.fine_tuned`, `base`, `adapter_version` and `status`
  (`None` on the catalogue's own entries). A fine-tuned model's name is sent as given, and
  `Decision.model` is the pinned name that answered.
- `RequestTooLargeError` for the API's 25 MiB request body limit: raised for every 413,
  never retried, and before anything is sent or uploaded when a request, a `decide_many`
  request or a batch line is over 26,214,400 bytes as it will be sent (images to upload
  counted by their `upload_id`). `check_request` measures it too.

### Changed

- `PermissionDeniedError` is documented as every 403 that is not a refusal, a booster
  turned off for the workspace included; `OverloadedError` as also raised, and retried,
  for a required booster that failed (`booster_unavailable`).

## [0.2.0] - Unreleased

Implements version 1.2 of the SDK specification (`SPEC_VERSION` is `"1.2"`) and passes
conformance suite 1.10.

### Added

- Videos sent as frames: `Video(id, frames, fps, start, end)` and `VideoFrame(t, data,
  media_type)`, taken by `decide`, `check`, `classify`, `score`, `number`,
  `gate`, `decide_many` requests, batch requests and `check_request` as `videos`. A frame's
  data is bytes (its type read from the first bytes) or base64 text with its media type.
  Frames always go inline.
- Before anything is sent: ids unique across images and videos, frame times of at least 0
  and strictly increasing, JPEG, PNG or WebP frames, `fps` above 0 and at most 4, and
  `end` after `start`, each raising `BadRequestError` with the API's message. More than 4
  videos raises the API's `ValidationError` (a `too_long` entry on the videos). Frame
  counts are the API's.
- `Decision.usage.video_frames`, `Decision.usage.video_seconds` (`None` when absent) and
  `Decision.warnings` (an empty list when absent).
- The `point` and `box` question types: the `point(instructions, image, units)` and
  `box(instructions, image, units)` builders, and `PointAnswer` (`x`, `y`) and `BoxAnswer`
  (`x1`, `y1`, `x2`, `y2`), each with `present`, `confidence`, `image`, `width` and
  `height`. A request with one needs an image, and `image` names one of the request's
  images (required with two or more); with `units` these are checked before anything is
  sent and raise `ValidationError` with the API's entries.

### Removed

- `systemone()`. `POST /v1/systemone` is not part of the API; every decision is sent to
  `POST /v1/decide`.

### Changed

- A request holds up to 16 images (4 before); a 17th raises the API's `ValidationError`
  before anything is sent: a `too_long` entry on the images, listed with the request's
  other entries, a local image in its `input` without its bytes. Up to 16 images are
  uploaded through one presign; `upload()` of more raises `BadRequestError`
  (`files: 1 to 16 files, got 17.`) before anything is sent.

## [0.1.0] - Unreleased

First version. Implements version 1.1 of the specification shared by every official
DecisionNode SDK, exported as `SPEC_VERSION`, and passes conformance suite 1.6.

### Added

- The `number` question type: the `number` builder, the `number` helper and
  `NumberAnswer` (`number`, `expected`, `confidence`, `probabilities` keyed by the value as
  the API writes it).
- `systemone` for `POST /v1/systemone`, the strict path.
- Truth answers read as `TruthAnswer` under either wire name the API uses for the type.
- Images by `upload_id`, and `upload(images)`. Local images whose bytes total
  `DEFAULT_INLINE_THRESHOLD` (512 KiB) or more in a request are uploaded first (one
  presign, parallel `PUT`s with exactly the signed headers) and sent by `upload_id`;
  smaller ones go inline. The client option `inline_threshold` changes the threshold.
  Uploads are remembered by the SHA-256 of their bytes and reused until 5 minutes before
  they expire; a stale one is uploaded again and the request sent once more.
- Every fixed rule the API documents is checked before any request, in the API's order,
  raising its own error class, `error_type`, message and 422 entries: the 128-level
  nesting limit; question types and shapes, option, level and truth rules and the number
  grid (`number` is refused on `systemone`); the state's type; image count, ids, format
  by magic bytes, size, pixel dimensions read from the JPEG, PNG and WebP headers and an
  upload reference's media type; image URLs (`https`, at most 2,048 characters, no user
  information, port 443, no literal private address); session `window` and
  `ttl_seconds`; frames refused by `Session.send` with nothing written and no `seq` used;
  batch `custom_id` rules, an empty add and `batches.list` paging. Limits a model
  publishes are checked from the values held after a `models()` call, found by the
  model's name or by the alias or pinned version an entry states.
- `check_request(method, **arguments)` and `Session.check_frame(...)` run every local
  check a call would run, raise the same error, and send and upload nothing.
- `upload_timeout` (default 60 seconds): each storage `PUT` attempt of an image has its
  own deadline, apart from `timeout`.
- A session's stream goes through the HTTP client's proxy and uses its TLS settings.
- Batch results are read from the network line by line, the whole file.
- A storage `PUT` answered `403` after its URL's lifetime is presigned again once.
- Sessions: `open_session`, `Session` and `AsyncSession` (`connect`, `send`, `replies`,
  `next_event`, `end`, `close`), `Reply`, `FrameError`, `SessionClosed`, and `supervise`.
  `SessionLimitError`, a `RateLimitError`, for a key at its open-session limit.
- The Batch API: `batches.create`, `add`, `finalize`, `get`, `list`, `results` and
  `cancel`, and `run_batch`; `Batch`, `BatchPage`, `BatchResult` and `BatchRun`.
- Connections kept open between calls (up to 100, 30 seconds idle), and an `http2`
  option with the `http2` extra.

- `DecisionNode` and `AsyncDecisionNode` clients with `decide`, `decide_many`, `check`,
  `classify`, `score`, `gate`, `models` and `decide_raw`. `check`, `classify`, `score`
  and `gate` take `images` and `model` like `decide`.
- `gate` with fixed wording, the same in every official SDK.
- `ModelInfo` for each entry of `models()`, and `DecisionResult` for each result of
  `decide_many`.
- Question builders `truth`, `choice` and `score`; wire-form mappings are accepted too.
- Typed answers: `TruthAnswer`, `ChoiceAnswer`, `ScoreAnswer`, and `UnknownAnswer` for
  answer types this version does not know.
- Images from a file path, bytes (or base64 text) or an `https` URL. The media type of a
  file, and of bytes given without one, is read from the first bytes (JPEG, PNG or WebP);
  a file that cannot be read raises `ValueError` when the `Image` is made.
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
  sent as JSON, an unreadable file, a bad `base_url`, `timeout` or `max_retries`) raise
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
  commit (suite 1.6).
