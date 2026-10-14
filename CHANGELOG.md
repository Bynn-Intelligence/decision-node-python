# Changelog

All notable changes to this package are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.3.3] - 2026-10-14

Implements version 1.3 of the SDK specification (`SPEC_VERSION` is `"1.3"`) and passes
conformance suite 1.20.

### Added

- Device keys, as `dn.keys`, for a Manage device keys key: `create` (a Decisions-only key
  and its secret, with an `Idempotency-Key`: the caller's `idempotency_key`, else a new one
  per call, the same on every attempt; retried also while the first attempt runs), `list`
  and `all` (by labels, `device_id` and `status`, a page of 1 to 200), `update`
  (only the fields given; `None` clears `labels`, `limits`, `daily_spend_cap` or
  `expires_at`), `revoke` and `revoke_by_labels` (with `device_ids` and `dry_run`). Each
  returns the API's object as sent. Checked before sending, as the API answers them: a
  key's name (1 to 48 characters once trimmed), `device_id`, labels (at most 16), the shape
  of `limits` and a `daily_spend_cap` from 0.01 to 1,000 in whole cents (each the API's 422
  entry, the first refused in the body's order), an update with nothing to change, a
  malformed `idempotency_key`, `list`'s filters and `limit`, and a bulk revoke without a
  label or with over 1,000 `device_ids`.
- Per-key usage, as `dn.usage`: `key` (`GET /v1/keys/{key_id}/usage`), `list`
  (`GET /v1/usage`, one page) and `all` (every page, `next_cursor` followed until it is
  `None`). Each returns the API's object as sent and retries as any read. The query's
  rules (`granularity`, `group_by`, labels, `device_id`, key ids and at most 100 of them,
  `scope`, `status`, `include_empty`, `limit`) are checked before sending with the API's
  messages.
- `FaceRecognitionReading.collection_name`, and on each `FaceMatch` its `collection_name`
  and the item's `metadata` (an object as sent), each `None` when absent, never `{}`. A
  `collection_name` that is not a text or a `metadata` that is not an object makes the
  reading an `UnknownReading`.
- An item's `metadata`: `enroll` takes it, an object of up to 20 keys (1 to 40 of `a-z`,
  `0-9` and `_`) whose values are texts of up to 500 characters, numbers or booleans,
  checked before sending as the API checks it.
- `collections.update_item(collection_id, item_id, reference, label, metadata)`: a `PATCH`
  sending only the fields given (`label=None` clears the label, `metadata` replaces the
  whole object), with an `Idempotency-Key` and the retries of `enroll`; with no field it
  raises the API's `ValidationError` before sending.
- `collections.list_items` takes `reference` or `reference_prefix` (not both), each checked
  as the API checks it.
- `DecisionNodeError.details`: the error body's `detail.details` when it is an object, else
  `None`. A 529 whose `details.reason` is `enrollment_unpriced` is never retried.
- `ModelInfo.price_per_million_input`, `price_per_million_output`,
  `price_per_million_image`, `price_per_million_video` and `batch_price_factor`: the
  entry's prices as sent, each `None` when it states none (a fine-tuned model states no
  `batch_price_factor`). `limits` and `raw` keep them too.

### Changed

- A 429 `key_spend_cap_reached` (a key past its daily spend cap) is raised at once and
  never retried: the cap resets at 00:00 UTC. It stays a `RateLimitError` with `details`
  `resets_at`, `cap` and `spent`.
- The README documents the key kinds, a publishable key's `403` `origin_not_allowed` and
  `model_not_allowed` (with `details["allowed"]`), and its `429`s with `details["reason"]`
  `client_rate` and `in_flight`, retried as every 429.

### Removed

- `Batch.price_factor`. The API does not send it: the batch price is each model's
  `batch_price_factor` in `models()` (none on a fine-tuned model), and each line's charge
  is in its usage.

## [0.3.2] - 2026-10-13

Implements version 1.3 of the SDK specification (`SPEC_VERSION` is `"1.3"`). First
published as 0.3.1 the same day.

### Added

- Several places in one answer: `point()` and `box()` take `max` (2 to 16), and the
  answer is a `PointsAnswer` or `BoxesAnswer` (`count`, `present`, `complete`, `points`
  of `PointPlace` or `boxes` of `BoxPlace`, `image`, `width`, `height`), told by its own
  list, in the order and numbers sent. `max` (a whole number from 2 to 16) is checked
  before sending; whether a model version answers several places is the API's to say.
- Boosters: `boosters` on `decide`, `check`, `classify`, `score`, `number`, `gate`,
  `decide_many` requests, batch requests and `check_request`, as booster names,
  `Booster(name, images, require, options)` objects (sent as `{"name": ...}` with each
  setting only when given, or as the bare name when none is) or wire-form mappings, sent
  in the order given. Left out or
  empty, the request has no `boosters` field. The client checks nothing about boosters
  itself; the API answers. Sessions take no boosters.
- `Decision.evidence`: booster name to `Evidence` (`version`, `status`, `latency_ms`,
  `billed_calls`, `results`), with `AgeReading`, `AiGeneratedReading` (its `TopGenerator`, `None` unless
  the reading says likely generated, `c2pa_signer`, and `c2pa`, `untrusted` for Content
  Credentials that validate but are not trusted) and `UnknownReading` for a reading of a
  booster this version does not know or with a field missing or of the wrong type. `None`
  when the request named no boosters.
- `Decision.usage.booster_tokens` and `Decision.usage.boosters` (`None` when absent).
- The readings of the `ocr`, `face_detection` and `face_recognition` boosters: `OcrReading`
  (`text`, `truncated`), `FaceDetectionReading` (`faces`, `detections` of `FaceDetection`
  with a `FaceBox` and a `confidence`) and `FaceRecognitionReading` (`faces`, `matches` of
  `FaceMatch` with a `FaceBox`, a `score`, the `collection` searched, the item's
  `reference`, and `name` or `label`, `None` when absent). Box corners are kept as
  received. A malformed reading is an `UnknownReading`.
- Collections of faces, as `dn.collections`: `create`, `list`, `get`, `delete`, `enroll`
  (one face from one image, sent as a decision's image is), `list_items` and
  `delete_item`, each returning the API's object as sent. `create` and `enroll` send an
  `Idempotency-Key`, a new UUID per call and the same on every attempt of it, so the API
  creates, enrolls and bills once; they retry like any call, and also while the API
  reports the first attempt still running (a 409 `idempotency_in_progress`, after its
  `Retry-After`). A 409 `idempotency_key_reused` is raised and never retried. A
  collection's type and name, an item's reference and label and the items list's `offset`
  and `limit` are checked before anything is sent; of the picture only a URL's rules are,
  every other fault is the API's 422 `image_unreadable`.
- A decision whose boosters name `ocr` waits 30 seconds per attempt unless a `timeout` was
  set on the client, and is never retried after a timeout, since the server finishes that
  attempt and may bill it; its other retries stay.
- `Usage.cost`: what the answer cost in USD, a `Cost` (`usd`, `model_usd`, `boosters_usd`,
  `free_tier`) read as sent, on decisions, batch result lines, a session's open and frame
  replies; `None` when absent, never zeros.
- `DecisionNodeError.boosters`: the `x-decisionnode-boosters` header as booster name to
  `BoosterStatus` (`status`, `latency_ms`), on every error; empty when the header is absent.
- Fine-tuned models: `ModelInfo.fine_tuned`, `base`, `adapter_version` and `status`
  (`None` on the catalogue's own entries). A fine-tuned model's name is sent as given, and
  `Decision.model` is the pinned name that answered.
- `RequestTooLargeError` for the API's 25 MiB request body limit: raised for every 413,
  never retried, and before anything is sent or uploaded when a request, a `decide_many`
  request or a batch line is over 26,214,400 bytes as it will be sent (images to upload
  counted by their `upload_id`). `check_request` measures it too.
- The fallback: `fallback` (`True` or `False`) on `decide`, the helpers, `decide_many`
  requests, batch requests and `check_request`, sent only when given. `Decision.fallback`
  is a `Fallback` (`served_model`, `requested_model`) on an answer from another model than
  the one asked for, read from `x-decisionnode-fallback` and
  `x-decisionnode-requested-model`; `None` otherwise.

### Changed

- The default model is `decisionnode-moe` (it was `decisionnode-latest`), and the docs use
  the family ids `decisionnode-moe` and `decisionnode-flash`, which always mean the current
  release; a pinned id is `decisionnode-<family>-<version>`, as `models()` lists it. The
  older ids are still accepted by the API and sent as given.
- `PermissionDeniedError` is documented as every 403 that is not a refusal, a booster
  turned off for the workspace included; `OverloadedError` as also raised, and retried,
  for a required booster that failed (`booster_unavailable`).
- An error whose body is not JSON (an HTML or text page, or no body) is classed by its
  status alone, with `error_type` `None`, `request_id` and `reason` from the headers and
  the client's own message, `HTTP <status> <text>` (`HTTP 413 Content Too Large`), with
  the request id when sent. The page is never read: `body` is `None`.
- An image id of more than 64 characters is refused before anything is sent:
  `BadRequestError` `api_usage_error`, "Image <n>: its id is longer than 64 characters."

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
