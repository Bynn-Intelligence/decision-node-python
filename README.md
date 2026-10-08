# DecisionNode Python SDK

The official Python client for the DecisionNode API: ask yes-or-no, multiple-choice,
scale and number questions about any text, JSON or image, and get calibrated
probabilities back. One request at a time, a live session answering a stream of frames,
or a batch of thousands at half the price. Sync and async clients, fully typed.

## Install

```sh
pip install bynn-decisionnode
```

or `uv add bynn-decisionnode`. Python 3.10 or later.

## Set your API key

```sh
export DECISIONNODE_API_KEY="your-api-key"
```

## Quickstart

```python
from decisionnode import DecisionNode, truth, choice, score

dn = DecisionNode()  # reads DECISIONNODE_API_KEY

d = dn.decide(
    state={"ticket": "I was charged twice and nobody has replied for 3 days."},
    questions={
        "route": choice("Where should this go?", {"billing": "money", "bug": "broken", "account": "login"}),
        "urgency": score("How urgent is this?", ["routine", "today", "urgent", "critical"]),
        "escalate": truth("Escalate to a human now?"),
    },
)

d["route"].choice  # "billing"
d["route"].probabilities  # {"billing": 0.87, "bug": 0.08, "account": 0.05}
d["urgency"].score  # 2.31
d["escalate"].truth  # 0.94
```

The same with asyncio:

```python
import asyncio
from decisionnode import AsyncDecisionNode, truth


async def main() -> None:
    async with AsyncDecisionNode() as dn:
        d = await dn.decide(state="Refund my order now!", questions={"angry": truth("Is the customer angry?")})
        print(d["angry"].truth)


asyncio.run(main())
```

## Questions

Four question types. Each builder returns the question in its wire form, and a plain
`dict` in that form is accepted anywhere a builder is.

| Builder | Asks | Answer |
|---|---|---|
| `truth(instructions, if_true=None, if_false=None)` | Is this true? | `TruthAnswer.truth`: the probability of yes, 0 to 1 |
| `choice(instructions, options)` | Which one? `options` is a dict of name to description, or a list of names | `ChoiceAnswer.choice`, `.confidence`, `.probabilities` (by option name) |
| `score(instructions, levels)` | Where on this scale? `levels` are ordered low to high | `ScoreAnswer.score` (expected level index), `.confidence`, `.probabilities` and `.legend` (by level index, as `int`) |
| `number(instructions, min, max, step)` | Which number? A value of the grid `min`, `min + step`, ... up to `max` (defaults 0, 255 and 1) | `NumberAnswer.number` (the most likely value), `.expected` (the weighted mean), `.confidence`, `.probabilities` (by the value as the API writes it: `"7"`, `"3.5"`) |

`state`, `instructions`, option descriptions and levels take a string or any
JSON-serialisable value. Question ids are any string and come back unchanged; read
answers by id, never by position.

```python
truth("Is this a refund request?", if_true="the customer asks for money back")
choice("Which language is this?", ["en", "de", "sv"])
score({"question": "How complete is the form?", "focus": "required fields"}, ["empty", "partial", "complete"])
number("How many cars are visible?", min=0, max=50)
number("Rating in halves?", min=0, max=5, step=0.5)
```

A number grid holds at most 4,096 values, with at most 6 decimals in `min` and `step`.
`number` leaves out of the request any of `min`, `max` and `step` you do not give, so the
API's defaults apply.

The values are the ones the API sent: probabilities are not rounded or renormalised,
so a list can sum to 0.99 or 1.01, and an entry can read `0.8200000000000001`. An
answer of a type this version does not know comes back as `UnknownAnswer` with its
fields in `.raw`, and so does an answer of a known type with a missing or wrongly typed
field; the rest of the decision parses as usual.

## One question at a time

```python
dn.check("Package arrived crushed.", "Is the customer reporting damage?").truth  # 0.97
dn.classify("Where is my parcel?", "Topic?", ["shipping", "billing", "returns"]).choice  # "shipping"
dn.score("Site is down for everyone!", "Severity?", ["low", "medium", "high"]).score  # 1.86
```

```python
dn.number("Drone frame over row B4.", "How many cars?", min=0, max=50).number  # 7
```

`check(state, instructions, if_true, if_false)`, `classify(state, instructions, options)`,
`score(state, instructions, levels)` and `number(state, instructions, min, max, step)`
send one question and return its answer. Like
`decide`, each of them and `gate` also take `images=` and `model=`. They promise a typed
answer, so when the server's answer to their question is missing or malformed they raise
`ServerError` (`status` 200, `reason` from `x-decisionnode-error`), where `decide`
would return the same answer as `UnknownAnswer`.

`gate(action, context)` judges an action before it runs. It sends the state
`{"action": ..., "context": ...}` and three questions, worded the same in every official
DecisionNode SDK, and returns the `Decision`:

| Id | Type | Question |
|---|---|---|
| `risk` | score | "How much harm could this action cause if it is wrong?", levels `none`, `low`, `moderate`, `high`, `severe` |
| `touches_production` | truth | "Does this action change a production system, production data or anything customers use?" |
| `verdict` | choice | "Should this action run as it is, wait for a person to confirm it, or be blocked?", options `allow` "safe to run without asking", `confirm` "a person should confirm it first", `block` "should not run" |

```python
g = dn.gate("DROP TABLE invoices;", {"environment": "production"})
if g["verdict"].choice != "allow":
    ask_a_person()
```

The thresholds are yours: the SDK never turns a probability into a decision.

## The decision

| Field | |
|---|---|
| `answers` | The answers by question id. `d["route"]` is `d.answers["route"]`. |
| `model` | The model version that answered. |
| `usage.input_tokens`, `usage.output_tokens` | Token counts, or `None` when the server sent none (never `0` in their place). |
| `usage.video_frames`, `usage.video_seconds` | For a request with videos: the frames the server read and the seconds of video they cover; `None` otherwise. |
| `warnings` | What the server changed when it read the request's videos (a lower rate or size), as sent; an empty list when there is nothing to say. |
| `request_id` | The request id. Quote it when you contact support. |
| `latency_ms` | The server's own time for the request, or `None` when the server sent none. |
| `safety` | What the safety check found (below), or `None`. |
| `raw` | The parsed response body. |

## Many requests

```python
from decisionnode import DecisionNodeError

results = dn.decide_many(
    [{"state": t, "questions": {"spam": truth("Is this spam?")}} for t in tickets],
    concurrency=8,
)
for r in results:
    if isinstance(r, DecisionNodeError):
        print("failed:", r)
    else:
        print(r["spam"].truth)
```

At most `concurrency` requests run at once (default 8; below 1 raises `ValueError`
before anything is sent). Every request is checked before the first is sent, so a bad
argument in any of them raises `ValueError`, a request the API would refuse raises its
error class, and nothing is sent. Results come back in input order, one `DecisionResult`
per request: a `Decision`, or the error that request ended with. Images with the same
bytes are uploaded once for all the requests. One failure does not stop
the others.

## The strict path

```python
d = dn.systemone(state="Refund my order now!", questions={"angry": truth("Is the customer angry?")})
```

`systemone(state, questions, images, model)` sends `POST /v1/systemone`, the strict form
of `decide`: it takes `truth`, `choice` and `score` questions, no other field, and always
a model (the client's when the call names none). Answers read as from `decide`.

## Images

```python
from decisionnode import Image

d = dn.decide(
    state={"claim": {"total": "740.00", "currency": "SEK"}},
    images=[Image("receipt", url="https://files.example.com/claims/4411/receipt.jpg")],
    questions={"total_matches": truth("Does the total on the receipt equal the total in the claim?")},
)
```

**Send images by URL when you can.** It is the fastest input: nothing is uploaded, the
request stays small, and the server fetches the image while it reads the request. A URL
must be `https` and at most 2,048 characters.

The other forms:

| Form | |
|---|---|
| `Image(id, url=...)` | An `https` URL the server fetches. The recommended form |
| `Image(id, path=...)` | A file, read when the `Image` is made |
| `Image(id, data=...)` | Bytes, or the same bytes as base64 text |
| `Image(id, upload_id=...)` | An image uploaded before (below), for an hour |

Local images (a path or data) are checked before anything is sent, without decoding them:
JPEG, PNG or WebP by their first bytes (a `media_type` you give must match them), at most
10 MB each, at most 40 megapixels, read from the image's header. A request holds at most 16
images, URLs included. A file that cannot be read raises `ValueError` when the `Image` is
made; an image the API would refuse raises `BadRequestError` before any request.

How local images travel is chosen for speed:

- When the local images of a request total under the inline threshold (512 KiB,
  `DEFAULT_INLINE_THRESHOLD`; set `inline_threshold` on the client to change it) they go
  inline in the request.
- At or above it they are uploaded first: one `POST /v1/uploads` for all of them, then one
  `PUT` each, in parallel, straight to storage, and the request names them by
  `upload_id`. The `PUT` carries only the headers the upload was signed with: never your
  key, never your custom headers.
- The client remembers what it uploaded, by the SHA-256 of the bytes, for as long as the
  upload lives (until 5 minutes before it expires, at most 256 images). The same image sent
  again, in another call, in `decide_many`, in a session frame or a batch line, goes by its
  `upload_id` whatever its size, without being uploaded again. When the server reports such
  an upload as expired or gone, the client uploads the image again and sends the request
  once more.

An `Image` can be reused across calls; its checks and hashes are computed once.

### Upload ahead

```python
upload_ids = dn.upload([Image("front", path="front.jpg"), Image("back", path="back.jpg")])
d = dn.decide(state="...", images=[Image("front", upload_id=upload_ids[0])], questions={...})
```

`upload(images)` uploads whatever the size and returns one `upload_id` per image. An
upload can be used for one hour by any key of your workspace.

## Videos

Video is available when the API enables it. A video is sent as the frames you sampled,
each with its time in seconds, and the questions see them as one video, in order and with
their times. Only frames are sent: no video file, no video URL and no sound go to the API.

```python
from decisionnode import VideoFrame, Video

frames = [VideoFrame(t, jpeg) for t, jpeg in camera_frames]  # (seconds, JPEG bytes) you already hold

d = dn.decide(
    state="Warehouse camera 4, loading dock, shift B.",
    videos=[Video("dock", frames, fps=2)],
    questions={
        "fall": truth("Does a person fall in the video?"),
        "forklifts": number("How many forklifts pass the door?", min=0, max=50),
    },
)
d["fall"].truth
d.usage.video_frames, d.warnings
```

| Argument | |
|---|---|
| `Video(id, frames, fps=None, start=None, end=None)` | `id` unique across the request's images and videos. `fps` caps the frames read a second (above 0, at most 4; the API reads 2 by default). `start` and `end` set the window read, on the frames' own times; `end` is after `start`. Left out, the API's defaults apply |
| `VideoFrame(t, data, media_type=None)` | `t` in seconds, at least 0, larger than the frame before it. `data` is JPEG, PNG or WebP bytes (the type is read from them) or base64 text with its `media_type` |

A request holds at most 4 videos. The ids, times, media types, `fps` and the window are
checked before anything is sent, and a fault raises `BadRequestError` as the API would. The
frame limits are the API's and are published with it. Frames always go inline: they are
never uploaded or resized. `decide`, `systemone`, the helpers, `decide_many` and batch
requests take `videos`; sessions do not.

To turn a video file into frames, use the `dn` command-line tool (`dn truth "Does a person
fall?" --video dock.mp4 --fps 2`); it runs FFmpeg on your machine and sends only the frames.
This package has no such dependency.

## Sessions

A session fixes the questions once, then answers every frame you send on a WebSocket,
newest first: a frame that arrives while another is being answered replaces any frame
still waiting, so you always get the answer to the latest reading, never a backlog.

```python
from decisionnode import DecisionNode, Reply, FrameError, SessionClosed, choice, truth

dn = DecisionNode()
with dn.open_session(
    instructions="You are the mission supervisor of a survey drone.",
    state={"mission": "Survey the north field"},
    questions={
        "mode": choice("Which flight mode now?", ["continue", "hold", "return", "land"]),
        "abort": truth("Should the mission be aborted now?"),
    },
    window=8,
) as session:
    session.connect()
    session.send({"battery": 0.31, "wind_mps": 11.4})
    for event in session.replies():
        if isinstance(event, Reply):
            print(event.seq, event["mode"].choice, event.dropped)
        elif isinstance(event, FrameError):
            print("frame", event.seq, event.error_type, event.message)
        else:  # SessionClosed, the last event
            print("closed", event.code, event.reopen)
```

| Call | |
|---|---|
| `open_session(state, questions, instructions, model, window, ttl_seconds)` | `POST /v1/sessions`. Returns a `Session` with every field of the answer (`session_id`, `model`, `window`, `ttl_seconds`, `expires_at`, `limits`, `questions`, `usage`, `answers` to the questions on the state alone, `safety`) |
| `Session.connect()` | Opens the stream. Your key goes in the handshake's `Authorization` header, never in the URL; your custom headers go too. The stream goes through the same proxy and uses the same certificates as the HTTP client |
| `Session.send(frame, image, questions, reset, seq)` | Sends one frame and returns its `seq`, counted from 1 unless you give one. Never waits for the answer. `image` is one `Image` (not a URL), sent inline or by `upload_id` as above |
| `Session.replies()` | Yields `Reply` and `FrameError` values as they come, then the `SessionClosed` |
| `Session.next_event(timeout)` | The next event, or `None` after `timeout` seconds |
| `Session.end()` | Ends the session (`DELETE`). A session that has already ended counts as ended |
| `Session.close()` | Closes the stream |

A `FrameError` is a value, not an exception: the session stays open. `SessionClosed` has
the close `code` and `reopen`, true when opening a new session with the same arguments is
the right next step (`1001`, `1006`, `1011`, `4404`, `4408`, `4409`, `4410`). A key that
already holds its open sessions gets `SessionLimitError` (a `RateLimitError`), which is
never retried. Pings are answered for you. The async client has `AsyncSession`, with the
same methods as coroutines and `async for event in session.replies()`.

### Supervise a feed

```python
def readings():
    while drone.flying:
        yield {"frame": drone.telemetry()}
        time.sleep(1)


last_close = dn.supervise({"questions": questions, "state": mission}, readings(), on_reply=print)
```

`supervise(body, feed, on_reply)` keeps one session answering a feed of readings. `body`
holds the `open_session` arguments, each reading the `send` arguments, and `on_reply` gets
every `Reply` and `FrameError`. When the session closes on a code that calls for it, a new
one is opened (at once after the idle and lifetime closes, else after a backoff of 1 to 30
seconds with jitter); on `1013` it connects to the same session again; on any other code
it stops and returns that `SessionClosed`. When the feed ends it waits for the replies
still owed, ends the session and returns the last `SessionClosed` seen, or `None`. With
the async client the feed may be an async iterator and `on_reply` a coroutine function.

## Batches

The Batch API answers many decide requests within 24 hours at half the price.

```python
requests = [
    {"custom_id": t.id, "state": t.text, "questions": {"route": choice("Where should this go?", ["billing", "bug"])}}
    for t in tickets
]
batch, results = dn.run_batch(requests, poll_interval=60)
for r in results:
    if r.decision is not None:
        print(r.custom_id, r.decision["route"].choice)
    else:
        print(r.custom_id, r.status, r.error)
```

`run_batch(requests, poll_interval, timeout)` creates a draft, adds the requests, queues
it, reads it every `poll_interval` seconds (at least 10; the server's `Retry-After` when
that is longer) until it ends, and returns the ended `Batch` with its results. Past
`timeout` (default 26 hours) it raises `TimeoutError`; the batch keeps running. If adding
the requests fails, the draft is cancelled and the error raised.

Each request is a `decide` request plus a `custom_id` (1 to 512 characters, unique in the
batch). Every request is checked before anything is sent; images follow the rules above,
per line.

| Call | |
|---|---|
| `batches.create(requests)` | A draft holding these requests (none makes an empty draft), sent as JSON Lines |
| `batches.add(batch_id, requests)` | Adds requests to a draft, in calls of at most 10 MiB, in order |
| `batches.finalize(batch_id)` | Queues the draft. Nothing runs or is billed before |
| `batches.get(batch_id)` | The `Batch`: `id`, `state`, `counters`, the times, `results_url`, `usage`, `error` |
| `batches.list(offset, limit)` | A `BatchPage`: `batches`, `offset`, `limit`, `total`, `next_offset` |
| `batches.results(batch_id)` | The `BatchResult` lines of an ended batch, in upload order, read from the network as you iterate |
| `batches.cancel(batch_id)` | Cancels the lines not started |

A `BatchResult` has `custom_id`, `status` (the HTTP status, or `"expired"` or
`"cancelled"`), `decision` for a `200` line, `error` for an error line (the exception the
live call would have raised, as a value), and `headers`.

`get`, `list`, `results` and `cancel` are retried like every call. `create`, `add` and
`finalize` change the batch, so they are retried only on `429` and `529`, which the API
answers before applying anything, and never after a timeout or a lost answer. A finalize
answered `409` is checked with `get`, and the batch is returned when an earlier finalize
went through.

## Models

```python
for m in dn.models():
    print(m.name, m.release_date, m.limits)
```

Each entry is a `ModelInfo`. `limits` holds every other field the model publishes: the
token limit, the number grid limits, the upload limits, and the `sessions` and `batch`
objects.

| Model | |
|---|---|
| `decisionnode-latest` | The main model. The default. |
| `decisionnode-flash-latest` | Faster and lower cost. |
| `decisionnode-1.0`, `decisionnode-1.0-flash` | Pinned versions. |

Choose per client (`DecisionNode(model="decisionnode-flash-latest")`) or per call
(`dn.decide(..., model="decisionnode-1.0")`). Each model publishes its limits in
`models()`.

## Errors

Every error derives from `DecisionNodeError`, with `status`, `error_type`, `request_id`
and `reason`: the server's explanation in words, from the `x-decisionnode-error` header,
for example `question 'q': a score takes up to 10 levels, got 11`. The error's own
message, `str(e)` (also `e.message`), is the server's `message` exactly (`""` when the
server sent none, never `None`), or the client's description when no response arrived.

| Status | Exception | Extra fields |
|---|---|---|
| 400 | `BadRequestError`; `ContextLimitError` when the request is over the model's token limit | |
| 401 | `AuthenticationError` (also raised before sending when no key is set) | |
| 402 | `InsufficientCreditError` | |
| 403 | `RefusalError`: refused under the usage policy | `category`, `safety` |
| 403, not a refusal | `PermissionDeniedError`: the key may not do this | |
| 404 | `NotFoundError` | |
| 422 | `ValidationError` | `entries` |
| 429 | `RateLimitError`; `SessionLimitError` for too many open sessions | `retry_after` (seconds) |
| 529 | `OverloadedError` | `retry_after` (seconds) |
| other 5xx, a 2xx whose body is not JSON, or a helper's own answer missing or malformed | `ServerError` | |
| 3xx and every other status | `DecisionNodeError` itself, with `status` | |
| no response | `ConnectionError`, `TimeoutError` | |

```python
from decisionnode import DecisionNodeError, ValidationError

try:
    dn.decide(state="...", questions={"q": score("Rate it", levels)})
except ValidationError as e:
    print(e.entries)
except DecisionNodeError as e:
    print(e.status, e.reason, e.request_id)
```

`ConnectionError` and `TimeoutError` here are the classes of this package, not
Python's built-ins of the same name.

A problem with an argument is found before anything is sent and raises Python's own
`ValueError`, never a `DecisionNodeError`: an empty question map, a value that cannot be
sent as JSON, a file that cannot be read, a `base_url` that is not `https`, a `timeout`
that is not above 0 or a negative `max_retries`.

A request the API would refuse by its fixed rules is refused before anything is sent
too, so the server never sees it. The error is the one the API would answer with (the
same class, `error_type`, message, and for a `ValidationError` the same `entries`), but
with no `status` and no `request_id`. The checks run in the API's order:

- more than 128 objects and arrays inside each other (`BadRequestError`, "Invalid request.");
- the shape of each question and of the state (`ValidationError`): a question that is not
  an object or has no `type`, values that must be a string, an object or an array, a
  choice or score without `criteria`, a score without levels, and every number grid rule;
- the question limits (`BadRequestError`): an unknown type ("Invalid request.", also a
  `number` question on `systemone`), a choice without options or with more than 255, a
  score with more than 10 levels, a truth question with neither instructions nor
  criteria, a number question without instructions;
- every image rule above, and for a URL: `https`, at most 2,048 characters, no user
  information, port 443, not a literal private address (`image_url_refused`);
- for sessions, `window` (1 to 64) and `ttl_seconds` (1 to 3,600) as `ValidationError`, and
  each frame in `Session.send`, which raises `BadRequestError` for a frame the server would
  refuse (an unknown question id, an empty frame, an image by URL...) without writing
  anything or using a `seq`;
- for batches, the `custom_id` rules, an add without requests and `batches.list` paging.

`decide_many` and batches check every request first. After a `models()` call the client
also checks the limits each model publishes there (a model that serves no sessions or
answers no number questions, its maximum window, batch size and `custom_id` length),
for a model named by its name, its alias or its pinned version. Limits counted in tokens
stay the server's.

### Check without sending

```python
dn.check_request("decide", state=ticket, questions=questions, images=images)
dn.check_request("open_session", questions=questions, window=8)
dn.check_request("batches.create", requests=requests)
session.check_frame({"battery": 0.31}, questions=["abort"])
```

`check_request(method, **arguments)` runs every check the call `method` (`decide`,
`systemone`, `open_session`, `batches.create` or `batches.add`) would run before sending,
with the same arguments, and raises the same error; it sends nothing, uploads nothing and
returns `None` when the request would go out. `Session.check_frame(...)` does the same for
a frame. Neither makes a request, so neither is a coroutine on the async client.

A failed storage upload raises `DecisionNodeError` with `error_type` `upload_failed`, the
storage's `status`, and its error code in `reason`. Storage `PUT`s are retried on 5xx and
dropped connections like any call.

Only a 2xx is a success. Redirects are never followed, also on an `http_client` you
pass in that is set to follow them: the client turns redirects off on every request it
sends, so the key is never sent to another URL or over plain HTTP.

## Your own HTTP client

`http_client` takes an `httpx.Client` (`httpx.AsyncClient` for the async client), for a
proxy, custom certificates or connection limits. The client checks it when it is made
and raises `ValueError` unless it can show that the key stays safe:

- It must be an `httpx.Client` (or `httpx.AsyncClient`) itself, not a subclass or
  another type: such a client could follow redirects whatever it is told.
- Its transports, including any in `mounts`, must be httpx's own `HTTPTransport`
  (`AsyncHTTPTransport`) or `MockTransport`. A transport of another type may follow
  redirects itself and is refused.
- For an `https` base URL, certificate and hostname checks must stay on: `verify=False`,
  or an `ssl.SSLContext` with `check_hostname` off or `verify_mode` other than
  `CERT_REQUIRED`, is refused. A custom CA (`verify=ssl.create_default_context(cafile=...)`)
  is fine.

```python
import httpx

dn = DecisionNode(http_client=httpx.Client(proxy="http://proxy.example.com:3128"))
```

The client does not close an `http_client` you pass in.

## Retries and timeouts

Each attempt times out after 10 seconds. The timeout covers the whole attempt, from
sending the request to the last byte of the response body, so a server that answers
slowly a little at a time cannot hold a call longer than that. A storage `PUT` of an
image has its own deadline for the whole attempt, `upload_timeout` (60 seconds by
default), since up to 10 MB may need longer than an answer does. Batch results are read
from the network line by line as you iterate them, each read bounded by the timeout, so a
file of any size is read whole. A failed request is retried twice on 408, 429,
any 5xx (529 included), a timeout or a dropped connection. The wait is `retry-after-ms`
when the server sends it, else `Retry-After` in seconds or as an HTTP date, capped at
60 seconds either way; without either, an exponential backoff from 0.5 seconds with
jitter, capped at 8 seconds. A 408 still failing after its retries raises
`DecisionNodeError` with `status` 408. 400, 401, 402, 403, 404 and
422 are never retried: the same request would fail again. A decision changes nothing on
the server, so a retry is always safe.

```python
dn = DecisionNode(timeout=5.0, max_retries=4)
```

## Safety

On the hosted API every decision carries the result of the safety check:

```python
d.safety  # Safety(checked=True, harm=0.031, self_harm=0.004, flagged=[], ...)
```

`flagged` lists the categories above your key's thresholds. When the check did not run,
`checked` is false and `reason` says why. A self-hosted server sends nothing, and
`safety` is `None`. A request the policy refuses raises `RefusalError` and is never
retried.

### The safety result in the body

The result always arrives in the `x-decisionnode-safety` response header. To also get it
in the response body, send the request header `x-decisionnode-safety-body: true`. Pass
it through `headers`, and it goes with every request of that client:

```python
dn = DecisionNode(headers={"x-decisionnode-safety-body": "true"})

d = dn.decide(state="...", questions={"q": truth("Is this spam?")})
d.raw["safety"]  # {"checked": True, "harm": 0.031, "self_harm": 0.004, "flagged": []}
```

`d.safety` is the same with or without the header; the header only adds the field to
`d.raw`.

## Configuration

| Argument | Environment variable | Default |
|---|---|---|
| `api_key` | `DECISIONNODE_API_KEY` | none: a call without a key fails before sending |
| `base_url` | `DECISIONNODE_BASE_URL` | `https://api.decisionnode.com` |
| `model` | `DECISIONNODE_MODEL` | `decisionnode-latest` |
| `timeout` | | `10.0` seconds for one whole attempt; must be above 0 |
| `max_retries` | | `2`; must be 0 or more |
| `headers` | | extra headers sent on every request |
| `http_client` | | an `httpx.Client` (`httpx.AsyncClient` for the async client), for proxies or custom certificates; see "Your own HTTP client" |
| `inline_threshold` | | `524288` bytes (`DEFAULT_INLINE_THRESHOLD`); local images of a request below it go inline, at or above it they are uploaded |
| `upload_timeout` | | `60.0` seconds for one storage `PUT` attempt of an image; must be above 0 |
| `http2` | | `False`; `True` uses HTTP/2 for the API connection (`pip install "bynn-decisionnode[http2]"`) |

An argument wins over the environment, and the environment over the default. An empty
string (`api_key=""`, also `model=""` on a call) counts as not given. A client
is safe to share between threads and tasks; close it with `dn.close()`
(`await dn.close()`) or use it as a context manager.

Create one client and reuse it: it keeps its connections open between calls (up to 100,
each kept for 30 seconds idle), so a call after the first needs no new connection or TLS
handshake, and it remembers its image uploads.

The base URL must be `https`, so the key never travels unencrypted. Plain `http` is
accepted only for `localhost`, `127.0.0.1` and `::1`, for a server on the same machine;
any other `http` URL, from the argument or the environment, raises `ValueError` when the
client is made.

## Raw requests

`decide_raw(body)` sends a request body exactly as given and returns the parsed
response body, for an API field this version of the SDK does not know yet:

```python
dn.decide_raw(
    {
        "model": "decisionnode-latest",
        "state": "",
        "questions": {"q": {"type": "truth", "instructions": "Is hotdog a sandwich?"}},
    }
)
```

## Logging

The SDK sends no telemetry; the only identifying header it adds is
`User-Agent: decisionnode-python/<version>`. It logs to the `decisionnode` logger at
debug level, with the method, path, status, request id and timing of each attempt,
never the state, questions, images or key (a storage URL is logged without its
signature):

```python
import logging

logging.getLogger("decisionnode").setLevel(logging.DEBUG)
```

## Development

```sh
uv sync
uv run pytest
```

The tests include the DecisionNode SDK conformance suite
(`Bynn-Intelligence/decision-node-conformance`), run with both the sync and the async
client. Set `DECISIONNODE_CONFORMANCE_DIR` to a checkout of it, or check it out at
`conformance/` in this repository; without it the conformance tests fail rather than
skip. The tests pass suite 1.7; CI checks the suite out at the commit pinned in
`CONFORMANCE_REF` in `.github/workflows/ci.yml`, with every action pinned to a commit SHA
and no checkout keeping its credentials.

```sh
DECISIONNODE_CONFORMANCE_DIR=/path/to/decision-node-conformance uv run pytest
```

This package implements version 1.2 of the specification shared by every official
DecisionNode SDK, exported as `decisionnode.SPEC_VERSION`.

## License

MIT. See [LICENSE](LICENSE).
