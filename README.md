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

Six question types. Each builder returns the question in its wire form, and a plain
`dict` in that form is accepted anywhere a builder is.

| Builder | Asks | Answer |
|---|---|---|
| `truth(instructions, if_true=None, if_false=None)` | Is this true? | `TruthAnswer.truth`: the probability of yes, 0 to 1 |
| `choice(instructions, options)` | Which one? `options` is a dict of name to description, or a list of names | `ChoiceAnswer.choice`, `.confidence`, `.probabilities` (by option name) |
| `score(instructions, levels)` | Where on this scale? `levels` are ordered low to high | `ScoreAnswer.score` (expected level index), `.confidence`, `.probabilities` and `.legend` (by level index, as `int`) |
| `number(instructions, min, max, step)` | Which number? A value of the grid `min`, `min + step`, ... up to `max` (defaults 0, 255 and 1) | `NumberAnswer.number` (the most likely value), `.expected` (the weighted mean), `.confidence`, `.probabilities` (by the value as the API writes it: `"7"`, `"3.5"`) |
| `point(instructions, image=None, units=None, max=None)` | Where in the image? (below) | `PointAnswer.x`, `.y`, `.present`, `.confidence`, `.image`, `.width`, `.height`; with `max`, a `PointsAnswer` |
| `box(instructions, image=None, units=None, max=None)` | Draw a box around it (below) | `BoxAnswer.x1`, `.y1`, `.x2`, `.y2`, `.present`, `.confidence`, `.image`, `.width`, `.height`; with `max`, a `BoxesAnswer` |

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
| `model` | The model version that answered; for a fine-tuned model its pinned name (below). |
| `usage.input_tokens`, `usage.output_tokens` | Token counts, or `None` when the server sent none (never `0` in their place). |
| `usage.video_frames`, `usage.video_seconds` | For a request with videos: the frames the server read and the seconds of video they cover; `None` otherwise. |
| `usage.booster_tokens`, `usage.boosters` | For a request with boosters: the share of `input_tokens` the boosters' readings take, and the booster calls billed; `None` otherwise. |
| `usage.cost` | What the answer cost in USD, as a `Cost`: `usd`, `model_usd`, `boosters_usd` and `free_tier`, read as the server sent them and never summed by the SDK; `None` when the server sent none. A session's open and each frame reply carry their own. |
| `evidence` | For a request with boosters: what each booster read (below); `None` otherwise. |
| `fallback` | On a fallback answer, the model that answered and the model asked for (below); `None` otherwise. |
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
10 MB each, at most 40 megapixels, read from the image's header. A file that cannot be read
raises `ValueError` when the `Image` is made; an image the API would refuse raises
`BadRequestError` before any request. A request holds at most 16 images, URLs included; a
17th raises `ValidationError` before any request, as the API answers it (a `too_long`
entry on the images).

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

A request body may be at most 25 MiB (26,214,400 bytes). Inline images count toward it;
uploaded images and URLs do not. The client measures each request as it will be sent,
`decide_many` requests and batch lines included, and raises `RequestTooLargeError` before
anything is sent or uploaded when it is over. Images given as a path or bytes above the
inline threshold are uploaded, so only a large state or base64 data you send yourself can
reach the limit.

An `Image` can be reused across calls; its checks and hashes are computed once.

### Upload ahead

```python
upload_ids = dn.upload([Image("front", path="front.jpg"), Image("back", path="back.jpg")])
d = dn.decide(state="...", images=[Image("front", upload_id=upload_ids[0])], questions={...})
```

`upload(images)` uploads 1 to 16 images whatever their size and returns one `upload_id` per image. An
upload can be used for one hour by any key of your workspace.

### Referring to images and videos

There are two ways to say which image a question is about:

1. **By id (recommended).** Give each image a short, meaningful id and use it in the
   instructions, in backticks: "Does the total on `receipt` match the amount in the claim?"
2. **By number.** Images are numbered from 1 in the order of `images`: "image 2", "the
   second image", "Picture 2" and "the last image" all work.

The model sees each image as `Picture 1 (receipt):`, `Picture 2 (claim_photo):`, so both
ways point to the same image. Prefer ids: they stay right when images are added or
reordered. Use names, not digits, as ids (`front`, not `2`). Videos work the same way, by id
or by number ("video 1"), numbered separately in the order of `videos`. The `image` of a
point or box question takes the id.

```python
d = dn.decide(
    state="Expense claim 4411: 740.00 EUR.",
    images=[
        Image("receipt", url="https://files.example.com/claims/4411/receipt.jpg"),
        Image("claim_photo", url="https://files.example.com/claims/4411/photo.jpg"),
    ],
    questions={"matches": truth("Does the total on `receipt` match the amount in the claim?")},
)
```

## Points and boxes

A `point` question finds one spot in an image; a `box` question draws a rectangle around
something. They mix freely with the other types in one request, and several of them on
the same image share one read of it.

```python
from decisionnode import Image, box, point, truth

d = dn.decide(
    images=[Image("belt", url="https://files.example.com/line-3/cam-2/0917.jpg")],
    questions={
        "pick": point("Point to the yellow package closest to the robot arm."),
        "damage": box("Draw a box around the damaged package.", image="belt"),
        "any_damage": truth("Is any package on the belt damaged?"),
    },
)

pick = d["pick"]
if pick.present > 0.8 and pick.confidence > 0.7:
    robot.pick(pick.x, pick.y)
else:
    send_to_operator()
```

| Field | |
|---|---|
| `x`, `y` | The point, from the top-left corner of the image: `x` to the right, `y` down. Whole pixels by default; with `units="normalized"`, fractions of the width and height with 4 decimals |
| `x1`, `y1`, `x2`, `y2` | The box: its top-left corner (`x1`, `y1`) and bottom-right corner (`x2`, `y2`), in the same units |
| `present` | The probability that what you asked for is in the image at all. Under a low value the place is only the best guess |
| `confidence` | The probability that the place is right, if the thing is there: for a point, that it lies on the thing; for a box, that it overlaps the true box by at least half (intersection over union of 0.5 or more) |
| `image`, `width`, `height` | The image the answer refers to and its size in pixels, so you can draw or convert the answer without another lookup |

Every field is always there. When several things match, the answer is the best match;
say which one you mean in the instructions ("the leftmost yellow package"), or ask for
several places.

### Several places

With `max`, a point or box question answers up to `max` places, one per thing the
instructions describe: every car in the lot, each person on the platform.

```python
d = dn.decide(
    images=[Image("lot", url="https://files.example.com/lot/0001.jpg")],
    questions={
        "cars": box("Every car in the lot.", max=8),
        "people": point("Each person standing on the platform.", max=6),
    },
)

cars = d["cars"]  # a BoxesAnswer
cars.count  # 5
for b in cars.boxes:
    draw(b.x1, b.y1, b.x2, b.y2, b.confidence)
if cars.complete < 0.5:
    ...  # more cars than 8 may be in the picture: raise max or narrow the instructions
```

| Field | |
|---|---|
| `count` | How many places follow, 0 to `max` |
| `present` | The probability that at least one such thing is in the image |
| `complete` | The probability that nothing was left out: that the true number is not above `max`. Low when the picture holds more than you asked for |
| `points` or `boxes` | The places (`PointPlace` with `x`, `y`, `confidence`; `BoxPlace` with `x1`, `y1`, `x2`, `y2`, `confidence`), left to right by the centre, ties from the top; an empty list when none is found |
| `image`, `width`, `height` | As for one place |

`max` is a whole number from 2 to 16, checked before anything is sent. A model version
that does not answer several places says so in `models()`; once the client holds that
list, a question with `max` for such a version raises `ValidationError` before anything
is sent. `decide`, `decide_many` and batch requests take `max`.

A point or box question needs an image in the request. With two or more images, name the
one to look in with `image`, the image's id; a reference by number ("Picture 2") goes in
the instructions. These rules and `units` (`"pixels"` or `"normalized"`) are
checked before anything is sent, and a fault raises `ValidationError` as the API would.
`decide`, `decide_many` and batch requests take point and box questions.

## Videos

A video is sent as the frames you sampled, each with its time in seconds, and the
questions see them as one video, in order and with their times. Only frames are sent: no video file, no video URL and no sound go to the API.

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

The ids, times, media types, `fps` and the window are checked before anything is sent, and
a fault raises `BadRequestError` as the API would. A request holds at most 4 videos; a fifth
raises `ValidationError` (a `too_long` entry on the videos). The
frame limits are the API's and are published with it. Frames always go inline: they are
never uploaded or resized. `decide`, the helpers, `decide_many` and batch requests take
`videos`; sessions do not.

To turn a video file into frames, use the `dn` command-line tool (`dn truth "Does a person
fall?" --video dock.mp4 --fps 2`); it runs FFmpeg on your machine and sends only the frames.
This package has no such dependency.

## Boosters

A booster is a specialist model that reads the request's images before the decision. Its
reading is added to what the model sees, and comes back in `evidence`. The model still
answers every question: a booster never answers one, and a request without boosters is
answered exactly as before.

| Booster | Reads | Tells the model |
|---|---|---|
| `age` | A picture of a person | How old the person is, with a range and a confidence (the largest face is read) |
| `ai_generated` | Any picture | How likely the picture was made by an image generator, which generator most likely, and whether the file carries Content Credentials (C2PA) and who signed them |

```python
from decisionnode import Image, truth

d = dn.decide(
    images=[
        Image("a1b2", url="https://files.example.com/a1b2.jpg"),
        Image("c3d4", url="https://files.example.com/c3d4.jpg"),
    ],
    boosters=["age", "ai_generated"],
    questions={
        "adult": truth("Is the person in the first picture an adult?"),
        "synthetic": truth("Is the second picture AI-generated?"),
    },
)

d["adult"].truth  # 0.97
age = d.evidence["age"]
age.status, age.version  # "ok", "age-v3"
age.results[0].age_years, age.results[0].low, age.results[0].high  # 27, 22, 33
d.evidence["ai_generated"].results[1].generated  # 0.93
d.usage.booster_tokens, d.usage.boosters  # 72, 4
```

`boosters` is a list of booster names, in the order you want them reported. `decide`, the
helpers, `decide_many` requests and batch requests take it. Sessions do not.

A booster that needs a setting is a `Booster`. Here `ai_generated` reads only the second
picture, and the request fails if it cannot:

```python
from decisionnode import Booster

d = dn.decide(
    images=images,
    boosters=["age", Booster("ai_generated", images=["c3d4"], require=True)],
    questions=questions,
)
```

| Argument | |
|---|---|
| `Booster(name, images=None, require=None, options=None)` | `images`: the ids of the request's images the booster reads; left out, every image. `require=True`: a booster that fails or does not answer in time fails the request with `OverloadedError` (`error_type` `booster_unavailable`), which is retried like every 529; left out, the decision runs without that reading and `evidence` says why. `options`: settings of a booster, for boosters that have any |

It is sent as `{"name": "ai_generated", "images": ["c3d4"], "require": true}`, each setting
only when given; a plain mapping in that form works too. A `Booster` with no setting is sent
as its name, so the request stays the plain list.

The client checks nothing about boosters itself: the API answers a malformed list, an
unknown name and a booster turned off for your workspace before any booster runs and
before any charge. A booster added to the API works with this version of the package.

A request whose boosters name `ocr` waits 30 seconds per attempt by default; a `timeout`
you set on the client is used as given. It is never retried after a timeout, because the
server finishes that attempt and may bill it: set a timeout of at least 30 seconds for such
requests.

`d.evidence` maps each booster the request named, in its order, to an `Evidence`:

| Field | |
|---|---|
| `version` | The version of the booster that read, such as `age-v3` |
| `status` | `ok`; `timeout` (no answer in time); `error`; `skipped` (none of the request's images is one it reads) |
| `latency_ms` | How long the booster took |
| `billed_calls` | The calls of this booster billed on this request |
| `results` | One reading per image the booster was to read, each naming its `image`. A picture it did not read (it timed out or failed) or could not decode (`readable` false) is an `UnknownReading` holding those fields, and `status` says why |

| Reading | Fields |
|---|---|
| `AgeReading` | `image`, `faces` (how many faces were found; the largest is read), and `age_years`, `low`, `high`, `confidence`, each `None` when `faces` is 0 |
| `AiGeneratedReading` | `image`, `generated` (the probability that a generator made the picture), `confidence`, `method` (`inference`: read from the pixels; `c2pa`: the file's trusted Content Credentials declare it generated or edited by AI), `top_generator` (`name`, `probability` of the likeliest generator when the reading says likely generated; `None` otherwise, and always on a `c2pa` reading), `c2pa_signer` (who signed the file's Content Credentials when they are trusted, else `None`), `c2pa` (`untrusted` when the file carries Content Credentials that validate but are not trusted, which change nothing in the reading; else `None`) |
| `UnknownReading` | `image` and `raw`: a reading of a booster this version does not know, or one with a field missing or of the wrong type. The decision does not fail |

Booster calls are billed from your prepaid balance with the request's tokens, one per
booster and image read with status `ok`: a booster that timed out or failed is free, and so
are the boosters of a refused request. The readings' tokens are input tokens at the model's
price. The price per call is on the
[pricing page](https://decisionnode.com/docs/pricing#boosters).

| Status | `error_type` | Exception |
|---|---|---|
| 400 | `api_usage_error`: `boosters` malformed (not a list, more than 4 entries, a booster named twice, an unknown key in `options`); the reason is in `e.reason` | `BadRequestError` |
| 400 | `booster_unknown`: a name that is not a booster | `BadRequestError` |
| 400 | `booster_input_mismatch`: a booster that reads images on a request without any, or an `images` entry that is not an image of the request | `BadRequestError` |
| 403 | `booster_not_enabled`: the booster is turned off for your workspace | `PermissionDeniedError` |
| 529 | `booster_unavailable`: a booster with `require=True` failed or did not answer in time | `OverloadedError`, retried |

Every error of a request that named boosters carries `e.boosters`: what each booster did,
in the request's order, as booster name to `BoosterStatus` (`status`, `latency_ms`), so you
can see which required booster failed.

```python
from decisionnode import OverloadedError

try:
    d = dn.decide(images=images, boosters=[Booster("age", require=True)], questions=questions)
except OverloadedError as e:
    if e.error_type == "booster_unavailable":
        print(e.boosters["age"].status)  # "timeout"
```

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
| `decisionnode-moe` | The main model, always its current release. The default. |
| `decisionnode-flash` | Faster and lower cost, always its current release. |

Choose per client (`DecisionNode(model="decisionnode-flash")`) or per call
(`dn.decide(..., model="decisionnode-moe")`). Each model publishes its limits in
`models()`.

`models()` also lists the pinned ids served, `decisionnode-<family>-<version>` (such as
`decisionnode-moe-<version>`); send one to stay on that release. Every answer's `model` is
the pinned id that answered. `model` is free text: a pinned id, a fine-tuned name and the
older ids the API still accepts (`decisionnode-1.0`, `decisionnode-1.0-flash`,
`decisionnode-latest`, `decisionnode-flash-latest`) are sent as given.

### Fallback

A request for `decisionnode-moe` that no worker of that model can take at once is
answered by Flash instead, billed at its price. `d.model` is the model that answered, and
`d.fallback` names both: the model that answered by its pinned id, and the model asked for
as it was asked (a family id or an older id as the family id, a pinned id as itself):

```python
d = dn.decide(state=ticket, questions=questions, model="decisionnode-moe")
if d.fallback is not None:
    d.fallback.served_model  # "decisionnode-flash-<version>"
    d.fallback.requested_model  # "decisionnode-moe"
```

`fallback=False` turns it off for one request: such a request gets the 529
`OverloadedError` instead, which is retried like every 529. `decide`, the helpers,
`decide_many` requests and batch requests take it; left out, the API's default (on)
applies. Sessions, batch lines and fine-tuned models never fall back.

### Fine-tuned models

A model your workspace trained in the console is called by its name, by your workspace's
keys only, and answers like its base: the same question types, answers, limits and
errors.

```python
d = dn.decide(state=claim, questions=questions, model="decisionnode-moe@acme/fraud")
d.model  # "decisionnode-moe@acme/fraud:3", the version that answered
```

The name is `<base>@<workspace>/<name>`, which means the version deployed most recently,
or `<base>@<workspace>/<name>:<n>` for version `n`. The client sends it as given; a name
that does not exist, a version that is not deployed and another workspace's model all
raise `BadRequestError` (`Unknown model: <name>`). `Decision.model` is always the pinned
name, so you can log which version answered.

`models()` lists your workspace's fine-tuned models beside the catalogue's, each with
`fine_tuned` (`True`), `base` (the base model), `adapter_version` (the version number) and
`status` (`deployed`, which answers decisions, or `ready`: trained and not deployed). On
the catalogue's own entries these four are `None`.

## Errors

Every error derives from `DecisionNodeError`, with `status`, `error_type`, `request_id`,
`boosters` (for a request with boosters, what each booster did; an empty mapping otherwise)
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
| 403, not a refusal | `PermissionDeniedError`: your workspace may not do this, for example use a booster that is turned off for it | |
| 404 | `NotFoundError` | |
| 413 | `RequestTooLargeError`: the body is over 25 MiB; never retried (also raised before sending) | |
| 422 | `ValidationError` | `entries` |
| 429 | `RateLimitError`; `SessionLimitError` for too many open sessions | `retry_after` (seconds) |
| 529 | `OverloadedError`, also when a required booster failed (`error_type` `booster_unavailable`) | `retry_after` (seconds) |
| other 5xx, a 2xx whose body is not JSON, or a helper's own answer missing or malformed | `ServerError` | |
| 3xx and every other status | `DecisionNodeError` itself, with `status` | |
| no response | `ConnectionError`, `TimeoutError` | |

An error whose body is not JSON (a page from a proxy or the service's edge, or no body at
all) is classed by its status alone, as in the table: a 400 is a `BadRequestError`, a 403
a `PermissionDeniedError`, a 422 a `ValidationError` with no `entries`. Its `error_type` is
`None`, `request_id` and `reason` come from the response headers when it has them, and the
page is never read: the message is the client's own, `HTTP <status> <text>` (for example
`HTTP 413 Content Too Large`), followed by ` (request id <id>)` when the response has an
`x-request-id`. It is retried as any other error of its status: a 408 is, a 413 is not.

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
- a body over 25 MiB as it will be sent (`RequestTooLargeError`, `error_type`
  `request_too_large_error`);
- the shape of each question and of the state (`ValidationError`): a question that is not
  an object or has no `type`, values that must be a string, an object or an array, a
  choice or score without `criteria`, a score without levels, and every number grid rule;
- the question limits (`BadRequestError`): an unknown type ("Invalid request."), a choice
  without options or with more than 255, a
  score with more than 10 levels, a truth question with neither instructions nor
  criteria, a number question without instructions;
- every image rule above, an id for each image of at most 64 characters (counted as
  characters, not bytes: "Image 1: its id is longer than 64 characters."), each id used
  once, and for a URL: `https`, at most 2,048 characters, no user information, port 443,
  not a literal private address (`image_url_refused`);
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
`open_session`, `batches.create` or `batches.add`) would run before sending,
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

Each attempt times out after 10 seconds (30 for a request whose boosters name `ocr`,
unless you set a timeout; see Boosters). The timeout covers the whole attempt, from
sending the request to the last byte of the response body, so a server that answers
slowly a little at a time cannot hold a call longer than that. A storage `PUT` of an
image has its own deadline for the whole attempt, `upload_timeout` (60 seconds by
default), since up to 10 MB may need longer than an answer does. Batch results are read
from the network line by line as you iterate them, each read bounded by the timeout, so a
file of any size is read whole. A failed request is retried twice on 408, 429,
any 5xx (529 included), a timeout or a dropped connection; a request naming `ocr` is not
retried after a timeout, since the server finishes that attempt. The wait is `retry-after-ms`
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
| `model` | `DECISIONNODE_MODEL` | `decisionnode-moe` |
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
        "model": "decisionnode-moe",
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
skip. The tests pass suite 1.12; CI checks the suite out at the commit pinned in
`CONFORMANCE_REF` in `.github/workflows/ci.yml`, with every action pinned to a commit SHA
and no checkout keeping its credentials.

```sh
DECISIONNODE_CONFORMANCE_DIR=/path/to/decision-node-conformance uv run pytest
```

This package implements version 1.2 of the specification shared by every official
DecisionNode SDK, exported as `decisionnode.SPEC_VERSION`.

## License

MIT. See [LICENSE](LICENSE).
