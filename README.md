# DecisionNode Python SDK

The official Python client for the DecisionNode API: ask yes-or-no, multiple-choice and
scale questions about any text, JSON or image, and get calibrated probabilities back.
Sync and async clients, fully typed.

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

Three question types. Each builder returns the question in its wire form, and a plain
`dict` in that form is accepted anywhere a builder is.

| Builder | Asks | Answer |
|---|---|---|
| `truth(instructions, if_true=None, if_false=None)` | Is this true? | `TruthAnswer.truth`: the probability of yes, 0 to 1 |
| `choice(instructions, options)` | Which one? `options` is a dict of name to description, or a list of names | `ChoiceAnswer.choice`, `.confidence`, `.probabilities` (by option name) |
| `score(instructions, levels)` | Where on this scale? `levels` are ordered low to high | `ScoreAnswer.score` (expected level index), `.confidence`, `.probabilities` and `.legend` (by level index, as `int`) |

`state`, `instructions`, option descriptions and levels take a string or any
JSON-serialisable value. Question ids are any string and come back unchanged; read
answers by id, never by position.

```python
truth("Is this a refund request?", if_true="the customer asks for money back")
choice("Which language is this?", ["en", "de", "sv"])
score({"question": "How complete is the form?", "focus": "required fields"}, ["empty", "partial", "complete"])
```

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

`check(state, instructions, if_true, if_false)`, `classify(state, instructions, options)`
and `score(state, instructions, levels)` send one question and return its answer. Like
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
argument in any of them raises `ValueError` and nothing is sent. Results come back in input order, one `DecisionResult` per
request: a `Decision`, or the error that request ended with. One failure does not stop
the others.

## Images

```python
from decisionnode import Image

d = dn.decide(
    state={"claim": {"total": "740.00", "currency": "SEK"}},
    images=[Image("receipt", path="receipt.jpg")],
    questions={"total_matches": truth("Does the total on the receipt equal the total in the claim?")},
)
```

`Image(id, path=...)` reads the file when the `Image` is made and takes its media type
from its first bytes (JPEG, PNG or WebP). `Image(id, data=...)` sends bytes you already
have; without `media_type` the type is taken from the first bytes the same way.
`Image(id, url=...)` sends an `https` URL the server fetches. A file that cannot be
read, or bytes in another format without a `media_type`, raise `ValueError` when the
`Image` is made. Up to 4 images a request, 10 MB each.

## Models

```python
for m in dn.models():
    print(m.name, m.release_date, m.limits)
```

Each entry is a `ModelInfo`.

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
| 429 | `RateLimitError` | `retry_after` (seconds) |
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
sent as JSON, an image that cannot be used, a `base_url` that is not `https`, a
`timeout` that is not above 0 or a negative `max_retries`. Everything else is the
server's to judge, because the limits differ by model and are published at `models()`.

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
slowly a little at a time cannot hold a call longer than that. A failed request is retried twice on 408, 429,
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

An argument wins over the environment, and the environment over the default. An empty
string (`api_key=""`, also `model=""` on a call) counts as not given. A client
is safe to share between threads and tasks, and reuses its connections; close it with
`dn.close()` (`await dn.close()`) or use it as a context manager.

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
never the state, questions or key:

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
client. Set `DECISIONNODE_CONFORMANCE_DIR` to a checkout of it; without it the
conformance tests fail rather than skip. The tests pass suite 1.3; CI checks the suite
out at the commit pinned in `CONFORMANCE_REF` in `.github/workflows/ci.yml`, with every
action pinned to a commit SHA and no checkout keeping its credentials.

```sh
DECISIONNODE_CONFORMANCE_DIR=/path/to/decision-node-conformance uv run pytest
```

This package implements version 1.0 of the specification shared by every official
DecisionNode SDK, exported as `decisionnode.SPEC_VERSION`.

## License

MIT. See [LICENSE](LICENSE).
