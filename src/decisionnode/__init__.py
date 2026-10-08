"""DecisionNode for Python.

from decisionnode import DecisionNode, truth, choice, score, number

dn = DecisionNode()  # reads DECISIONNODE_API_KEY
d = dn.decide(
    state={"ticket": "I was charged twice and nobody has replied for 3 days."},
    questions={
        "route": choice("Where should this go?", ["billing", "bug", "account"]),
        "escalate": truth("Escalate to a human now?"),
        "days": number("How many days has the customer waited?", min=0, max=30),
    },
)
d["route"].choice      # "billing"
d["escalate"].truth    # 0.94
d["days"].number       # 3
"""

from ._batches import AsyncBatchResults, Batch, BatchPage, BatchResult, BatchResults, BatchRun
from ._client import AsyncBatches, AsyncDecisionNode, Batches, BatchRequest, DecideRequest, DecisionNode, DecisionResult
from ._errors import (
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    ContextLimitError,
    DecisionNodeError,
    InsufficientCreditError,
    NotFoundError,
    OverloadedError,
    PermissionDeniedError,
    RateLimitError,
    RefusalError,
    ServerError,
    SessionLimitError,
    TimeoutError,
    ValidationError,
)
from ._images import DEFAULT_INLINE_THRESHOLD, Image
from ._questions import (
    ChoiceQuestion,
    NumberQuestion,
    Question,
    ScoreQuestion,
    TruthQuestion,
    choice,
    number,
    score,
    truth,
)
from ._sessions import AsyncSession, FrameError, Reply, Session, SessionClosed, SessionEvent
from ._types import (
    Answer,
    AnyAnswer,
    ChoiceAnswer,
    Decision,
    ModelInfo,
    NumberAnswer,
    Safety,
    ScoreAnswer,
    TruthAnswer,
    UnknownAnswer,
    Usage,
)
from ._version import SPEC_VERSION, __version__
from ._videos import Video, VideoFrame

__all__ = [
    "__version__",
    "SPEC_VERSION",
    "DEFAULT_INLINE_THRESHOLD",
    # clients
    "DecisionNode",
    "AsyncDecisionNode",
    "DecideRequest",
    "DecisionResult",
    # questions
    "truth",
    "choice",
    "score",
    "number",
    "Question",
    "TruthQuestion",
    "ChoiceQuestion",
    "ScoreQuestion",
    "NumberQuestion",
    "Image",
    "Video",
    "VideoFrame",
    # results
    "Decision",
    "Answer",
    "AnyAnswer",
    "TruthAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "NumberAnswer",
    "UnknownAnswer",
    "Usage",
    "Safety",
    "ModelInfo",
    # sessions
    "Session",
    "AsyncSession",
    "Reply",
    "FrameError",
    "SessionClosed",
    "SessionEvent",
    # batches
    "Batches",
    "AsyncBatches",
    "BatchRequest",
    "Batch",
    "BatchPage",
    "BatchResult",
    "BatchResults",
    "AsyncBatchResults",
    "BatchRun",
    # errors
    "DecisionNodeError",
    "BadRequestError",
    "ContextLimitError",
    "AuthenticationError",
    "InsufficientCreditError",
    "RefusalError",
    "PermissionDeniedError",
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    "SessionLimitError",
    "OverloadedError",
    "ServerError",
    "ConnectionError",
    "TimeoutError",
]
