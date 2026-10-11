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
from ._boosters import (
    AgeReading,
    AiGeneratedReading,
    AnyReading,
    Booster,
    BoosterInput,
    BoosterStatus,
    Evidence,
    OcrReading,
    Reading,
    TopGenerator,
    UnknownReading,
)
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
    RequestTooLargeError,
    ServerError,
    SessionLimitError,
    TimeoutError,
    ValidationError,
)
from ._images import DEFAULT_INLINE_THRESHOLD, Image
from ._questions import (
    BoxQuestion,
    ChoiceQuestion,
    NumberQuestion,
    PointQuestion,
    Question,
    ScoreQuestion,
    TruthQuestion,
    box,
    choice,
    number,
    point,
    score,
    truth,
)
from ._sessions import AsyncSession, FrameError, Reply, Session, SessionClosed, SessionEvent
from ._types import (
    Answer,
    AnyAnswer,
    BoxAnswer,
    ChoiceAnswer,
    Cost,
    Decision,
    Fallback,
    ModelInfo,
    NumberAnswer,
    PointAnswer,
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
    "point",
    "box",
    "Question",
    "TruthQuestion",
    "ChoiceQuestion",
    "ScoreQuestion",
    "NumberQuestion",
    "PointQuestion",
    "BoxQuestion",
    "Image",
    "Video",
    "VideoFrame",
    # boosters
    "Booster",
    "BoosterInput",
    # results
    "Decision",
    "Fallback",
    "Answer",
    "AnyAnswer",
    "TruthAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "NumberAnswer",
    "PointAnswer",
    "BoxAnswer",
    "UnknownAnswer",
    "Cost",
    "Usage",
    "Safety",
    "ModelInfo",
    "Evidence",
    "Reading",
    "AnyReading",
    "AgeReading",
    "AiGeneratedReading",
    "OcrReading",
    "TopGenerator",
    "UnknownReading",
    "BoosterStatus",
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
    "RequestTooLargeError",
    "ValidationError",
    "RateLimitError",
    "SessionLimitError",
    "OverloadedError",
    "ServerError",
    "ConnectionError",
    "TimeoutError",
]
