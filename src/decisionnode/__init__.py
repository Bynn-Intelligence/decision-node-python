"""DecisionNode for Python.

from decisionnode import DecisionNode, truth, choice, score

dn = DecisionNode()  # reads DECISIONNODE_API_KEY
d = dn.decide(
    state={"ticket": "I was charged twice and nobody has replied for 3 days."},
    questions={
        "route": choice("Where should this go?", ["billing", "bug", "account"]),
        "escalate": truth("Escalate to a human now?"),
    },
)
d["route"].choice      # "billing"
d["escalate"].truth    # 0.94
"""

from ._client import AsyncDecisionNode, DecideRequest, DecisionNode, DecisionResult
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
    TimeoutError,
    ValidationError,
)
from ._images import Image
from ._questions import ChoiceQuestion, Question, ScoreQuestion, TruthQuestion, choice, score, truth
from ._types import (
    Answer,
    AnyAnswer,
    ChoiceAnswer,
    Decision,
    ModelInfo,
    Safety,
    ScoreAnswer,
    TruthAnswer,
    UnknownAnswer,
    Usage,
)
from ._version import SPEC_VERSION, __version__

__all__ = [
    "__version__",
    "SPEC_VERSION",
    # clients
    "DecisionNode",
    "AsyncDecisionNode",
    "DecideRequest",
    "DecisionResult",
    # questions
    "truth",
    "choice",
    "score",
    "Question",
    "TruthQuestion",
    "ChoiceQuestion",
    "ScoreQuestion",
    "Image",
    # results
    "Decision",
    "Answer",
    "AnyAnswer",
    "TruthAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "UnknownAnswer",
    "Usage",
    "Safety",
    "ModelInfo",
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
    "OverloadedError",
    "ServerError",
    "ConnectionError",
    "TimeoutError",
]
