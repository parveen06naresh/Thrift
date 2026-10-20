"""Thrift: an OpenAI-compatible LLM gateway that cuts token spend with certified quality.

Same answers. Fewer tokens.
"""
__version__ = "0.2.0"

from .cache import SemanticCache  # noqa: E402
from .client import AsyncThrift, Thrift  # noqa: E402
from .compress import Compressor  # noqa: E402
from .router import Ladder, Router  # noqa: E402

__all__ = ["AsyncThrift", "Compressor", "Ladder", "Router", "SemanticCache", "Thrift", "__version__"]
