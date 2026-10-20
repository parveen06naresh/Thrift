"""Thrift: an OpenAI-compatible LLM gateway that cuts token spend with certified quality.

Same answers. Fewer tokens.
"""
__version__ = "0.1.0"

from .cache import SemanticCache  # noqa: E402
from .compress import Compressor  # noqa: E402
from .router import Router  # noqa: E402

__all__ = ["Compressor", "Router", "SemanticCache", "__version__"]
