"""Pure-CPU fake-quantization protocol for InfiniteVL Stage-2.

Every result produced by this package is a quality result from a reference
PyTorch backend.  It is not evidence of packed low-bit memory or speed gains.
"""

from .config import MODEL_ID, MODEL_REVISION, ProtocolConfig, QuantScheme

__all__ = ["MODEL_ID", "MODEL_REVISION", "ProtocolConfig", "QuantScheme"]
