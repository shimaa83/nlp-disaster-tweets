from contextvars import ContextVar
from typing import Optional

# متغير سياق لحفظ الـ Correlation ID للطلب الحالي
_correlation_id_ctx_var: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)


def get_correlation_id() -> Optional[str]:
    """Retrieve the current correlation ID."""
    return _correlation_id_ctx_var.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context."""
    _correlation_id_ctx_var.set(correlation_id)
