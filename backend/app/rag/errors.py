from app.core.domain import DomainError


class RagError(DomainError):
    """An anticipated RAG failure (unsupported type, too large, model unavailable, ...)."""
