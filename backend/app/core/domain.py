"""Base class for anticipated failures that carry a stable machine-readable code."""


class DomainError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
