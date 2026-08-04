class FreeFlowError(Exception):
    """Base exception for FreeFlow API errors."""


class AuthenticationError(FreeFlowError):
    """Raised when authentication fails."""


class APIConnectionError(FreeFlowError):
    """Raised when the API cannot be reached."""