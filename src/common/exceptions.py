"""Project-level exceptions for the QA system."""


class AppError(Exception):
    """Base exception for application-level errors."""


class ConfigError(AppError):
    pass


class ValidationError(AppError):
    pass


class NotFoundError(AppError):
    pass


class ExternalServiceError(AppError):
    pass


class GraphError(ExternalServiceError):
    pass


class LLMError(ExternalServiceError):
    pass
