"""Errors for Eiffel."""


class EiffelError(Exception):
    """Base class for Eiffel errors."""

    pass


class ConfigError(EiffelError):
    """Error raised when a configuration is invalid."""

    pass
