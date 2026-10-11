"""Shared exceptions for Route A data contracts."""


class DataContractError(ValueError):
    """Raised when source or derived data violates the frozen contract."""
