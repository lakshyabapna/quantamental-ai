"""Quantamental AI - Custom Exception Hierarchy.

Defines domain-specific exceptions for calendar operations, data provider ingestion,
data quality validation, and quantitative financial optimization.
"""

from typing import Any, Dict, Optional


class QuantamentalException(Exception):
    """Base exception for all domain-specific exceptions in Quantamental AI."""

    def __init__(
        self,
        message: str,
        error_code: str = "QUANTAMENTAL_INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary representation for API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# Configuration Exceptions
class ConfigurationError(QuantamentalException):
    """Raised when environment or runtime configuration is invalid."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            details=details,
        )


# Calendar Exceptions
class CalendarException(QuantamentalException):
    """Base exception for trading calendar operations."""

    def __init__(
        self,
        message: str,
        error_code: str = "CALENDAR_ERROR",
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class CalendarDateRangeError(CalendarException):
    """Raised when start_date is strictly greater than end_date or dates are invalid."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="CALENDAR_DATE_RANGE_ERROR",
            status_code=400,
            details=details,
        )


class InvalidCalendarDateError(CalendarException):
    """Raised when an invalid date object or format is passed to calendar functions."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="INVALID_CALENDAR_DATE_ERROR",
            status_code=400,
            details=details,
        )


class HolidayCircularNotFoundError(CalendarException):
    """Raised when requested holiday circular metadata is not found."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="HOLIDAY_CIRCULAR_NOT_FOUND",
            status_code=404,
            details=details,
        )


# Data Provider Exceptions
class DataProviderException(QuantamentalException):
    """Base exception for data provider operations."""

    def __init__(
        self,
        message: str,
        error_code: str = "DATA_PROVIDER_ERROR",
        status_code: int = 502,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class ProviderValidationError(DataProviderException):
    """Raised when data returned by a provider fails schema or business validation."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="PROVIDER_VALIDATION_ERROR",
            status_code=422,
            details=details,
        )


class ProviderDataNotFoundError(DataProviderException):
    """Raised when requested historical market data or symbol cannot be found."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="PROVIDER_DATA_NOT_FOUND",
            status_code=404,
            details=details,
        )


class ProviderConnectionError(DataProviderException):
    """Raised when connection to data provider fails or times out."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="PROVIDER_CONNECTION_ERROR",
            status_code=503,
            details=details,
        )


# Data Quality Exceptions
class DataQualityException(QuantamentalException):
    """Base exception for data hygiene and quality violations."""

    def __init__(
        self,
        message: str,
        error_code: str = "DATA_QUALITY_ERROR",
        status_code: int = 422,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class QuarantineAssetError(DataQualityException):
    """Raised when an asset fails quality checks and is quarantined from portfolio analysis."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="QUARANTINE_ASSET_ERROR",
            status_code=422,
            details=details,
        )


# Financial & Optimization Exceptions
class OptimizationException(QuantamentalException):
    """Base exception for portfolio optimization operations."""

    def __init__(
        self,
        message: str,
        error_code: str = "OPTIMIZATION_ERROR",
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class InfeasibleProblemError(OptimizationException):
    """Raised when an optimization problem is mathematically infeasible."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="INFEASIBLE_PROBLEM_ERROR",
            status_code=422,
            details=details,
        )


class NonPositiveDefiniteMatrixError(OptimizationException):
    """Raised when a covariance or correlation matrix fails positive-definiteness requirements."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="NON_POSITIVE_DEFINITE_MATRIX_ERROR",
            status_code=422,
            details=details,
        )


class IllConditionedMatrixError(OptimizationException):
    """Raised when matrix condition number exceeds maximum numerical stability threshold."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="ILL_CONDITIONED_MATRIX_ERROR",
            status_code=422,
            details=details,
        )


# Financial Calculation Exceptions
class FinancialCalculationError(QuantamentalException):
    """Base exception for mathematical and financial calculation errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "FINANCIAL_CALCULATION_ERROR",
        status_code: int = 422,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class InsufficientDataError(FinancialCalculationError):
    """Raised when there are fewer observations than the minimum required for calculation."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_DATA_ERROR",
            status_code=422,
            details=details,
        )


class InvalidPriceError(FinancialCalculationError):
    """Raised when asset prices are non-positive, NaN, infinite, or malformed."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="INVALID_PRICE_ERROR",
            status_code=422,
            details=details,
        )


class UnalignedObservationsError(FinancialCalculationError):
    """Raised when cross-sectional observations cannot be aligned on a common observation window."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="UNALIGNED_OBSERVATIONS_ERROR",
            status_code=422,
            details=details,
        )


class CovarianceValidationError(FinancialCalculationError):
    """Raised when a computed covariance matrix fails post-calculation validation.

    Examples include non-finite entries or failed symmetry checks.
    See ARCHITECTURE.md §4.2 and FINANCIAL_METHODOLOGY.md §4.1.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="COVARIANCE_VALIDATION_ERROR",
            status_code=422,
            details=details,
        )


class HighamConvergenceError(FinancialCalculationError):
    """Raised when Higham alternating projection fails to converge within max_iterations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="HIGHAM_CONVERGENCE_ERROR",
            status_code=422,
            details=details,
        )
