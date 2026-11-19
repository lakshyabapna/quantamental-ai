"""Quantamental AI - Capital Allocation Line (CAL) Engine.

Implements Phase 2.3, Step 4: Deterministic Capital Allocation Line (CAL)
generation combining the risk-free asset and the optimal Tangency portfolio.

Mathematical Reference (FINANCIAL_METHODOLOGY.md §5.2, §5.3 & Modern Portfolio Theory):

    The Capital Allocation Line (CAL) represents all feasible risk-return profiles
    formed by combining the risk-free asset (with return r_f and volatility 0)
    and the optimal risky Tangency portfolio (with expected return mu_T and volatility sigma_T):

        Complete Portfolio Expected Return:
            E[R_c] = r_f + S_CAL * sigma_c

        CAL Slope (Sharpe Ratio of the Tangency Portfolio):
            S_CAL = (mu_T - r_f) / sigma_T

        Complete Portfolio Asset Allocations:
            Weight in Tangency Portfolio:  w_T  = sigma_c / sigma_T
            Weight in Risk-Free Asset:     w_rf = 1 - w_T

    Properties:
        - At sigma_c = 0:          E[R_c] = r_f    (100% invested in risk-free asset)
        - At sigma_c = sigma_T:    E[R_c] = mu_T   (100% invested in tangency portfolio)
        - At 0 < sigma_c < sigma_T: Lending portfolio (positive risk-free lending)
        - At sigma_c > sigma_T:    Theoretical leveraged portfolio (borrowing at r_f to invest in tangency)

    MVP No-Leverage Policy:
        - Under the canonical MVP portfolio constraints, portfolios are strictly long-only
          and fully invested with NO LEVERAGE (w_i >= 0, sum(w_i) = 1.0, w_T <= 1.0, w_rf >= 0.0).
        - By default, the CAL coordinate grid ends at 1.0 * tangency_volatility (lending and
          pure tangency points only), and any coordinate with weight_tangency > 1.0 is rejected.
        - Generating leveraged coordinates (sigma_c > sigma_T) requires explicit opt-in via
          allow_leverage=True. Leveraged coordinates are theoretical and not executable under MVP constraints.

    Canonical design references:
        - FINANCIAL_METHODOLOGY.md §2, §5.2, §5.3
        - ARCHITECTURE.md §4.2: Deterministic Financial Engine optimization pipeline.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from app.core.exceptions import (
    FinancialCalculationError,
    InsufficientDataError,
    OptimizationException,
)
from app.portfolio_engine.tangency import TangencyOptimizationResult

logger = logging.getLogger("quantamental.cal")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CAL_POINTS: int = 50
_TOLERANCE_RF_MATCH: float = 1e-6


# ---------------------------------------------------------------------------
# Result Containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CALPoint:
    """Immutable coordinate on the Capital Allocation Line.

    Attributes:
        volatility: Annualized complete-portfolio volatility (sigma_c >= 0).
        expected_return: Annualized complete-portfolio expected return E[R_c].
        weight_tangency: Fraction of portfolio invested in the Tangency portfolio (sigma_c / sigma_T).
        weight_risk_free: Fraction of portfolio invested in the risk-free asset (1 - weight_tangency).
    """

    volatility: float
    expected_return: float
    weight_tangency: float
    weight_risk_free: float

    def __iter__(self) -> Iterator[float]:
        """Support tuple unpacking: vol, ret = point."""
        yield self.volatility
        yield self.expected_return

    def to_dict(self) -> Dict[str, Any]:
        """Convert point to JSON-serializable dictionary."""
        return {
            "volatility": self.volatility,
            "expected_return": self.expected_return,
            "weight_tangency": self.weight_tangency,
            "weight_risk_free": self.weight_risk_free,
        }


@dataclass(frozen=True, init=False)
class CapitalAllocationLineResult:
    """Immutable container for Capital Allocation Line (CAL) results.

    Defensively immutable: internal tuples, metadata dictionary, and nested
    Tangency portfolio cannot be mutated by callers.

    Attributes:
        slope: Annualized CAL slope (Sharpe ratio of the Tangency portfolio).
        risk_free_rate: Annualized benchmark risk-free rate used for the CAL.
        tangency_expected_return: Annualized expected return of the Tangency portfolio.
        tangency_volatility: Annualized volatility of the Tangency portfolio.
        tangency_portfolio: Source Tangency portfolio optimization result (if provided).
        points: Tuple of CALPoint coordinates along the CAL (preserves caller ordering).
        metadata: Diagnostic and execution metadata.
    """

    slope: float
    risk_free_rate: float
    tangency_expected_return: float
    tangency_volatility: float
    tangency_portfolio: Optional[TangencyOptimizationResult]
    points: Tuple[CALPoint, ...]
    metadata: Dict[str, Any]
    allow_leverage: bool = False
    _points: Tuple[CALPoint, ...] = field(repr=False)
    _tangency_portfolio: Optional[TangencyOptimizationResult] = field(repr=False)
    _metadata: Dict[str, Any] = field(repr=False)

    def __init__(
        self,
        slope: float,
        risk_free_rate: float,
        tangency_expected_return: float,
        tangency_volatility: float,
        tangency_portfolio: Optional[TangencyOptimizationResult],
        points: Sequence[CALPoint],
        metadata: Dict[str, Any],
        allow_leverage: bool = False,
    ):
        object.__setattr__(self, "slope", float(slope))
        object.__setattr__(self, "risk_free_rate", float(risk_free_rate))
        object.__setattr__(self, "tangency_expected_return", float(tangency_expected_return))
        object.__setattr__(self, "tangency_volatility", float(tangency_volatility))
        object.__setattr__(self, "allow_leverage", bool(allow_leverage))
        object.__setattr__(self, "_points", tuple(points))

        if tangency_portfolio is not None:
            object.__setattr__(
                self,
                "_tangency_portfolio",
                dataclasses.replace(
                    tangency_portfolio,
                    weights=tangency_portfolio.weights.copy(),
                    metadata=copy.deepcopy(tangency_portfolio.metadata),
                ),
            )
        else:
            object.__setattr__(self, "_tangency_portfolio", None)

        object.__setattr__(self, "_metadata", copy.deepcopy(metadata))

    @property
    def points(self) -> Tuple[CALPoint, ...]:
        """Return tuple of CAL points."""
        return self._points

    @property
    def coordinates(self) -> Tuple[CALPoint, ...]:
        """Alias for points."""
        return self._points

    @property
    def tangency_portfolio(self) -> Optional[TangencyOptimizationResult]:
        """Return a defensive copy of the source Tangency result."""
        if self._tangency_portfolio is None:
            return None
        return dataclasses.replace(
            self._tangency_portfolio,
            weights=self._tangency_portfolio.weights.copy(),
            metadata=copy.deepcopy(self._tangency_portfolio.metadata),
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return a defensive deep copy of metadata."""
        return copy.deepcopy(self._metadata)

    def __iter__(self) -> Iterator[CALPoint]:
        """Allow direct iteration over CAL points."""
        return iter(self._points)

    def __len__(self) -> int:
        """Return number of points on the CAL."""
        return len(self._points)

    def to_dict(self) -> Dict[str, Any]:
        """Convert CAL result container to JSON-serializable dictionary."""
        return {
            "slope": self.slope,
            "risk_free_rate": self.risk_free_rate,
            "tangency_expected_return": self.tangency_expected_return,
            "tangency_volatility": self.tangency_volatility,
            "tangency_portfolio": self.tangency_portfolio.to_dict() if self.tangency_portfolio is not None else None,
            "points": [p.to_dict() for p in self.points],
            "allow_leverage": self.allow_leverage,
            "metadata": self.metadata,
        }


# Aliases for convenience and flexible API imports
CALResult = CapitalAllocationLineResult
CALCoordinate = CALPoint


# ---------------------------------------------------------------------------
# Input Validation Helpers
# ---------------------------------------------------------------------------

def _validate_numeric_scalar(value: object, name: str) -> float:
    """Validate that value is a finite real numeric scalar, strictly rejecting bools, complex, and strings."""
    if isinstance(value, (bool, np.bool_)):
        raise FinancialCalculationError(
            f"{name} must be a numeric float, got bool ({value!r}).",
            details={"param": name, "type": "bool", "value": value},
        )
    if isinstance(value, (complex, np.complexfloating)):
        raise FinancialCalculationError(
            f"{name} must be a real numeric value, got complex ({value!r}).",
            details={"param": name, "type": "complex", "value": str(value)},
        )
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise FinancialCalculationError(
            f"{name} must be a numeric float or int, got {type(value).__name__}.",
            details={"param": name, "type": type(value).__name__},
        )
    val_float = float(value)
    if not np.isfinite(val_float):
        raise FinancialCalculationError(
            f"{name} must be finite, got {value!r}.",
            details={"param": name, "value": value},
        )
    return val_float


def _validate_tangency_inputs(
    tangency_portfolio: Optional[Union[TangencyOptimizationResult, object]],
    tangency_expected_return: Optional[object],
    tangency_volatility: Optional[object],
    risk_free_rate: Optional[object],
) -> Tuple[float, float, float, Optional[TangencyOptimizationResult]]:
    """Validate Tangency portfolio inputs and extract (mu_T, sigma_T, rf, source_tangency)."""
    # Case 1: tangency_portfolio is provided as TangencyOptimizationResult
    if isinstance(tangency_portfolio, TangencyOptimizationResult):
        # Strict validation: do NOT silently substitute or accept fallback/failed results
        if not tangency_portfolio.success or tangency_portfolio.fallback_executed:
            raise OptimizationException(
                "Cannot generate Capital Allocation Line from an unconverged or fallback Tangency portfolio. "
                "CAL requires a mathematically verified optimal risky portfolio.",
                details={
                    "success": tangency_portfolio.success,
                    "fallback_executed": tangency_portfolio.fallback_executed,
                    "status": tangency_portfolio.status,
                    "message": tangency_portfolio.message,
                },
            )

        mu_t = _validate_numeric_scalar(tangency_portfolio.expected_return, "tangency_expected_return")
        sigma_t = _validate_numeric_scalar(tangency_portfolio.volatility, "tangency_volatility")
        if sigma_t <= 0.0:
            raise FinancialCalculationError(
                f"Tangency volatility must be strictly positive (> 0), got {sigma_t:.6f}.",
                details={"tangency_volatility": sigma_t},
            )

        # Risk-free rate resolution
        if risk_free_rate is not None:
            rf = _validate_numeric_scalar(risk_free_rate, "risk_free_rate")
            if abs(rf - tangency_portfolio.risk_free_rate) > _TOLERANCE_RF_MATCH:
                raise FinancialCalculationError(
                    f"Supplied risk_free_rate ({rf:.6f}) does not match the risk_free_rate used "
                    f"to optimize the Tangency portfolio ({tangency_portfolio.risk_free_rate:.6f}).",
                    details={
                        "supplied_rf": rf,
                        "tangency_rf": tangency_portfolio.risk_free_rate,
                    },
                )
        else:
            rf = _validate_numeric_scalar(tangency_portfolio.risk_free_rate, "risk_free_rate")

        return mu_t, sigma_t, rf, tangency_portfolio

    # Case 2: tangency_portfolio is None, caller supplies scalar components
    elif tangency_portfolio is None:
        if tangency_expected_return is None:
            raise FinancialCalculationError(
                "Must provide either a TangencyOptimizationResult or tangency_expected_return.",
                details={"missing": "tangency_expected_return"},
            )
        if tangency_volatility is None:
            raise FinancialCalculationError(
                "Must provide either a TangencyOptimizationResult or tangency_volatility.",
                details={"missing": "tangency_volatility"},
            )
        if risk_free_rate is None:
            raise FinancialCalculationError(
                "risk_free_rate must be provided.",
                details={"missing": "risk_free_rate"},
            )

        mu_t = _validate_numeric_scalar(tangency_expected_return, "tangency_expected_return")
        sigma_t = _validate_numeric_scalar(tangency_volatility, "tangency_volatility")
        if sigma_t <= 0.0:
            raise FinancialCalculationError(
                f"Tangency volatility must be strictly positive (> 0), got {sigma_t:.6f}.",
                details={"tangency_volatility": sigma_t},
            )
        rf = _validate_numeric_scalar(risk_free_rate, "risk_free_rate")

        return mu_t, sigma_t, rf, None

    else:
        raise FinancialCalculationError(
            f"tangency_portfolio must be a TangencyOptimizationResult, got {type(tangency_portfolio).__name__}.",
            details={"type": type(tangency_portfolio).__name__},
        )


def _validate_volatilities(volatilities: object) -> List[float]:
    """Validate sequence of non-negative volatility coordinates, preserving input ordering."""
    if volatilities is None:
        raise FinancialCalculationError(
            "volatilities sequence cannot be None.",
            details={"type": "NoneType"},
        )

    # Reject scalars directly
    if isinstance(volatilities, (int, float, complex, str, bytes, bool, np.number, np.bool_)):
        raise FinancialCalculationError(
            f"volatilities must be an iterable sequence of non-negative floats, got scalar {type(volatilities).__name__}.",
            details={"type": type(volatilities).__name__},
        )

    # Reject boolean numpy arrays or pandas Series explicitly
    if isinstance(volatilities, np.ndarray) and np.issubdtype(volatilities.dtype, np.bool_):
        raise FinancialCalculationError(
            "volatilities cannot contain boolean values.",
            details={"dtype": str(volatilities.dtype)},
        )
    if isinstance(volatilities, pd.Series) and pd.api.types.is_bool_dtype(volatilities):
        raise FinancialCalculationError(
            "volatilities cannot contain boolean values.",
            details={"dtype": str(volatilities.dtype)},
        )

    try:
        iterator = iter(volatilities)  # type: ignore
    except TypeError as exc:
        raise FinancialCalculationError(
            f"volatilities must be an iterable sequence, got {type(volatilities).__name__}.",
            details={"type": type(volatilities).__name__},
        ) from exc

    cleaned_vols: List[float] = []
    for i, item in enumerate(iterator):
        if isinstance(item, (bool, np.bool_)):
            raise FinancialCalculationError(
                f"volatilities[{i}] must be a numeric float, got bool ({item!r}).",
                details={"index": i, "type": "bool", "value": item},
            )
        if isinstance(item, (complex, np.complexfloating)):
            raise FinancialCalculationError(
                f"volatilities[{i}] must be a real numeric value, got complex ({item!r}).",
                details={"index": i, "type": "complex", "value": str(item)},
            )
        if not isinstance(item, (int, float, np.integer, np.floating)):
            raise FinancialCalculationError(
                f"volatilities[{i}] must be a numeric float, got {type(item).__name__}.",
                details={"index": i, "type": type(item).__name__},
            )
        v = float(item)
        if not np.isfinite(v):
            raise FinancialCalculationError(
                f"volatilities[{i}] must be finite, got {item!r}.",
                details={"index": i, "value": item},
            )
        if v < 0.0:
            raise FinancialCalculationError(
                f"Portfolio volatility coordinates must be non-negative (>= 0), got {v:.6f} at index {i}.",
                details={"index": i, "volatility": v},
            )
        cleaned_vols.append(v)

    if len(cleaned_vols) == 0:
        raise InsufficientDataError(
            "volatilities sequence is empty; at least 1 volatility coordinate is required.",
            details={"len": 0},
        )

    return cleaned_vols


# ---------------------------------------------------------------------------
# Mathematical Functions
# ---------------------------------------------------------------------------

def calculate_cal_slope(
    tangency_portfolio: Optional[Union[TangencyOptimizationResult, object]] = None,
    *,
    tangency_expected_return: Optional[object] = None,
    tangency_volatility: Optional[object] = None,
    risk_free_rate: Optional[object] = None,
) -> float:
    """Calculate the slope of the Capital Allocation Line (Sharpe ratio of Tangency portfolio).

    Mathematical Formula:
        S_CAL = (mu_T - r_f) / sigma_T

    Parameters
    ----------
    tangency_portfolio:
        Optional TangencyOptimizationResult from optimize_maximum_sharpe_ratio.
        If provided, expected return, volatility, and risk-free rate are extracted.
    tangency_expected_return:
        Annualized expected return of the Tangency portfolio (if tangency_portfolio is None).
    tangency_volatility:
        Annualized volatility of the Tangency portfolio (must be strictly positive > 0).
    risk_free_rate:
        Annualized risk-free rate.

    Returns
    -------
    float
        Annualized CAL slope.

    Raises
    ------
    FinancialCalculationError:
        If inputs are invalid, non-finite, complex, boolean, or volatility <= 0.
    OptimizationException:
        If tangency_portfolio is a fallback or unconverged result.
    """
    mu_t, sigma_t, rf, _ = _validate_tangency_inputs(
        tangency_portfolio=tangency_portfolio,
        tangency_expected_return=tangency_expected_return,
        tangency_volatility=tangency_volatility,
        risk_free_rate=risk_free_rate,
    )
    slope = float((mu_t - rf) / sigma_t)
    if not np.isfinite(slope):
        raise FinancialCalculationError(
            f"Calculated CAL slope overflowed or is non-finite: {slope!r}.",
            details={"slope": slope, "mu_T": mu_t, "sigma_T": sigma_t, "rf": rf},
        )
    return slope


def calculate_cal_return(
    volatility: object,
    slope: object,
    risk_free_rate: object,
) -> float:
    """Calculate complete-portfolio expected return along the Capital Allocation Line.

    Mathematical Formula:
        E[R_c] = r_f + S_CAL * sigma_c

    Parameters
    ----------
    volatility:
        Complete-portfolio volatility (sigma_c >= 0).
    slope:
        CAL slope S_CAL.
    risk_free_rate:
        Annualized risk-free rate r_f.

    Returns
    -------
    float
        Annualized complete-portfolio expected return.

    Raises
    ------
    FinancialCalculationError:
        If any input is non-numeric, non-finite, complex, boolean, or volatility < 0,
        or if the computed expected return overflows to non-finite.
    """
    vol = _validate_numeric_scalar(volatility, "volatility")
    if vol < 0.0:
        raise FinancialCalculationError(
            f"Portfolio volatility must be non-negative (>= 0), got {vol:.6f}.",
            details={"volatility": vol},
        )
    s = _validate_numeric_scalar(slope, "slope")
    rf = _validate_numeric_scalar(risk_free_rate, "risk_free_rate")

    expected_return = float(rf + s * vol)
    if not np.isfinite(expected_return):
        raise FinancialCalculationError(
            f"Calculated complete-portfolio expected return overflowed or is non-finite: {expected_return!r}.",
            details={"expected_return": expected_return, "volatility": vol, "slope": s, "rf": rf},
        )
    return expected_return


def generate_cal_coordinates(
    tangency_portfolio: Optional[Union[TangencyOptimizationResult, object]] = None,
    volatilities: Optional[object] = None,
    *,
    tangency_expected_return: Optional[object] = None,
    tangency_volatility: Optional[object] = None,
    risk_free_rate: Optional[object] = None,
    n_points: int = DEFAULT_CAL_POINTS,
    max_volatility_multiplier: Optional[float] = None,
    allow_leverage: bool = False,
) -> CapitalAllocationLineResult:
    """Generate Capital Allocation Line (CAL) coordinates and complete portfolio allocations.

    Evaluates:
        E[R_c] = r_f + S_CAL * sigma_c
        w_T = sigma_c / sigma_T
        w_rf = 1 - w_T

    MVP No-Leverage Policy:
        - By default (allow_leverage=False), generates coordinates spanning [0.0, 1.0 * sigma_T]
          (lending and pure tangency portfolios only, where w_T <= 1.0 and w_rf >= 0.0).
        - Any coordinate exceeding tangency volatility is rejected under the default policy.
        - Set allow_leverage=True to generate theoretical leveraged coordinates (sigma_c > sigma_T,
          w_T > 1.0, w_rf < 0.0). Note that leveraged coordinates are theoretical and not executable
          under MVP long-only, fully invested constraints.

    Preserves the exact ordering of input `volatilities`. If `volatilities` is None,
    generates a grid of `n_points` linearly spaced from 0.0 to
    `max_volatility_multiplier * sigma_T` (default multiplier is 1.0).

    Parameters
    ----------
    tangency_portfolio:
        Optional converged TangencyOptimizationResult instance.
    volatilities:
        Optional sequence of non-negative portfolio volatilities (sigma_c >= 0).
        If provided, ordering is strictly preserved.
    tangency_expected_return:
        Annualized expected return of the Tangency portfolio (if tangency_portfolio is None).
    tangency_volatility:
        Annualized volatility of the Tangency portfolio (must be > 0).
    risk_free_rate:
        Annualized risk-free rate.
    n_points:
        Number of points to generate if volatilities is None (default 50).
    max_volatility_multiplier:
        Upper bound multiplier for default grid. Defaults to 1.0 (no leverage).
        Values > 1.0 require allow_leverage=True.
    allow_leverage:
        Opt-in flag permitting theoretical leveraged CAL coordinates (default False).

    Returns
    -------
    CapitalAllocationLineResult
        Immutable result container with slope, risk_free_rate, coordinates, and metadata.

    Raises
    ------
    FinancialCalculationError:
        If inputs are invalid, non-finite, complex, boolean, or volatility < 0,
        if leverage is attempted with allow_leverage=False, or if computed outputs overflow.
    InsufficientDataError:
        If an empty sequence of volatilities is passed.
    OptimizationException:
        If tangency_portfolio is unconverged or a fallback portfolio.
    """
    if isinstance(allow_leverage, (bool, np.bool_)):
        allow_leverage_bool = bool(allow_leverage)
    else:
        raise FinancialCalculationError(
            f"allow_leverage must be a boolean, got {type(allow_leverage).__name__}.",
            details={"type": type(allow_leverage).__name__},
        )

    mu_t, sigma_t, rf, source_tangency = _validate_tangency_inputs(
        tangency_portfolio=tangency_portfolio,
        tangency_expected_return=tangency_expected_return,
        tangency_volatility=tangency_volatility,
        risk_free_rate=risk_free_rate,
    )

    slope = float((mu_t - rf) / sigma_t)
    if not np.isfinite(slope):
        raise FinancialCalculationError(
            f"Calculated CAL slope overflowed or is non-finite: {slope!r}.",
            details={"slope": slope, "mu_T": mu_t, "sigma_T": sigma_t, "rf": rf},
        )

    # Determine volatility coordinates
    if volatilities is not None:
        vols = _validate_volatilities(volatilities)
    else:
        if isinstance(n_points, (bool, np.bool_)) or not isinstance(n_points, (int, np.integer)):
            raise FinancialCalculationError(
                f"n_points must be an integer, got {type(n_points).__name__}.",
                details={"n_points": n_points},
            )
        if n_points < 1:
            raise FinancialCalculationError(
                f"n_points must be at least 1, got {n_points}.",
                details={"n_points": n_points},
            )

        # Default multiplier is 1.0 (ends at 1.0 * tangency_volatility)
        if max_volatility_multiplier is None:
            mult = 1.0
        else:
            mult = _validate_numeric_scalar(max_volatility_multiplier, "max_volatility_multiplier")
            if mult <= 0.0:
                raise FinancialCalculationError(
                    f"max_volatility_multiplier must be strictly positive (> 0), got {mult}.",
                    details={"max_volatility_multiplier": mult},
                )
            if not allow_leverage_bool and mult > 1.0:
                raise FinancialCalculationError(
                    f"max_volatility_multiplier={mult:.6f} > 1.0 produces leveraged coordinates, "
                    "which are not permitted under the MVP no-leverage policy. "
                    "Set allow_leverage=True to enable theoretical leveraged coordinates.",
                    details={"max_volatility_multiplier": mult, "allow_leverage": allow_leverage_bool},
                )

        max_vol = float(mult * sigma_t)
        if not np.isfinite(max_vol):
            raise FinancialCalculationError(
                f"Calculated maximum volatility overflowed or is non-finite: {max_vol!r}.",
                details={"max_vol": max_vol, "mult": mult, "sigma_T": sigma_t},
            )
        vols = list(np.linspace(0.0, max_vol, int(n_points)))
        if not allow_leverage_bool:
            # Defensive guard: ensure linspace roundoff does not exceed tangency volatility
            vols = [min(v, sigma_t) for v in vols]

    # Compute CAL points with output validation and no-leverage enforcement
    points: List[CALPoint] = []
    for sigma_c in vols:
        # Floating-point boundary precision: if sigma_c is within machine precision
        # of tangency volatility (e.g. numerical roundoff from upstream float operations),
        # snap to exact tangency (w_T = 1.0, w_rf = 0.0) so the boundary point remains valid
        # without permitting actual leverage.
        if abs(sigma_c - sigma_t) <= 1e-14 * max(sigma_t, 1.0):
            sigma_c_eval = sigma_t
            w_t = 1.0
            w_rf = 0.0
        else:
            sigma_c_eval = sigma_c
            w_t = float(sigma_c_eval / sigma_t)
            w_rf = float(1.0 - w_t)

        if not np.isfinite(w_t):
            raise FinancialCalculationError(
                f"Calculated tangency allocation weight overflowed or is non-finite at volatility {sigma_c}: {w_t!r}.",
                details={"volatility": sigma_c, "sigma_T": sigma_t},
            )

        if not np.isfinite(w_rf):
            raise FinancialCalculationError(
                f"Calculated risk-free allocation weight overflowed or is non-finite at volatility {sigma_c}: {w_rf!r}.",
                details={"weight_tangency": w_t},
            )

        if not allow_leverage_bool and (w_t > 1.0 or w_rf < 0.0):
            raise FinancialCalculationError(
                f"Leveraged coordinate (volatility {sigma_c:.6f} > tangency volatility {sigma_t:.6f}, "
                f"weight_tangency={w_t:.6f} > 1.0, weight_risk_free={w_rf:.6f} < 0.0) is not permitted "
                "under the MVP no-leverage policy. Set allow_leverage=True to enable theoretical leveraged coordinates.",
                details={
                    "volatility": sigma_c,
                    "tangency_volatility": sigma_t,
                    "weight_tangency": w_t,
                    "weight_risk_free": w_rf,
                    "allow_leverage": allow_leverage_bool,
                },
            )

        exp_ret = float(rf + slope * sigma_c_eval)
        if not np.isfinite(exp_ret):
            raise FinancialCalculationError(
                f"Calculated complete-portfolio expected return overflowed or is non-finite at volatility {sigma_c}: {exp_ret!r}.",
                details={"volatility": sigma_c, "expected_return": exp_ret, "slope": slope, "rf": rf},
            )

        points.append(
            CALPoint(
                volatility=sigma_c_eval,
                expected_return=exp_ret,
                weight_tangency=w_t,
                weight_risk_free=w_rf,
            )
        )

    # Construct metadata
    source_metadata: Optional[Dict[str, Any]] = None
    if source_tangency is not None:
        source_metadata = {
            "success": source_tangency.success,
            "status": source_tangency.status,
            "message": source_tangency.message,
            "iterations": source_tangency.iterations,
            "fallback_executed": source_tangency.fallback_executed,
            "condition_number": source_tangency.metadata.get("condition_number"),
        }

    metadata: Dict[str, Any] = {
        "annualized_units": "annualized",
        "slope": slope,
        "risk_free_rate": rf,
        "tangency_expected_return": mu_t,
        "tangency_volatility": sigma_t,
        "number_of_points": len(points),
        "allow_leverage": allow_leverage_bool,
        "is_leveraged": any(p.weight_tangency > 1.0 for p in points),
        "leverage_policy": (
            "Theoretical leveraged CAL coordinates permitted (not executable under MVP long-only, no-leverage constraints)."
            if allow_leverage_bool
            else "MVP long-only, no-leverage constraint enforced (all coordinates satisfy weight_tangency <= 1.0 and weight_risk_free >= 0.0)."
        ),
        "source_tangency_result": source_metadata,
        "min_volatility": min(vols) if len(vols) > 0 else 0.0,
        "max_volatility": max(vols) if len(vols) > 0 else 0.0,
        "input_ordering_preserved": True,
    }

    logger.debug(
        "Generated CAL: slope=%.6f, rf=%.6f, mu_T=%.6f, sigma_T=%.6f, n_points=%d, allow_leverage=%s",
        slope,
        rf,
        mu_t,
        sigma_t,
        len(points),
        allow_leverage_bool,
    )

    return CapitalAllocationLineResult(
        slope=slope,
        risk_free_rate=rf,
        tangency_expected_return=mu_t,
        tangency_volatility=sigma_t,
        tangency_portfolio=source_tangency,
        points=points,
        metadata=metadata,
        allow_leverage=allow_leverage_bool,
    )


# Alias for intuitive caller ergonomics
generate_capital_allocation_line = generate_cal_coordinates
