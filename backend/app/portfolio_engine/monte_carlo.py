"""Quantamental AI - Vectorized Monte Carlo Portfolio Simulator.

Implements Phase 2.5: High-throughput vectorized Monte Carlo portfolio simulation
generating long-only portfolios on the standard simplex via Dirichlet distribution,
and calculating annualized expected returns, variances, volatilities, and Sharpe ratios
for risk-return scatter visualization and empirical frontier validation.

Mathematical Reference (FINANCIAL_METHODOLOGY.md §5 & PROJECT_PLAN.md Phase 2 Task 6):
    Simplex Weight Generation:
        w_k ~ Dirichlet(alpha), where alpha_i = 1.0 for all i in {1, ..., N}
        Guarantees uniform density over the standard simplex:
            w_{k, i} >= 0,  sum_{i=1}^N w_{k, i} = 1.0
    Vectorized Portfolio Metrics (252-day annualization):
        portfolio_returns    = W @ mu
        portfolio_variances  = np.einsum("ij,jk,ik->i", W, Sigma, W)
        portfolio_volatility = np.sqrt(portfolio_variances)
        sharpe_ratio         = (portfolio_returns - r_f) / portfolio_volatility

Landmark Portfolios:
    - Simulated GMV: portfolio with minimum simulated volatility.
    - Simulated Max Sharpe: portfolio with maximum simulated Sharpe ratio.
    (Clearly labeled as simulated empirical approximations, not exact analytical optima).

Canonical design references:
    - FINANCIAL_METHODOLOGY.md §4, §5.1, §5.2, §5.3
    - ARCHITECTURE.md §4.2: Deterministic Financial Engine optimization pipeline.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from app.core.exceptions import (
    FinancialCalculationError,
    InsufficientDataError,
)
from app.portfolio_engine.efficient_frontier import (
    EfficientFrontierResult,
    FrontierPortfolio,
)

logger = logging.getLogger("quantamental.monte_carlo")

# ---------------------------------------------------------------------------
# Module Constants & Numerical Tolerances
# ---------------------------------------------------------------------------
DEFAULT_NUM_PORTFOLIOS: int = 10000
DEFAULT_RISK_FREE_RATE: float = 0.065  # 6.50% annualized GOI 91-day T-Bill baseline
_SYMMETRY_TOLERANCE: float = 1e-8
_PSD_EIGENVALUE_TOLERANCE: float = -1e-8
_VARIANCE_NON_NEGATIVE_TOLERANCE: float = 1e-8
_WEIGHT_SUM_TOLERANCE: float = 1e-6
_DOMAIN_TOLERANCE: float = 1e-6
_ZERO_VOLATILITY_THRESHOLD: float = 1e-12


# ---------------------------------------------------------------------------
# Result Containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, init=False)
class SimulatedLandmarkPortfolio:
    """Immutable representation of a landmark portfolio in a Monte Carlo simulation.

    Attributes:
        landmark_type: String identifier ('SIMULATED_GMV' or 'SIMULATED_MAX_SHARPE').
        weights: Optimal simulated asset weights (pd.Series indexed by asset labels).
        expected_return: Annualized expected return E[R_p].
        volatility: Annualized portfolio volatility sigma_p.
        variance: Annualized portfolio variance sigma_p^2.
        sharpe_ratio: Annualized Sharpe ratio (E[R_p] - r_f) / sigma_p.
        simulation_index: Integer 0-based index of this portfolio in the simulation array.
    """

    landmark_type: str
    expected_return: float
    volatility: float
    variance: float
    sharpe_ratio: float
    simulation_index: int
    weights: pd.Series
    _weights: pd.Series = field(repr=False)

    def __init__(
        self,
        landmark_type: str,
        expected_return: float,
        volatility: float,
        variance: float,
        sharpe_ratio: float,
        simulation_index: int,
        weights: pd.Series,
    ):
        object.__setattr__(self, "landmark_type", str(landmark_type))
        object.__setattr__(self, "expected_return", float(expected_return))
        object.__setattr__(self, "volatility", float(volatility))
        object.__setattr__(self, "variance", float(variance))
        object.__setattr__(self, "sharpe_ratio", float(sharpe_ratio))
        object.__setattr__(self, "simulation_index", int(simulation_index))
        object.__setattr__(self, "_weights", weights.copy())

    @property
    def weights(self) -> pd.Series:
        """Return a defensive copy of the portfolio weights Series."""
        return self._weights.copy()

    def __iter__(self) -> Iterator[float]:
        """Support tuple unpacking: vol, ret = landmark."""
        yield self.volatility
        yield self.expected_return

    def to_dict(self) -> Dict[str, Any]:
        """Convert landmark portfolio to JSON-serializable dictionary."""
        return {
            "landmark_type": self.landmark_type,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "variance": self.variance,
            "sharpe_ratio": self.sharpe_ratio,
            "simulation_index": self.simulation_index,
            "weights": {str(k): float(v) for k, v in self._weights.items()},
            "is_simulated_approximation": True,
        }


@dataclass(frozen=True, init=False)
class MonteCarloSimulationResult:
    """Immutable container for vectorized Monte Carlo portfolio simulation results.

    Defensively immutable: internal numpy arrays, pandas Series, and metadata
    cannot be mutated by callers through returned properties.

    Attributes:
        weights: 2D numpy array of shape (num_portfolios, num_assets).
        expected_returns: 1D numpy array of annualized expected returns of shape (num_portfolios,).
        volatilities: 1D numpy array of annualized volatilities of shape (num_portfolios,).
        variances: 1D numpy array of annualized variances of shape (num_portfolios,).
        sharpe_ratios: 1D numpy array of annualized Sharpe ratios of shape (num_portfolios,).
        simulated_gmv: Simulated landmark portfolio with minimum volatility.
        simulated_max_sharpe: Simulated landmark portfolio with maximum Sharpe ratio.
        asset_labels: Tuple of asset symbol names.
        num_portfolios: Number of simulated portfolios (K).
        num_assets: Number of assets (N).
        risk_free_rate: Annualized risk-free rate used for Sharpe ratios.
        random_seed: Optional integer random seed used for reproducibility.
        metadata: Deep-copied diagnostic dictionary.
    """

    simulated_gmv: SimulatedLandmarkPortfolio
    simulated_max_sharpe: SimulatedLandmarkPortfolio
    asset_labels: Tuple[str, ...]
    num_portfolios: int
    num_assets: int
    risk_free_rate: float
    random_seed: Optional[int]
    _weights: np.ndarray = field(repr=False)
    _expected_returns: np.ndarray = field(repr=False)
    _volatilities: np.ndarray = field(repr=False)
    _variances: np.ndarray = field(repr=False)
    _sharpe_ratios: np.ndarray = field(repr=False)
    _metadata: Dict[str, Any] = field(repr=False)

    def __init__(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        volatilities: np.ndarray,
        variances: np.ndarray,
        sharpe_ratios: np.ndarray,
        simulated_gmv: SimulatedLandmarkPortfolio,
        simulated_max_sharpe: SimulatedLandmarkPortfolio,
        asset_labels: Sequence[str],
        num_portfolios: int,
        num_assets: int,
        risk_free_rate: float,
        random_seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        object.__setattr__(self, "_weights", np.array(weights, copy=True, dtype=np.float64))
        object.__setattr__(self, "_expected_returns", np.array(expected_returns, copy=True, dtype=np.float64))
        object.__setattr__(self, "_volatilities", np.array(volatilities, copy=True, dtype=np.float64))
        object.__setattr__(self, "_variances", np.array(variances, copy=True, dtype=np.float64))
        object.__setattr__(self, "_sharpe_ratios", np.array(sharpe_ratios, copy=True, dtype=np.float64))
        object.__setattr__(self, "simulated_gmv", simulated_gmv)
        object.__setattr__(self, "simulated_max_sharpe", simulated_max_sharpe)
        object.__setattr__(self, "asset_labels", tuple(str(lbl) for lbl in asset_labels))
        object.__setattr__(self, "num_portfolios", int(num_portfolios))
        object.__setattr__(self, "num_assets", int(num_assets))
        object.__setattr__(self, "risk_free_rate", float(risk_free_rate))
        object.__setattr__(self, "random_seed", int(random_seed) if random_seed is not None else None)
        object.__setattr__(self, "_metadata", copy.deepcopy(metadata or {}))

    @property
    def weights(self) -> np.ndarray:
        """Return defensive copy of the (num_portfolios, num_assets) weight matrix."""
        return self._weights.copy()

    @property
    def weights_dataframe(self) -> pd.DataFrame:
        """Return portfolio weights as a pandas DataFrame with asset column names."""
        return pd.DataFrame(self._weights.copy(), columns=list(self.asset_labels))

    @property
    def expected_returns(self) -> np.ndarray:
        """Return defensive copy of the annualized expected returns array."""
        return self._expected_returns.copy()

    @property
    def volatilities(self) -> np.ndarray:
        """Return defensive copy of the annualized volatilities array."""
        return self._volatilities.copy()

    @property
    def variances(self) -> np.ndarray:
        """Return defensive copy of the annualized variances array."""
        return self._variances.copy()

    @property
    def sharpe_ratios(self) -> np.ndarray:
        """Return defensive copy of the annualized Sharpe ratios array."""
        return self._sharpe_ratios.copy()

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return defensive deep copy of metadata."""
        return copy.deepcopy(self._metadata)

    def __len__(self) -> int:
        """Return number of simulated portfolios."""
        return self.num_portfolios

    def to_dict(self) -> Dict[str, Any]:
        """Convert simulation result to JSON-serializable dictionary."""
        return {
            "num_portfolios": self.num_portfolios,
            "num_assets": self.num_assets,
            "risk_free_rate": self.risk_free_rate,
            "random_seed": self.random_seed,
            "asset_labels": list(self.asset_labels),
            "simulated_gmv": self.simulated_gmv.to_dict(),
            "simulated_max_sharpe": self.simulated_max_sharpe.to_dict(),
            "expected_returns": self._expected_returns.tolist(),
            "volatilities": self._volatilities.tolist(),
            "variances": self._variances.tolist(),
            "sharpe_ratios": self._sharpe_ratios.tolist(),
            "weights": self._weights.tolist(),
            "summary": {
                "min_expected_return": float(np.min(self._expected_returns)),
                "max_expected_return": float(np.max(self._expected_returns)),
                "min_volatility": float(np.min(self._volatilities)),
                "max_volatility": float(np.max(self._volatilities)),
                "min_sharpe_ratio": float(np.min(self._sharpe_ratios)),
                "max_sharpe_ratio": float(np.max(self._sharpe_ratios)),
            },
            "metadata": self.metadata,
        }


@dataclass(frozen=True, init=False)
class MonteCarloValidationResult:
    """Immutable validation report comparing Monte Carlo simulation against analytical frontier.

    Defensively immutable: internal details dictionary cannot be modified by callers.

    Attributes:
        is_valid: True if no simulated portfolios breach the analytical minimum-variance
            boundary beyond the allowed numerical tolerance.
        num_portfolios_tested: Number of simulated portfolios evaluated.
        num_breaches: Count of simulated points with volatility below analytical frontier.
        max_breach: Maximum negative violation delta (positive indicates breach magnitude).
        gmv_volatility_delta: Simulated GMV volatility minus analytical GMV volatility (should be >= -tol).
        tangency_sharpe_delta: Analytical Tangency Sharpe minus simulated Max Sharpe (should be >= -tol).
        tolerance: Numerical tolerance applied during cross-validation.
        details: Diagnostic dictionary.
    """

    is_valid: bool
    num_portfolios_tested: int
    num_breaches: int
    max_breach: float
    gmv_volatility_delta: float
    tangency_sharpe_delta: float
    tolerance: float
    details: Dict[str, Any]
    _details: Dict[str, Any] = field(repr=False)

    def __init__(
        self,
        is_valid: bool,
        num_portfolios_tested: int,
        num_breaches: int,
        max_breach: float,
        gmv_volatility_delta: float,
        tangency_sharpe_delta: float,
        tolerance: float,
        details: Optional[Dict[str, Any]] = None,
    ):
        object.__setattr__(self, "is_valid", bool(is_valid))
        object.__setattr__(self, "num_portfolios_tested", int(num_portfolios_tested))
        object.__setattr__(self, "num_breaches", int(num_breaches))
        object.__setattr__(self, "max_breach", float(max_breach))
        object.__setattr__(self, "gmv_volatility_delta", float(gmv_volatility_delta))
        object.__setattr__(self, "tangency_sharpe_delta", float(tangency_sharpe_delta))
        object.__setattr__(self, "tolerance", float(tolerance))
        object.__setattr__(self, "_details", copy.deepcopy(details or {}))

    @property
    def details(self) -> Dict[str, Any]:
        """Return defensive deep copy of details dictionary."""
        return copy.deepcopy(self._details)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation report to JSON-serializable dictionary."""
        return {
            "is_valid": self.is_valid,
            "num_portfolios_tested": self.num_portfolios_tested,
            "num_breaches": self.num_breaches,
            "max_breach": self.max_breach,
            "gmv_volatility_delta": self.gmv_volatility_delta,
            "tangency_sharpe_delta": self.tangency_sharpe_delta,
            "tolerance": self.tolerance,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Input Validation & Normalization Helpers
# ---------------------------------------------------------------------------

def _validate_and_normalize_covariance(
    covariance: object,
) -> Tuple[pd.DataFrame, pd.Index]:
    """Validate covariance matrix structure, square shape, finiteness, symmetry, and PSD."""
    if covariance is None:
        raise FinancialCalculationError(
            "covariance matrix cannot be None.",
            details={"type": "NoneType"},
        )

    if isinstance(covariance, pd.DataFrame):
        if covariance.empty:
            raise InsufficientDataError(
                "covariance DataFrame is empty.",
                details={"shape": covariance.shape},
            )
        if covariance.shape[0] != covariance.shape[1]:
            raise FinancialCalculationError(
                f"covariance DataFrame must be square (N x N), got shape {covariance.shape}.",
                details={"shape": covariance.shape},
            )
        if len(covariance) < 2:
            raise InsufficientDataError(
                f"covariance matrix must contain at least 2 assets, got {len(covariance)}.",
                details={"n_assets": len(covariance)},
            )

        if not covariance.index.equals(covariance.columns):
            raise FinancialCalculationError(
                "covariance DataFrame index and columns must match identically.",
                details={"index": list(covariance.index), "columns": list(covariance.columns)},
            )

        raw_vals = covariance.values
        if np.issubdtype(raw_vals.dtype, np.complexfloating):
            raise FinancialCalculationError(
                f"covariance contains non-numeric data with dtype {raw_vals.dtype} (complex numbers are not permitted).",
                details={"dtype": str(raw_vals.dtype)},
            )
        if not np.issubdtype(raw_vals.dtype, np.number) or np.issubdtype(raw_vals.dtype, np.bool_):
            raise FinancialCalculationError(
                f"covariance contains non-numeric data with dtype {raw_vals.dtype}.",
                details={"dtype": str(raw_vals.dtype)},
            )
        if not np.all(np.isfinite(raw_vals)):
            raise FinancialCalculationError(
                "covariance matrix contains NaN, infinite, or non-finite values.",
                details={"shape": covariance.shape},
            )

        # Symmetry check
        asymmetry = float(np.max(np.abs(raw_vals - raw_vals.T)))
        if asymmetry > _SYMMETRY_TOLERANCE:
            raise FinancialCalculationError(
                f"covariance matrix is materially asymmetric (max asymmetry={asymmetry:.2e} > tolerance={_SYMMETRY_TOLERANCE:.2e}).",
                details={"max_asymmetry": asymmetry, "tolerance": _SYMMETRY_TOLERANCE},
            )

        # Diagonal variance positivity check
        variances = np.diag(raw_vals)
        if not np.all(variances > 0.0):
            invalid_vars = {
                str(col): float(val)
                for col, val in zip(covariance.columns, variances)
                if val <= 0.0 or not np.isfinite(val)
            }
            raise FinancialCalculationError(
                "All diagonal variance elements in covariance matrix must be finite and strictly positive (> 0). Negative or zero variance is not permitted.",
                details={"invalid_diagonal_variances": invalid_vars},
            )

        # Positive semidefinite (PSD) check
        eigvals = np.linalg.eigvalsh(raw_vals)
        min_eig = float(np.min(eigvals))
        if min_eig < _PSD_EIGENVALUE_TOLERANCE:
            raise FinancialCalculationError(
                f"covariance matrix is not positive semidefinite: minimum eigenvalue ({min_eig:.2e}) < {_PSD_EIGENVALUE_TOLERANCE:.2e}.",
                details={"min_eigenvalue": min_eig, "eigenvalue_tolerance": _PSD_EIGENVALUE_TOLERANCE},
            )

        asset_labels = covariance.columns
        cov_df = covariance.astype(float)
        return cov_df, asset_labels

    elif isinstance(covariance, np.ndarray):
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise FinancialCalculationError(
                f"covariance array must be 2D square (N x N), got shape {covariance.shape}.",
                details={"shape": covariance.shape},
            )
        if covariance.shape[0] < 2:
            raise InsufficientDataError(
                f"covariance array must contain at least 2 assets, got {covariance.shape[0]}.",
                details={"n_assets": covariance.shape[0]},
            )
        if np.issubdtype(covariance.dtype, np.complexfloating):
            raise FinancialCalculationError(
                "covariance contains complex numbers.",
                details={"dtype": str(covariance.dtype)},
            )
        if not np.issubdtype(covariance.dtype, np.number) or np.issubdtype(covariance.dtype, np.bool_):
            raise FinancialCalculationError(
                f"covariance array contains non-numeric data with dtype {covariance.dtype}.",
                details={"dtype": str(covariance.dtype)},
            )
        if not np.all(np.isfinite(covariance)):
            raise FinancialCalculationError(
                "covariance array contains NaN, infinite, or non-finite values.",
                details={"shape": covariance.shape},
            )

        # Symmetry check
        asymmetry = float(np.max(np.abs(covariance - covariance.T)))
        if asymmetry > _SYMMETRY_TOLERANCE:
            raise FinancialCalculationError(
                f"covariance array is materially asymmetric (max asymmetry={asymmetry:.2e} > tolerance={_SYMMETRY_TOLERANCE:.2e}).",
                details={"max_asymmetry": asymmetry, "tolerance": _SYMMETRY_TOLERANCE},
            )

        # Diagonal variance positivity check
        variances = np.diag(covariance)
        if not np.all(variances > 0.0):
            invalid_vars = {
                f"asset_{i}": float(val)
                for i, val in enumerate(variances)
                if val <= 0.0 or not np.isfinite(val)
            }
            raise FinancialCalculationError(
                "All diagonal variance elements in covariance array must be finite and strictly positive (> 0). Negative or zero variance is not permitted.",
                details={"invalid_diagonal_variances": invalid_vars},
            )

        # Positive semidefinite (PSD) check
        eigvals = np.linalg.eigvalsh(covariance)
        min_eig = float(np.min(eigvals))
        if min_eig < _PSD_EIGENVALUE_TOLERANCE:
            raise FinancialCalculationError(
                f"covariance array is not positive semidefinite: minimum eigenvalue ({min_eig:.2e}) < {_PSD_EIGENVALUE_TOLERANCE:.2e}.",
                details={"min_eigenvalue": min_eig, "eigenvalue_tolerance": _PSD_EIGENVALUE_TOLERANCE},
            )

        n = covariance.shape[0]
        asset_labels = pd.Index([f"asset_{i}" for i in range(n)])
        cov_df = pd.DataFrame(covariance.astype(float), index=asset_labels, columns=asset_labels)
        return cov_df, asset_labels

    else:
        raise FinancialCalculationError(
            f"covariance must be a pandas DataFrame or 2D numpy array, got {type(covariance).__name__}.",
            details={"type": type(covariance).__name__},
        )


def _validate_and_normalize_expected_returns(
    expected_returns: object,
    asset_labels: pd.Index,
) -> pd.Series:
    """Validate expected returns vector and align strictly with asset labels."""
    if expected_returns is None:
        raise FinancialCalculationError(
            "expected_returns vector cannot be None.",
            details={"type": "NoneType"},
        )

    if isinstance(expected_returns, pd.Series):
        if expected_returns.empty:
            raise InsufficientDataError(
                "expected_returns Series is empty.",
                details={"len": 0},
            )
        if len(expected_returns) != len(asset_labels):
            raise FinancialCalculationError(
                f"expected_returns length ({len(expected_returns)}) does not match asset count ({len(asset_labels)}).",
                details={"expected_returns_len": len(expected_returns), "asset_count": len(asset_labels)},
            )
        if not set(asset_labels).issubset(set(expected_returns.index)):
            missing = sorted(set(asset_labels) - set(expected_returns.index))
            raise FinancialCalculationError(
                f"expected_returns Series is missing asset labels: {missing}.",
                details={"missing": missing},
            )
        raw_vals = expected_returns.values
        if not np.issubdtype(raw_vals.dtype, np.number) or np.issubdtype(raw_vals.dtype, np.bool_):
            raise FinancialCalculationError(
                f"expected_returns contains non-numeric data with dtype {raw_vals.dtype}.",
                details={"dtype": str(raw_vals.dtype)},
            )
        if not np.all(np.isfinite(raw_vals)):
            raise FinancialCalculationError(
                "expected_returns Series contains NaN, infinite, or non-finite values.",
                details={"values": expected_returns.to_dict()},
            )

        aligned = expected_returns.loc[asset_labels].astype(float)
        return aligned

    elif isinstance(expected_returns, (list, tuple, np.ndarray)):
        arr = np.asarray(expected_returns)
        if arr.ndim != 1 or len(arr) != len(asset_labels):
            raise FinancialCalculationError(
                f"expected_returns sequence must be 1D with length {len(asset_labels)}, got shape {arr.shape}.",
                details={"shape": arr.shape, "expected_len": len(asset_labels)},
            )
        if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
            raise FinancialCalculationError(
                f"expected_returns sequence contains non-numeric data with dtype {arr.dtype}.",
                details={"dtype": str(arr.dtype)},
            )
        if not np.all(np.isfinite(arr)):
            raise FinancialCalculationError(
                "expected_returns sequence contains NaN, infinite, or non-finite values.",
                details={"len": len(arr)},
            )

        return pd.Series(arr.astype(float), index=asset_labels)

    else:
        raise FinancialCalculationError(
            f"expected_returns must be a pandas Series or 1D array/sequence, got {type(expected_returns).__name__}.",
            details={"type": type(expected_returns).__name__},
        )


def _validate_risk_free_rate(risk_free_rate: object) -> float:
    """Validate that risk-free rate is a finite real numeric scalar."""
    if isinstance(risk_free_rate, (bool, np.bool_)):
        raise FinancialCalculationError(
            f"risk_free_rate must be a numeric float, got bool ({risk_free_rate!r}).",
            details={"type": "bool", "value": risk_free_rate},
        )
    if isinstance(risk_free_rate, (complex, np.complexfloating)):
        raise FinancialCalculationError(
            f"risk_free_rate must be a real numeric value, got complex ({risk_free_rate!r}).",
            details={"type": "complex", "value": str(risk_free_rate)},
        )
    if not isinstance(risk_free_rate, (int, float, np.integer, np.floating)):
        raise FinancialCalculationError(
            f"risk_free_rate must be a numeric float or int, got {type(risk_free_rate).__name__}.",
            details={"type": type(risk_free_rate).__name__},
        )
    rf = float(risk_free_rate)
    if not np.isfinite(rf):
        raise FinancialCalculationError(
            f"risk_free_rate must be finite, got {risk_free_rate!r}.",
            details={"value": risk_free_rate},
        )
    return rf


def _resolve_random_generator(
    random_seed: Optional[object],
    generator: Optional[object],
) -> Tuple[np.random.Generator, Optional[int]]:
    """Validate and resolve random seed and numpy Generator."""
    if random_seed is not None and generator is not None:
        raise FinancialCalculationError(
            "Cannot specify both 'random_seed' and 'generator' simultaneously. Pass only one.",
            details={"has_seed": True, "has_generator": True},
        )

    if generator is not None:
        if not isinstance(generator, np.random.Generator):
            raise FinancialCalculationError(
                f"generator must be an instance of numpy.random.Generator, got {type(generator).__name__}.",
                details={"type": type(generator).__name__},
            )
        return generator, None

    if random_seed is not None:
        if isinstance(random_seed, (bool, np.bool_)):
            raise FinancialCalculationError(
                f"random_seed must be an integer, got bool ({random_seed!r}).",
                details={"type": "bool", "value": random_seed},
            )
        if not isinstance(random_seed, (int, np.integer)):
            raise FinancialCalculationError(
                f"random_seed must be an integer, got {type(random_seed).__name__}.",
                details={"type": type(random_seed).__name__},
            )
        seed_val = int(random_seed)
        if seed_val < 0 or seed_val > 2**32 - 1:
            raise FinancialCalculationError(
                f"random_seed must be an integer between 0 and 2^32 - 1, got {seed_val}.",
                details={"value": seed_val},
            )
        return np.random.default_rng(seed_val), seed_val

    # Unseeded generator
    return np.random.default_rng(), None


def _validate_num_portfolios(num_portfolios: object) -> int:
    """Validate number of simulation iterations K."""
    if isinstance(num_portfolios, (bool, np.bool_)):
        raise FinancialCalculationError(
            f"num_portfolios must be a positive integer, got bool ({num_portfolios!r}).",
            details={"type": "bool", "value": num_portfolios},
        )
    if not isinstance(num_portfolios, (int, np.integer)):
        raise FinancialCalculationError(
            f"num_portfolios must be an integer, got {type(num_portfolios).__name__}.",
            details={"type": type(num_portfolios).__name__},
        )
    k = int(num_portfolios)
    if k < 1:
        raise InsufficientDataError(
            f"num_portfolios must be at least 1, got {k}.",
            details={"num_portfolios": k},
        )
    return k


# ---------------------------------------------------------------------------
# Core Simulation Engine
# ---------------------------------------------------------------------------

def simulate_monte_carlo_portfolios(
    expected_returns: Union[pd.Series, Sequence[float], np.ndarray],
    covariance: Union[pd.DataFrame, np.ndarray],
    num_portfolios: int = DEFAULT_NUM_PORTFOLIOS,
    *,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    random_seed: Optional[int] = None,
    generator: Optional[np.random.Generator] = None,
) -> MonteCarloSimulationResult:
    """Generate vectorized Monte Carlo portfolios on the standard simplex via Dirichlet sampling.

    For N assets and K simulations:
        1. Draws weights W ~ Dirichlet(alpha = 1.0) of shape (K, N).
        2. Vectorially computes annualized portfolio returns:
               returns = W @ mu
        3. Vectorially computes annualized portfolio variances:
               variances = np.einsum("ij,jk,ik->i", W, covariance, W)
        4. Computes annualized volatilities:
               volatilities = sqrt(variances)
        5. Computes annualized Sharpe ratios:
               sharpe_ratios = (returns - risk_free_rate) / volatilities
        6. Identifies landmark empirical portfolios: Simulated GMV and Simulated Max Sharpe.

    Parameters
    ----------
    expected_returns:
        Annualized expected returns vector (pd.Series or 1D array of length N).
    covariance:
        Annualized covariance matrix (N x N pd.DataFrame or 2D array).
    num_portfolios:
        Number of simulated portfolios K (defaults to 10,000). Must be >= 1.
    risk_free_rate:
        Annualized risk-free rate used for Sharpe calculation (defaults to 0.065, or 6.5%).
    random_seed:
        Optional non-negative integer seed for bit-for-bit reproducibility.
    generator:
        Optional numpy.random.Generator instance (mutually exclusive with random_seed).

    Returns
    -------
    MonteCarloSimulationResult
        Defensively immutable container holding all simulated weights and metrics.

    Raises
    ------
    FinancialCalculationError
        Invalid dimensions, non-square/asymmetric/non-PSD covariance, non-finite data,
        or simultaneous seed + generator input.
    InsufficientDataError
        Fewer than 2 assets or fewer than 1 portfolio.
    """
    # 1. Input normalization and strict validation
    cov_df, asset_labels = _validate_and_normalize_covariance(covariance)
    exp_ret_series = _validate_and_normalize_expected_returns(expected_returns, asset_labels)
    rf = _validate_risk_free_rate(risk_free_rate)
    k = _validate_num_portfolios(num_portfolios)
    rng, seed_used = _resolve_random_generator(random_seed, generator)

    n_assets = len(asset_labels)
    cov_mat = cov_df.values.astype(np.float64)
    mu_vec = exp_ret_series.values.astype(np.float64)

    # 2. Vectorized simplex portfolio generation via Dirichlet(alpha=1.0)
    alpha = np.ones(n_assets, dtype=np.float64)
    weights = rng.dirichlet(alpha, size=k).astype(np.float64)

    # Validate generated weights feasibility defensively
    if not np.all(np.isfinite(weights)):
        raise FinancialCalculationError(
            "Monte Carlo weight generation produced non-finite values.",
            details={"shape": list(weights.shape)},
        )
    if (weights < -1e-12).any() or (weights > 1.0 + 1e-12).any():
        raise FinancialCalculationError(
            f"Generated weights violate long-only bounds [0, 1]: min={float(np.min(weights)):.6f}, max={float(np.max(weights)):.6f}.",
            details={"min_weight": float(np.min(weights)), "max_weight": float(np.max(weights))},
        )
    weight_sums = np.sum(weights, axis=1)
    if not np.allclose(weight_sums, 1.0, atol=_WEIGHT_SUM_TOLERANCE):
        max_sum_dev = float(np.max(np.abs(weight_sums - 1.0)))
        raise FinancialCalculationError(
            f"Generated weights deviate from budget sum 1.0 beyond tolerance: max deviation={max_sum_dev:.2e}.",
            details={"max_deviation": max_sum_dev, "tolerance": _WEIGHT_SUM_TOLERANCE},
        )

    # 3. Vectorized metric evaluation
    # Expected returns: W @ mu
    portfolio_returns = weights @ mu_vec

    # Variances: einsum("ij,jk,ik->i", W, cov, W) evaluates sum_j,k W_ij * Cov_jk * W_ik
    portfolio_variances = np.einsum("ij,jk,ik->i", weights, cov_mat, weights)

    # Validate non-negative portfolio variance
    materially_negative_mask = portfolio_variances < -_VARIANCE_NON_NEGATIVE_TOLERANCE
    if np.any(materially_negative_mask):
        min_neg_var = float(np.min(portfolio_variances[materially_negative_mask]))
        raise FinancialCalculationError(
            f"Computed portfolio variance is materially negative (min variance={min_neg_var:.2e} < -{_VARIANCE_NON_NEGATIVE_TOLERANCE:.2e}). "
            "Covariance matrix is materially non-positive semidefinite.",
            details={
                "min_variance": min_neg_var,
                "tolerance": _VARIANCE_NON_NEGATIVE_TOLERANCE,
                "negative_count": int(np.sum(materially_negative_mask)),
            },
        )

    # Clamp tiny numerical negatives within tolerance [-1e-8, 0.0) to 0.0 due to floating-point rounding
    portfolio_variances = np.maximum(portfolio_variances, 0.0)

    # Volatilities: sqrt(variances)
    portfolio_volatilities = np.sqrt(portfolio_variances)

    # Sharpe ratio with explicit zero-volatility handling
    sharpe_ratios = np.zeros(k, dtype=np.float64)
    zero_vol_mask = portfolio_volatilities <= _ZERO_VOLATILITY_THRESHOLD

    if np.any(zero_vol_mask):
        excess_returns = portfolio_returns[zero_vol_mask] - rf
        non_zero_excess = np.abs(excess_returns) > _ZERO_VOLATILITY_THRESHOLD
        if np.any(non_zero_excess):
            raise FinancialCalculationError(
                "Encountered simulated portfolio with zero volatility and non-zero excess return; Sharpe ratio is undefined.",
                details={"zero_volatility_count": int(np.sum(zero_vol_mask))},
            )
        sharpe_ratios[zero_vol_mask] = 0.0

    non_zero_mask = ~zero_vol_mask
    sharpe_ratios[non_zero_mask] = (portfolio_returns[non_zero_mask] - rf) / portfolio_volatilities[non_zero_mask]

    # Validate output finiteness
    if not np.all(np.isfinite(portfolio_returns)) or not np.all(np.isfinite(portfolio_volatilities)) or not np.all(np.isfinite(sharpe_ratios)):
        raise FinancialCalculationError(
            "Computed simulation metrics contain non-finite values.",
            details={"has_inf": True},
        )

    # 4. Identify landmark empirical portfolios
    gmv_idx = int(np.argmin(portfolio_volatilities))
    simulated_gmv = SimulatedLandmarkPortfolio(
        landmark_type="SIMULATED_GMV",
        expected_return=float(portfolio_returns[gmv_idx]),
        volatility=float(portfolio_volatilities[gmv_idx]),
        variance=float(portfolio_variances[gmv_idx]),
        sharpe_ratio=float(sharpe_ratios[gmv_idx]),
        simulation_index=gmv_idx,
        weights=pd.Series(weights[gmv_idx], index=asset_labels),
    )

    max_sharpe_idx = int(np.argmax(sharpe_ratios))
    simulated_max_sharpe = SimulatedLandmarkPortfolio(
        landmark_type="SIMULATED_MAX_SHARPE",
        expected_return=float(portfolio_returns[max_sharpe_idx]),
        volatility=float(portfolio_volatilities[max_sharpe_idx]),
        variance=float(portfolio_variances[max_sharpe_idx]),
        sharpe_ratio=float(sharpe_ratios[max_sharpe_idx]),
        simulation_index=max_sharpe_idx,
        weights=pd.Series(weights[max_sharpe_idx], index=asset_labels),
    )

    metadata = {
        "num_portfolios": k,
        "num_assets": n_assets,
        "risk_free_rate": rf,
        "random_seed": seed_used,
        "is_reproducible": seed_used is not None or generator is not None,
        "gmv_index": gmv_idx,
        "max_sharpe_index": max_sharpe_idx,
    }

    return MonteCarloSimulationResult(
        weights=weights,
        expected_returns=portfolio_returns,
        volatilities=portfolio_volatilities,
        variances=portfolio_variances,
        sharpe_ratios=sharpe_ratios,
        simulated_gmv=simulated_gmv,
        simulated_max_sharpe=simulated_max_sharpe,
        asset_labels=list(asset_labels),
        num_portfolios=k,
        num_assets=n_assets,
        risk_free_rate=rf,
        random_seed=seed_used,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Object-Oriented Simulation Class Interface
# ---------------------------------------------------------------------------

class MonteCarloSimulator:
    """Object-oriented interface for Monte Carlo portfolio simulation.

    Parameters
    ----------
    risk_free_rate:
        Annualized risk-free rate used for Sharpe ratio calculation (default 0.065).
    random_seed:
        Optional integer seed for reproducible simulations.
    generator:
        Optional pre-configured numpy.random.Generator.
    """

    def __init__(
        self,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        random_seed: Optional[int] = None,
        generator: Optional[np.random.Generator] = None,
    ):
        self.risk_free_rate = _validate_risk_free_rate(risk_free_rate)
        self._rng, self.random_seed = _resolve_random_generator(random_seed, generator)

    def simulate(
        self,
        expected_returns: Union[pd.Series, Sequence[float], np.ndarray],
        covariance: Union[pd.DataFrame, np.ndarray],
        num_portfolios: int = DEFAULT_NUM_PORTFOLIOS,
    ) -> MonteCarloSimulationResult:
        """Run Monte Carlo simulation using the configured generator and risk-free rate."""
        return simulate_monte_carlo_portfolios(
            expected_returns=expected_returns,
            covariance=covariance,
            num_portfolios=num_portfolios,
            risk_free_rate=self.risk_free_rate,
            generator=self._rng,
        )


# ---------------------------------------------------------------------------
# Analytical Cross-Validation Helper
# ---------------------------------------------------------------------------

def validate_monte_carlo_against_frontier(
    simulation_result: MonteCarloSimulationResult,
    frontier_result: EfficientFrontierResult,
    *,
    volatility_tolerance: float = 1e-4,
) -> MonteCarloValidationResult:
    """Cross-validate simulated portfolios against an analytical Efficient Frontier.

    Mathematical Invariant:
        In standard mean-variance theory, the analytical minimum-variance frontier
        forms the absolute lower volatility boundary for any target expected return.
        Consequently, all feasible long-only portfolios must satisfy:
            sigma_{simulated}(mu) >= sigma_{analytical_frontier}(mu) - tolerance

        Comparison is partitioned into two branches around the analytical GMV vertex (mu_GMV):
        1. Upper (efficient) branch (mu >= mu_GMV):
           Evaluated against the interpolated upper minimum-variance frontier.
        2. Lower (inefficient) branch (mu < mu_GMV):
           If inefficient portfolios are present in the frontier result, evaluated against
           the interpolated lower minimum-variance frontier.
           If insufficient lower-branch points are available, evaluated against the global
           GMV minimum volatility floor: sigma_{simulated} >= sigma_{analytical_GMV} - tolerance.
        3. Global boundary:
           No simulated portfolio may violate the analytical GMV volatility floor
           regardless of its expected return.

    Parameters
    ----------
    simulation_result:
        MonteCarloSimulationResult from simulate_monte_carlo_portfolios.
    frontier_result:
        Successfully converged EfficientFrontierResult from compute_efficient_frontier.
    volatility_tolerance:
        Maximum allowable numerical breach below the analytical frontier (default 1e-4).

    Returns
    -------
    MonteCarloValidationResult
        Validation report detailing whether the analytical boundary was respected.

    Raises
    ------
    FinancialCalculationError
        If simulation_result or frontier_result are invalid types, if volatility_tolerance
        is non-numeric, non-finite or negative, or if the analytical frontier solver
        failed or executed a fallback.
    """
    # 1. Validate volatility_tolerance
    if isinstance(volatility_tolerance, (bool, np.bool_)):
        raise FinancialCalculationError(
            f"volatility_tolerance must be a numeric float, got bool ({volatility_tolerance!r}).",
            details={"type": "bool", "value": volatility_tolerance},
        )
    if not isinstance(volatility_tolerance, (int, float, np.integer, np.floating)):
        raise FinancialCalculationError(
            f"volatility_tolerance must be a numeric float, got {type(volatility_tolerance).__name__}.",
            details={"type": type(volatility_tolerance).__name__},
        )
    vol_tol = float(volatility_tolerance)
    if not np.isfinite(vol_tol) or vol_tol < 0.0:
        raise FinancialCalculationError(
            f"volatility_tolerance must be finite and non-negative, got {volatility_tolerance!r}.",
            details={"value": volatility_tolerance},
        )

    # 2. Type validation
    if not isinstance(simulation_result, MonteCarloSimulationResult):
        raise FinancialCalculationError(
            f"simulation_result must be a MonteCarloSimulationResult, got {type(simulation_result).__name__}.",
            details={"type": type(simulation_result).__name__},
        )
    if not isinstance(frontier_result, EfficientFrontierResult):
        raise FinancialCalculationError(
            f"frontier_result must be an EfficientFrontierResult, got {type(frontier_result).__name__}.",
            details={"type": type(frontier_result).__name__},
        )

    # 3. Solver convergence & classification validation
    if not frontier_result.success or frontier_result.fallback_executed or not frontier_result.classification_valid:
        raise FinancialCalculationError(
            f"Analytical frontier result is unconverged or executed a fallback: "
            f"success={frontier_result.success}, fallback_executed={frontier_result.fallback_executed}, "
            f"classification_valid={frontier_result.classification_valid}. "
            "Cross-validation cannot be performed against an invalid or fallback analytical reference.",
            details={
                "success": frontier_result.success,
                "fallback_executed": frontier_result.fallback_executed,
                "classification_valid": frontier_result.classification_valid,
            },
        )

    if frontier_result.gmv_portfolio is None or not frontier_result.gmv_portfolio.success or frontier_result.gmv_portfolio.fallback_executed:
        raise FinancialCalculationError(
            "Analytical frontier result lacks a converged GMV reference portfolio.",
            details={"has_gmv": frontier_result.gmv_portfolio is not None},
        )

    # 4. Extract GMV anchor point
    analytical_gmv_vol = float(frontier_result.gmv_portfolio.volatility)
    sim_gmv_vol = float(simulation_result.simulated_gmv.volatility)
    gmv_vol_delta = sim_gmv_vol - analytical_gmv_vol

    gmv_expected_return_raw = frontier_result.metadata.get("gmv_expected_return")
    if gmv_expected_return_raw is None or not np.isfinite(float(gmv_expected_return_raw)):
        raise FinancialCalculationError(
            "Analytical frontier metadata lacks a valid finite 'gmv_expected_return'.",
            details={"gmv_expected_return": gmv_expected_return_raw},
        )
    gmv_ret = float(gmv_expected_return_raw)

    # Helper function to construct, deduplicate, and sort a branch interpolation grid
    def _build_branch_grid(
        portfolios: Sequence[FrontierPortfolio],
        anchor_return: float,
        anchor_volatility: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pts = [(anchor_return, anchor_volatility)]
        for p in portfolios:
            if p.success and not p.fallback_executed:
                pts.append((float(p.expected_return), float(p.volatility)))

        # Sort by expected return ascending
        pts.sort(key=lambda item: item[0])

        # Deduplicate points with identical / near-identical returns (< 1e-10)
        # Keep minimum volatility for a given return level (conservative boundary)
        deduped: List[Tuple[float, float]] = []
        for r, v in pts:
            if not deduped:
                deduped.append((r, v))
            else:
                last_r, last_v = deduped[-1]
                if abs(r - last_r) <= 1e-10:
                    deduped[-1] = (last_r, min(last_v, v))
                else:
                    deduped.append((r, v))

        rets_arr = np.array([pt[0] for pt in deduped], dtype=np.float64)
        vols_arr = np.array([pt[1] for pt in deduped], dtype=np.float64)
        return rets_arr, vols_arr

    # 5. Build Upper Branch Grid
    eff_ports = frontier_result.efficient_portfolios
    if not eff_ports:
        raise FinancialCalculationError(
            "Analytical frontier has no efficient portfolios to evaluate against.",
            details={"efficient_portfolios_count": 0},
        )
    rets_upper, vols_upper = _build_branch_grid(eff_ports, gmv_ret, analytical_gmv_vol)
    if len(rets_upper) < 2:
        raise FinancialCalculationError(
            f"Analytical frontier upper branch has insufficient points ({len(rets_upper)}) to interpolate.",
            details={"point_count": len(rets_upper)},
        )
    min_upper_ret = float(rets_upper[0])
    max_upper_ret = float(rets_upper[-1])

    # 6. Build Lower Branch Grid
    ineff_ports = frontier_result.inefficient_portfolios
    if ineff_ports:
        rets_lower, vols_lower = _build_branch_grid(ineff_ports, gmv_ret, analytical_gmv_vol)
        lower_branch_interpolated = bool(len(rets_lower) >= 2)
        min_lower_ret: Optional[float] = float(rets_lower[0]) if lower_branch_interpolated else None
        max_lower_ret: Optional[float] = float(rets_lower[-1]) if lower_branch_interpolated else None
    else:
        rets_lower = np.array([gmv_ret], dtype=np.float64)
        vols_lower = np.array([analytical_gmv_vol], dtype=np.float64)
        lower_branch_interpolated = False
        min_lower_ret = None
        max_lower_ret = None

    # 7. Evaluate simulated portfolios against minimum-variance boundaries
    sim_rets = simulation_result.expected_returns
    sim_vols = simulation_result.volatilities
    k = len(sim_rets)

    min_feasible_vols = np.full(k, analytical_gmv_vol, dtype=np.float64)

    # Upper branch evaluation: portfolios with return >= gmv_ret
    upper_mask = sim_rets >= gmv_ret
    if np.any(upper_mask):
        upper_sim_rets = sim_rets[upper_mask]
        upper_out_of_domain = upper_sim_rets > (max_upper_ret + _DOMAIN_TOLERANCE)
        if np.any(upper_out_of_domain):
            max_sim_ret = float(np.max(upper_sim_rets))
            raise FinancialCalculationError(
                f"Monte Carlo simulation contains portfolios with expected return ({max_sim_ret:.6f}) "
                f"exceeding analytical frontier upper-branch domain maximum ({max_upper_ret:.6f}) "
                f"beyond tolerance ({_DOMAIN_TOLERANCE:.2e}). "
                "Analytical frontier does not cover the complete upper-branch return domain of simulated portfolios.",
                details={
                    "simulated_max_return": max_sim_ret,
                    "upper_branch_max_return": max_upper_ret,
                    "domain_tolerance": _DOMAIN_TOLERANCE,
                    "out_of_domain_count": int(np.sum(upper_out_of_domain)),
                },
            )

        # Clip within numerical tolerance [min_upper_ret, max_upper_ret] for clean interpolation without extrapolation
        eval_upper_rets = np.clip(upper_sim_rets, min_upper_ret, max_upper_ret)
        upper_interp = np.interp(eval_upper_rets, rets_upper, vols_upper)
        min_feasible_vols[upper_mask] = np.maximum(min_feasible_vols[upper_mask], upper_interp)

    # Lower branch evaluation: portfolios with return < gmv_ret
    lower_mask = ~upper_mask
    if np.any(lower_mask):
        if lower_branch_interpolated:
            assert min_lower_ret is not None and max_lower_ret is not None
            lower_sim_rets = sim_rets[lower_mask]
            lower_out_of_domain = lower_sim_rets < (min_lower_ret - _DOMAIN_TOLERANCE)
            if np.any(lower_out_of_domain):
                min_sim_ret = float(np.min(lower_sim_rets))
                raise FinancialCalculationError(
                    f"Monte Carlo simulation contains portfolios with expected return ({min_sim_ret:.6f}) "
                    f"below analytical frontier lower-branch domain minimum ({min_lower_ret:.6f}) "
                    f"beyond tolerance ({_DOMAIN_TOLERANCE:.2e}). "
                    "Analytical frontier does not cover the complete lower-branch return domain of simulated portfolios.",
                    details={
                        "simulated_min_return": min_sim_ret,
                        "lower_branch_min_return": min_lower_ret,
                        "domain_tolerance": _DOMAIN_TOLERANCE,
                        "out_of_domain_count": int(np.sum(lower_out_of_domain)),
                    },
                )

            # Clip within numerical tolerance [min_lower_ret, max_lower_ret] for clean interpolation without extrapolation
            eval_lower_rets = np.clip(lower_sim_rets, min_lower_ret, max_lower_ret)
            lower_interp = np.interp(eval_lower_rets, rets_lower, vols_lower)
            min_feasible_vols[lower_mask] = np.maximum(min_feasible_vols[lower_mask], lower_interp)
        else:
            # Fallback: analytical frontier lacks lower branch; floor against analytical GMV volatility
            min_feasible_vols[lower_mask] = analytical_gmv_vol

    # Calculate violations
    vol_deltas = min_feasible_vols - sim_vols  # Positive value indicates breach
    breach_mask = vol_deltas > vol_tol
    breach_count = int(np.sum(breach_mask))
    max_breach = float(np.max(vol_deltas[breach_mask])) if breach_count > 0 else 0.0

    upper_breaches = int(np.sum(breach_mask & upper_mask))
    lower_breaches = int(np.sum(breach_mask & lower_mask))
    gmv_breaches = int(np.sum((analytical_gmv_vol - sim_vols) > vol_tol))

    # Tangency Sharpe delta if available in metadata
    tangency_sharpe_delta = 0.0
    if "tangency_portfolio" in frontier_result.metadata and frontier_result.metadata["tangency_portfolio"] is not None:
        tang_data = frontier_result.metadata["tangency_portfolio"]
        if isinstance(tang_data, dict) and "sharpe_ratio" in tang_data:
            analytical_sr = float(tang_data["sharpe_ratio"])
            sim_max_sr = float(simulation_result.simulated_max_sharpe.sharpe_ratio)
            tangency_sharpe_delta = analytical_sr - sim_max_sr

    is_valid = breach_count == 0

    return MonteCarloValidationResult(
        is_valid=is_valid,
        num_portfolios_tested=k,
        num_breaches=breach_count,
        max_breach=max_breach,
        gmv_volatility_delta=gmv_vol_delta,
        tangency_sharpe_delta=tangency_sharpe_delta,
        tolerance=vol_tol,
        details={
            "analytical_gmv_volatility": analytical_gmv_vol,
            "analytical_gmv_return": gmv_ret,
            "simulated_gmv_volatility": sim_gmv_vol,
            "simulated_gmv_return": float(simulation_result.simulated_gmv.expected_return),
            "min_upper_return": min_upper_ret,
            "max_upper_return": max_upper_ret,
            "min_lower_return": min_lower_ret,
            "max_lower_return": max_lower_ret,
            "upper_branch_point_count": len(rets_upper),
            "lower_branch_point_count": len(rets_lower) if lower_branch_interpolated else 0,
            "lower_branch_interpolated": lower_branch_interpolated,
            "lower_branch_validation_guarantee": (
                "interpolated_minimum_variance_boundary"
                if lower_branch_interpolated
                else "gmv_floor_only (analytical frontier lacks lower branch; boundary cannot be verified for returns below GMV)"
            ),
            "lower_branch_breach_count": lower_breaches,
            "upper_branch_breach_count": upper_breaches,
            "gmv_floor_breach_count": gmv_breaches,
        },
    )
