"""Quantamental AI - Higham Positive-Definite Covariance Regularization.

Implements Phase 2.2, Step 3: Higham (2002) nearest-correlation-matrix
projection and positive-definite covariance regularization.

Mathematical Reference (FINANCIAL_METHODOLOGY.md §4.2.2 & ARCHITECTURE.md §4.2):
    1. Pre-standardization variance validation:
       Validate that all diagonal elements are strictly positive:
           σ_i² = Σ_{ii} > 0 and isfinite(σ_i²)  ∀ i ∈ {1, ..., N}
    2. Standardization to correlation scale:
           D^{1/2} = diag(σ_1, ..., σ_N)
           C = D^{-1/2} Σ D^{-1/2},  C_{ij} = Σ_{ij} / (σ_i σ_j)
    3. Higham alternating projections with Dykstra correction:
       Iteratively projects between the positive-semidefinite cone S_+
       and the affine subspace of symmetric matrices with unit diagonal U:
           C_proj = Higham(C)
    4. Controlled positive-definite regularization:
       Spectral decomposition of C_proj:
           C_proj = V Λ Vᵀ,  Λ = diag(λ_1, ..., λ_N)
       Apply eigenvalue floor:
           Λ̃ = diag(max(λ_i, ε_floor)),  ε_floor = 10⁻⁷ (settings.MIN_EIGENVALUE_FLOOR)
           C̃ = V Λ̃ Vᵀ
       Renormalize diagonal to restore exact unit diagonals:
           S_diag = diag(C̃)
           C_reg = S_diag^{-1/2} C̃ S_diag^{-1/2}
       NOTE: Eigenvalue flooring followed by diagonal renormalization can slightly
       perturb the final minimum eigenvalue of C_reg relative to ε_floor.
    5. Rescaling to covariance units:
           Σ_reg = D^{1/2} C_reg D^{1/2}
       Original asset standard deviations σ_i are preserved exactly.
       NOTE: A correlation eigenvalue floor of 10⁻⁷ does NOT imply an absolute
       minimum eigenvalue of 10⁻⁷ for Σ_reg, since scaling by asset variances
       multiplies eigenvalues by the scale of asset volatilities.
    6. Post-regularization validation:
       Validates finiteness, symmetry, strict positive definiteness (λ_min > 0),
       condition number (κ < 10⁶), and variance preservation.

Canonical design references:
    - Higham, N. J. (2002). "Computing the nearest correlation matrix—a problem
      from finance". IMA Journal of Numerical Analysis, 22(3), 329–343.
    - ARCHITECTURE.md §4.2: Deterministic Financial Engine covariance pipeline.
    - FINANCIAL_METHODOLOGY.md §4.2.2: 6-step Higham projection procedure.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from app.config import settings
from app.core.exceptions import (
    CovarianceValidationError,
    FinancialCalculationError,
    HighamConvergenceError,
    InsufficientDataError,
)

logger = logging.getLogger("quantamental.regularization")

# ---------------------------------------------------------------------------
# Module-level defaults & constants
# ---------------------------------------------------------------------------

DEFAULT_EIGENVALUE_FLOOR: float = settings.MIN_EIGENVALUE_FLOOR  # 1e-7
DEFAULT_TOLERANCE: float = 1e-7
DEFAULT_MAX_ITERATIONS: int = 100
DEFAULT_MAX_CONDITION_NUMBER: float = settings.MAX_CONDITION_NUMBER  # 1e6
DEFAULT_VARIANCE_PRESERVATION_TOL: float = 1e-6
DEFAULT_INPUT_SYMMETRY_TOLERANCE: float = 1e-8
_OUTPUT_SYMMETRY_TOL: float = 1e-10


# ---------------------------------------------------------------------------
# Internal algorithm helpers
# ---------------------------------------------------------------------------


def _validate_regularization_parameters(
    eigenvalue_floor: object,
    tolerance: object,
    max_iterations: object,
    max_condition_number: object,
    variance_preservation_tol: object,
    input_symmetry_tolerance: object,
) -> None:
    """Validate numerical configuration arguments strictly."""
    if (
        isinstance(eigenvalue_floor, bool)
        or not isinstance(eigenvalue_floor, (int, float))
        or not np.isfinite(eigenvalue_floor)
        or eigenvalue_floor <= 0.0
    ):
        raise FinancialCalculationError(
            f"eigenvalue_floor must be a finite positive float, got {eigenvalue_floor!r}.",
            details={"eigenvalue_floor": eigenvalue_floor},
        )

    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise FinancialCalculationError(
            f"tolerance must be a finite positive float, got {tolerance!r}.",
            details={"tolerance": tolerance},
        )

    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        raise FinancialCalculationError(
            f"max_iterations must be a positive integer, got {max_iterations!r}.",
            details={"max_iterations": max_iterations},
        )

    if (
        isinstance(max_condition_number, bool)
        or not isinstance(max_condition_number, (int, float))
        or not np.isfinite(max_condition_number)
        or max_condition_number <= 0.0
    ):
        raise FinancialCalculationError(
            f"max_condition_number must be a finite positive float, got {max_condition_number!r}.",
            details={"max_condition_number": max_condition_number},
        )

    if (
        isinstance(variance_preservation_tol, bool)
        or not isinstance(variance_preservation_tol, (int, float))
        or not np.isfinite(variance_preservation_tol)
        or variance_preservation_tol <= 0.0
    ):
        raise FinancialCalculationError(
            f"variance_preservation_tol must be a finite positive float, got {variance_preservation_tol!r}.",
            details={"variance_preservation_tol": variance_preservation_tol},
        )

    if (
        isinstance(input_symmetry_tolerance, bool)
        or not isinstance(input_symmetry_tolerance, (int, float))
        or not np.isfinite(input_symmetry_tolerance)
        or input_symmetry_tolerance <= 0.0
    ):
        raise FinancialCalculationError(
            f"input_symmetry_tolerance must be a finite positive float, got {input_symmetry_tolerance!r}.",
            details={"input_symmetry_tolerance": input_symmetry_tolerance},
        )


def _validate_input_covariance(
    covariance: object,
    input_symmetry_tolerance: float,
) -> Tuple[np.ndarray, pd.Index, pd.Index]:
    """Validate input covariance DataFrame structure, values, and symmetry."""
    if covariance is None:
        raise FinancialCalculationError(
            "covariance must be a pandas DataFrame, got None.",
            details={"type": "NoneType"},
        )
    if not isinstance(covariance, pd.DataFrame):
        raise FinancialCalculationError(
            f"covariance must be a pandas DataFrame, got {type(covariance).__name__}.",
            details={"type": type(covariance).__name__},
        )
    if covariance.empty:
        raise InsufficientDataError(
            "covariance DataFrame is empty; at least 2 assets are required.",
            details={"shape": list(covariance.shape)},
        )

    n_rows, n_cols = covariance.shape
    if n_rows != n_cols:
        raise FinancialCalculationError(
            f"covariance matrix must be square, got shape ({n_rows}, {n_cols}).",
            details={"shape": list(covariance.shape)},
        )
    if n_rows < 2:
        raise InsufficientDataError(
            f"covariance matrix requires at least 2 assets; got {n_rows}.",
            details={"n_assets": n_rows},
        )

    # Label duplication checks
    if covariance.columns.duplicated().any():
        dupes = covariance.columns[covariance.columns.duplicated(keep=False)].tolist()
        raise FinancialCalculationError(
            "covariance DataFrame contains duplicate column labels.",
            details={"duplicate_columns": dupes},
        )
    if covariance.index.duplicated().any():
        dupes_idx = covariance.index[covariance.index.duplicated(keep=False)].tolist()
        raise FinancialCalculationError(
            "covariance DataFrame contains duplicate index labels.",
            details={"duplicate_index": dupes_idx},
        )

    # Identical index and column labels and order
    if list(covariance.index) != list(covariance.columns):
        raise FinancialCalculationError(
            "covariance row and column labels must be identical and in the exact same order.",
            details={
                "index_labels": list(covariance.index),
                "column_labels": list(covariance.columns),
            },
        )

    # Column dtype checks
    non_numeric_cols = [
        col
        for col in covariance.columns
        if not pd.api.types.is_numeric_dtype(covariance[col])
    ]
    if non_numeric_cols:
        raise FinancialCalculationError(
            "covariance DataFrame contains non-numeric columns.",
            details={"non_numeric_columns": non_numeric_cols},
        )

    complex_cols = [
        col
        for col in covariance.columns
        if pd.api.types.is_complex_dtype(covariance[col])
    ]
    if complex_cols:
        raise FinancialCalculationError(
            "covariance DataFrame contains complex-valued columns.",
            details={"complex_columns": complex_cols},
        )

    # Missing values check (preceding float64 conversion)
    if covariance.isnull().values.any():
        nan_counts = covariance.isnull().sum()
        nan_assets = nan_counts[nan_counts > 0].to_dict()
        raise FinancialCalculationError(
            "covariance DataFrame contains NaN or missing values.",
            details={"nan_counts_by_asset": {str(k): int(v) for k, v in nan_assets.items()}},
        )

    # Convert to NumPy float64
    try:
        cov_f64: np.ndarray = covariance.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise FinancialCalculationError(
            f"Could not convert covariance DataFrame to float64 array: {exc}.",
            details={"error": str(exc)},
        ) from exc

    if not np.all(np.isfinite(cov_f64)):
        raise FinancialCalculationError(
            "covariance matrix contains infinite values.",
            details={"shape": list(cov_f64.shape)},
        )

    # Diagonal variance positivity check (Step 1 requirement)
    variances = np.diag(cov_f64)
    if not np.all(variances > 0.0):
        invalid_vars = {
            str(col): float(val)
            for col, val in zip(covariance.columns, variances)
            if val <= 0.0 or not np.isfinite(val)
        }
        raise FinancialCalculationError(
            "All diagonal variance elements must be finite and strictly positive (> 0).",
            details={"invalid_diagonal_variances": invalid_vars},
        )

    # Symmetry check
    asymmetry = float(np.max(np.abs(cov_f64 - cov_f64.T)))
    if asymmetry > input_symmetry_tolerance:
        raise FinancialCalculationError(
            f"covariance matrix is materially asymmetric "
            f"(max asymmetry={asymmetry:.2e} > tolerance={input_symmetry_tolerance:.2e}). "
            "Materially asymmetric inputs are not silently repaired.",
            details={
                "max_asymmetry": asymmetry,
                "input_symmetry_tolerance": input_symmetry_tolerance,
            },
        )

    # Eliminate tiny floating-point rounding asymmetry if within tolerance
    cov_f64 = (cov_f64 + cov_f64.T) / 2.0

    return cov_f64, covariance.index, covariance.columns


def _higham_alternating_projections(
    C: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> Tuple[np.ndarray, int, float]:
    """Execute Higham (2002) alternating projections with Dykstra's correction.

    Parameters
    ----------
    C:
        Initial symmetric correlation-scale matrix with unit diagonal.
    tolerance:
        Relative Frobenius-norm convergence tolerance.
    max_iterations:
        Iteration budget before raising HighamConvergenceError.

    Returns
    -------
    (C_proj, iterations, final_rel_diff)
        C_proj: Nearest PSD correlation matrix.
        iterations: Number of iterations executed.
        final_rel_diff: Final relative Frobenius-norm difference.

    Raises
    ------
    HighamConvergenceError:
        If convergence is not achieved within max_iterations.
    """
    n = C.shape[0]
    Y = C.copy()
    delta_S = np.zeros((n, n), dtype=np.float64)

    rel_diff = float("inf")
    converged = False

    for iteration in range(1, max_iterations + 1):
        Y_prev = Y.copy()

        # Step 1: Dykstra correction
        R = Y - delta_S
        R = (R + R.T) / 2.0

        # Step 2: Project onto positive semidefinite cone S_+
        w, V = np.linalg.eigh(R)
        w_plus = np.maximum(w, 0.0)
        X = (V * w_plus) @ V.T
        X = (X + X.T) / 2.0

        # Step 3: Update Dykstra correction increment
        delta_S = X - R

        # Step 4: Project onto affine set of symmetric matrices with unit diagonal
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
        Y = (Y + Y.T) / 2.0

        # Step 5: Convergence assessment
        norm_Y = float(np.linalg.norm(Y, ord="fro"))
        diff_Y = float(np.linalg.norm(Y - Y_prev, ord="fro"))
        diff_XY = float(np.linalg.norm(Y - X, ord="fro"))

        denom = max(norm_Y, 1e-12)
        rel_diff_Y = diff_Y / denom
        rel_diff_XY = diff_XY / denom
        rel_diff = max(rel_diff_Y, rel_diff_XY)

        if rel_diff <= tolerance:
            converged = True
            break

    if not converged:
        raise HighamConvergenceError(
            f"Higham alternating projection failed to converge within {max_iterations} iterations "
            f"(final rel_diff={rel_diff:.2e}, tolerance={tolerance:.2e}).",
            details={
                "max_iterations": max_iterations,
                "tolerance": tolerance,
                "final_rel_diff": float(rel_diff),
            },
        )

    return Y, iteration, float(rel_diff)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def regularize_covariance_higham(
    covariance: pd.DataFrame,
    *,
    eigenvalue_floor: float = DEFAULT_EIGENVALUE_FLOOR,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_condition_number: float = DEFAULT_MAX_CONDITION_NUMBER,
    variance_preservation_tol: float = DEFAULT_VARIANCE_PRESERVATION_TOL,
    input_symmetry_tolerance: float = DEFAULT_INPUT_SYMMETRY_TOLERANCE,
) -> pd.DataFrame:
    """Project an empirical covariance matrix to a strictly positive-definite matrix.

    Implements Higham's (2002) nearest-correlation-matrix projection with
    Dykstra correction, followed by controlled eigenvalue flooring, diagonal
    renormalization, and covariance rescaling using original standard deviations.

    Parameters
    ----------
    covariance:
        Labelled square DataFrame representing the input covariance matrix.
        Rows and columns must have identical asset labels in the exact same order.
        All diagonal variance entries must be strictly positive (> 0).
    eigenvalue_floor:
        Lower bound applied to eigenvalues of the projected correlation matrix
        prior to diagonal renormalization (default: settings.MIN_EIGENVALUE_FLOOR = 1e-7).
    tolerance:
        Relative Frobenius-norm convergence tolerance for Higham alternating
        projections (default: 1e-7).
    max_iterations:
        Maximum number of alternating projection iterations (default: 100).
    max_condition_number:
        Maximum allowable condition number (λ_max / λ_min) for the final
        rescaled covariance matrix (default: settings.MAX_CONDITION_NUMBER = 1e6).
    variance_preservation_tol:
        Maximum allowable relative discrepancy between original variances and
        regularized diagonal variances (default: 1e-6).
    input_symmetry_tolerance:
        Maximum allowable absolute asymmetry |Σ - Σᵀ|_max in the input matrix
        before raising FinancialCalculationError (default: 1e-8).

    Returns
    -------
    pd.DataFrame
        Strictly positive-definite, symmetric regularized covariance matrix
        preserving the original asset labels, ordering, and diagonal variances.

    Raises
    ------
    FinancialCalculationError:
        If covariance is not a DataFrame, not square, contains non-numeric,
        complex, missing, infinite, non-positive diagonal entries, or is
        materially asymmetric. Also raised for invalid parameter configurations.
    InsufficientDataError:
        If covariance is empty or has fewer than 2 assets.
    HighamConvergenceError:
        If Higham alternating projections fails to converge within max_iterations.
    CovarianceValidationError:
        If the regularized covariance matrix fails post-calculation validation
        (non-finite values, asymmetry, non-positive definiteness, condition number
        exceeding max_condition_number, or variance preservation failure).
    """
    # ------------------------------------------------------------------
    # Step 0: Validate configuration parameters
    # ------------------------------------------------------------------
    _validate_regularization_parameters(
        eigenvalue_floor=eigenvalue_floor,
        tolerance=tolerance,
        max_iterations=max_iterations,
        max_condition_number=max_condition_number,
        variance_preservation_tol=variance_preservation_tol,
        input_symmetry_tolerance=input_symmetry_tolerance,
    )

    # ------------------------------------------------------------------
    # Step 1: Validate input covariance matrix
    # ------------------------------------------------------------------
    cov_f64, index_labels, column_labels = _validate_input_covariance(
        covariance=covariance,
        input_symmetry_tolerance=input_symmetry_tolerance,
    )

    original_variances = np.diag(cov_f64).copy()
    original_std = np.sqrt(original_variances)

    # ------------------------------------------------------------------
    # Step 2: Standardize covariance to correlation scale
    # ------------------------------------------------------------------
    # C = D^(-1/2) Σ D^(-1/2)
    inv_std = 1.0 / original_std
    C = cov_f64 * np.outer(inv_std, inv_std)
    np.fill_diagonal(C, 1.0)
    C = (C + C.T) / 2.0

    # ------------------------------------------------------------------
    # Step 3: Higham alternating projections with Dykstra correction
    # ------------------------------------------------------------------
    logger.debug(
        "Applying Higham nearest-correlation projection: n=%d, tol=%.2e, max_iter=%d",
        cov_f64.shape[0],
        tolerance,
        max_iterations,
    )
    C_proj, iterations, final_rel_diff = _higham_alternating_projections(
        C=C,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    logger.debug(
        "Higham alternating projection converged in %d iterations (rel_diff=%.2e)",
        iterations,
        final_rel_diff,
    )

    # ------------------------------------------------------------------
    # Step 4: Eigenvalue floor and diagonal renormalization
    # ------------------------------------------------------------------
    # Spectral decomposition: C_proj = V Λ Vᵀ
    w, V = np.linalg.eigh(C_proj)
    w_floored = np.maximum(w, eigenvalue_floor)
    C_tilde = (V * w_floored) @ V.T
    C_tilde = (C_tilde + C_tilde.T) / 2.0

    # Diagonal renormalization to restore exact unit diagonals:
    # S_diag = diag(C_tilde), C_reg = S_diag^(-1/2) C_tilde S_diag^(-1/2)
    diag_tilde = np.diag(C_tilde)
    inv_sqrt_diag = 1.0 / np.sqrt(diag_tilde)
    C_reg = C_tilde * np.outer(inv_sqrt_diag, inv_sqrt_diag)
    np.fill_diagonal(C_reg, 1.0)
    C_reg = (C_reg + C_reg.T) / 2.0

    # ------------------------------------------------------------------
    # Step 5: Rescaling to covariance units
    # ------------------------------------------------------------------
    # Σ_reg = D^(1/2) C_reg D^(1/2)
    cov_reg = C_reg * np.outer(original_std, original_std)
    cov_reg = (cov_reg + cov_reg.T) / 2.0

    # ------------------------------------------------------------------
    # Step 6: Post-rescaling multi-criteria validation
    # ------------------------------------------------------------------
    # a. Finiteness
    if not np.all(np.isfinite(cov_reg)):
        raise CovarianceValidationError(
            "Regularized covariance matrix contains non-finite values.",
            details={"shape": list(cov_reg.shape)},
        )

    # b. Symmetry
    max_asym = float(np.max(np.abs(cov_reg - cov_reg.T)))
    if max_asym > _OUTPUT_SYMMETRY_TOL:
        raise CovarianceValidationError(
            f"Regularized covariance matrix failed symmetry check "
            f"(max asymmetry={max_asym:.2e} > tol={_OUTPUT_SYMMETRY_TOL:.2e}).",
            details={"max_asymmetry": max_asym, "tolerance": _OUTPUT_SYMMETRY_TOL},
        )

    # c. Positive diagonal variances
    final_variances = np.diag(cov_reg)
    if not np.all(final_variances > 0.0):
        raise CovarianceValidationError(
            "Regularized covariance matrix has non-positive diagonal elements.",
            details={"min_diagonal": float(np.min(final_variances))},
        )

    # d. Strict positive definiteness & Condition number
    w_final = np.linalg.eigvalsh(cov_reg)
    min_eig = float(w_final[0])
    max_eig = float(w_final[-1])

    if min_eig <= 0.0:
        raise CovarianceValidationError(
            f"Regularized covariance matrix is not strictly positive definite "
            f"(min_eigenvalue={min_eig:.2e} <= 0).",
            details={"min_eigenvalue": min_eig, "max_eigenvalue": max_eig},
        )

    condition_number = max_eig / min_eig
    if condition_number >= max_condition_number:
        raise CovarianceValidationError(
            f"Regularized covariance matrix condition number ({condition_number:.2e}) "
            f"exceeds maximum allowable threshold ({max_condition_number:.2e}).",
            details={
                "condition_number": condition_number,
                "max_condition_number": max_condition_number,
                "min_eigenvalue": min_eig,
                "max_eigenvalue": max_eig,
            },
        )

    # e. Variance preservation: max_i |Σ_reg,ii - σ_i²| / σ_i² <= tol
    rel_var_diff = float(np.max(np.abs(final_variances - original_variances) / original_variances))
    if rel_var_diff > variance_preservation_tol:
        raise CovarianceValidationError(
            f"Original asset variances were not preserved within tolerance "
            f"(max relative difference={rel_var_diff:.2e} > tol={variance_preservation_tol:.2e}).",
            details={
                "max_relative_variance_diff": rel_var_diff,
                "variance_preservation_tol": variance_preservation_tol,
            },
        )

    logger.info(
        "Covariance regularization successful: shape=%s, cond_num=%.2e, min_eig=%.2e, iters=%d",
        cov_reg.shape,
        condition_number,
        min_eig,
        iterations,
    )

    return pd.DataFrame(cov_reg, index=index_labels, columns=column_labels)
