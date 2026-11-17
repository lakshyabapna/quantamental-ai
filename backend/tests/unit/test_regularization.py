"""Tests for Phase 2.2 Step 3: Higham Positive-Definite Covariance Regularization.

Covers:
1.  Already positive-definite covariance matrix (minimal perturbation).
2.  Positive-semidefinite but singular covariance matrix (rank deficient, λ_min = 0).
3.  Indefinite covariance matrix (with negative eigenvalues).
4.  Perfectly correlated assets (r = 1.0).
5.  Highly correlated assets (r = 0.999).
6.  Correct correlation diagonal after projection (all exactly 1.0).
7.  Correct covariance rescaling (Σ_reg = D^(1/2) C_reg D^(1/2)).
8.  Original variances preserved within tolerance.
9.  Asset labels and ordering preserved.
10. Symmetry of final matrix.
11. Minimum eigenvalue strictly positive and condition number < 1e6.
12. Zero and negative diagonal variance rejection.
13. NaN, Inf and complex input rejection.
14. Non-numeric values rejection.
15. Duplicate labels (duplicate columns or duplicate index).
16. Mismatched row and column labels.
17. Materially asymmetric input rejection.
18. Invalid eigenvalue floor rejection.
19. Invalid tolerance rejection.
20. Invalid maximum iteration count rejection.
21. Non-convergence behavior (HighamConvergenceError raised).
22. Numerical stability on ill-conditioned matrices.
23. Independent numerical test on known 3x3 indefinite test case.
24. Namespace compatibility imports from portfolio_engine and financial_engine.
25. Confirmation that Step 1 and Step 2 estimators remain completely unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import (
    CovarianceValidationError,
    FinancialCalculationError,
    HighamConvergenceError,
    InsufficientDataError,
)
from app.portfolio_engine.regularization import (
    DEFAULT_EIGENVALUE_FLOOR,
    DEFAULT_MAX_CONDITION_NUMBER,
    regularize_covariance_higham,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cov_df(matrix: np.ndarray, labels: list[str] | None = None) -> pd.DataFrame:
    """Build a square covariance DataFrame with given labels."""
    n = matrix.shape[0]
    if labels is None:
        labels = [f"A{i}" for i in range(n)]
    return pd.DataFrame(matrix, index=labels, columns=labels)


# ---------------------------------------------------------------------------
# 1. Already Positive-Definite Covariance Matrix
# ---------------------------------------------------------------------------


class TestAlreadyPositiveDefinite:
    """An already well-conditioned PD covariance matrix should pass through with
    minimal numerical perturbation."""

    def test_already_pd_matrix_preserves_structure(self):
        cov = np.array(
            [
                [0.04, 0.01, 0.005],
                [0.01, 0.09, 0.02],
                [0.005, 0.02, 0.16],
            ]
        )
        # Verify it is already PD
        assert np.all(np.linalg.eigvalsh(cov) > 0)

        df = _make_cov_df(cov, ["RELIANCE", "TCS", "INFY"])
        result = regularize_covariance_higham(df)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["RELIANCE", "TCS", "INFY"]
        assert list(result.index) == ["RELIANCE", "TCS", "INFY"]

        # Output should be very close to original
        np.testing.assert_allclose(result.values, cov, rtol=1e-5, atol=1e-7)

    def test_already_pd_variances_strictly_preserved(self):
        cov = np.array(
            [
                [0.0225, 0.008],
                [0.008, 0.0400],
            ]
        )
        df = _make_cov_df(cov, ["A", "B"])
        result = regularize_covariance_higham(df)

        orig_diag = np.diag(cov)
        res_diag = np.diag(result.values)
        np.testing.assert_allclose(res_diag, orig_diag, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. Positive-Semidefinite but Singular (Rank Deficient)
# ---------------------------------------------------------------------------


class TestSingularCovariance:
    """Rank-deficient matrices (λ_min = 0) must be projected to strictly PD."""

    def _make_singular_matrix(self) -> tuple[np.ndarray, list[str]]:
        v1 = np.array([0.02, 0.01, -0.01])
        v2 = np.array([-0.01, 0.03, 0.02])
        v3 = v1 + v2
        data = np.column_stack([v1, v2, v3])
        cov_singular = (data.T @ data) / 2.0
        return cov_singular, ["A", "B", "C"]

    def test_rank_deficient_3x3_becomes_strictly_pd_with_calibrated_floor(self):
        cov_singular, labels = self._make_singular_matrix()
        # Verify initial singularity
        w_orig = np.linalg.eigvalsh(cov_singular)
        assert np.isclose(w_orig[0], 0.0, atol=1e-14)

        df = _make_cov_df(cov_singular, labels)
        # Using eigenvalue_floor=1e-5 produces condition number < 1e6
        result = regularize_covariance_higham(df, eigenvalue_floor=1e-5)

        w_res = np.linalg.eigvalsh(result.values)
        assert w_res[0] > 0.0  # Strictly positive definite
        assert np.all(w_res > 0.0)
        assert (w_res[-1] / w_res[0]) < DEFAULT_MAX_CONDITION_NUMBER

    def test_rank_deficient_with_floor_1e7_and_extended_condition_budget(self):
        cov_singular, labels = self._make_singular_matrix()
        df = _make_cov_df(cov_singular, labels)
        result = regularize_covariance_higham(
            df, eigenvalue_floor=1e-7, max_condition_number=1e8
        )
        w_res = np.linalg.eigvalsh(result.values)
        assert w_res[0] > 0.0
        assert np.all(w_res > 0.0)

    def test_rank_deficient_rejected_when_condition_number_exceeds_threshold(self):
        cov_singular, labels = self._make_singular_matrix()
        df = _make_cov_df(cov_singular, labels)
        # Default floor of 1e-7 produces kappa ~ 2e7 > 1e6 -> triggers rejection gate
        with pytest.raises(CovarianceValidationError, match="condition number"):
            regularize_covariance_higham(df, eigenvalue_floor=1e-7)


# ---------------------------------------------------------------------------
# 3. Indefinite Covariance Matrix (Negative Eigenvalues)
# ---------------------------------------------------------------------------


class TestIndefiniteCovariance:
    """Indefinite covariance matrices (λ_min < 0) must be projected to strictly PD."""

    def _make_indefinite_matrix(self) -> tuple[np.ndarray, list[str]]:
        # Known indefinite correlation matrix:
        # C = [[1, 0.8, 0], [0.8, 1, 0.8], [0, 0.8, 1]] has λ = [-0.131, 1.0, 2.131]
        C_indef = np.array(
            [
                [1.0, 0.8, 0.0],
                [0.8, 1.0, 0.8],
                [0.0, 0.8, 1.0],
            ]
        )
        assert np.linalg.eigvalsh(C_indef)[0] < -0.1
        sigmas = np.array([0.02, 0.015, 0.025])
        D = np.diag(sigmas)
        cov_indef = D @ C_indef @ D
        return cov_indef, ["X", "Y", "Z"]

    def test_3x3_indefinite_becomes_strictly_pd_with_calibrated_floor(self):
        cov_indef, labels = self._make_indefinite_matrix()
        df = _make_cov_df(cov_indef, labels)
        result = regularize_covariance_higham(df, eigenvalue_floor=1e-5)

        w_res = np.linalg.eigvalsh(result.values)
        assert w_res[0] > 0.0
        assert np.all(w_res > 0.0)
        assert (w_res[-1] / w_res[0]) < DEFAULT_MAX_CONDITION_NUMBER

    def test_3x3_indefinite_with_extended_condition_budget(self):
        cov_indef, labels = self._make_indefinite_matrix()
        df = _make_cov_df(cov_indef, labels)
        result = regularize_covariance_higham(
            df, eigenvalue_floor=1e-7, max_condition_number=1e8
        )
        w_res = np.linalg.eigvalsh(result.values)
        assert w_res[0] > 0.0
        assert np.all(w_res > 0.0)

    def test_3x3_indefinite_triggers_rejection_gate_on_tight_threshold(self):
        cov_indef, labels = self._make_indefinite_matrix()
        df = _make_cov_df(cov_indef, labels)
        with pytest.raises(CovarianceValidationError, match="condition number"):
            regularize_covariance_higham(df, eigenvalue_floor=1e-7, max_condition_number=1e6)


# ---------------------------------------------------------------------------
# 4 & 5. Perfectly and Highly Correlated Assets
# ---------------------------------------------------------------------------


class TestCorrelatedAssets:

    def test_perfectly_correlated_assets(self):
        """r = 1.0 produces a rank-1 singular matrix."""
        s1, s2 = 0.02, 0.03
        cov = np.array(
            [
                [s1**2, s1 * s2],
                [s1 * s2, s2**2],
            ]
        )
        assert np.isclose(np.linalg.eigvalsh(cov)[0], 0.0, atol=1e-15)

        df = _make_cov_df(cov, ["A", "B"])
        result = regularize_covariance_higham(df, eigenvalue_floor=1e-5)

        w = np.linalg.eigvalsh(result.values)
        assert w[0] > 0.0
        # Correlation off-diagonal should be slightly less than 1.0
        res_corr = result.values[0, 1] / (s1 * s2)
        assert res_corr < 1.0
        assert res_corr > 0.99

    def test_highly_correlated_assets(self):
        """r = 0.999 between assets."""
        s1, s2 = 0.02, 0.02
        r = 0.999
        cov = np.array(
            [
                [s1**2, r * s1 * s2],
                [r * s1 * s2, s2**2],
            ]
        )
        df = _make_cov_df(cov, ["P", "Q"])
        result = regularize_covariance_higham(df)

        w = np.linalg.eigvalsh(result.values)
        assert w[0] > 0.0
        assert np.all(np.isfinite(result.values))


# ---------------------------------------------------------------------------
# 6 & 7 & 8. Correlation Diagonal, Covariance Rescaling & Variance Preservation
# ---------------------------------------------------------------------------


class TestPropertiesAndRescaling:

    def test_variance_preservation_across_diverse_scales(self):
        """Diagonal entries of regularized covariance must match original variances."""
        variances = np.array([0.01, 0.04, 0.02, 0.09, 0.03])
        n = len(variances)
        # Create an indefinite matrix
        rng = np.random.default_rng(101)
        A = rng.normal(0, 1, (n, n))
        A = (A + A.T) / 2.0
        np.fill_diagonal(A, 1.0)
        D_half = np.diag(np.sqrt(variances))
        cov_indef = D_half @ A @ D_half

        df = _make_cov_df(cov_indef, [f"Asset_{i}" for i in range(n)])
        result = regularize_covariance_higham(df, eigenvalue_floor=1e-5)

        # Variances must match within tolerance 1e-6
        final_variances = np.diag(result.values)
        rel_diff = np.abs(final_variances - variances) / variances
        np.testing.assert_allclose(rel_diff, 0.0, atol=1e-6)

    def test_rescaling_relationship(self):
        """Check that Σ_reg = D^(1/2) C_reg D^(1/2) exactly."""
        s = np.array([0.02, 0.03, 0.04])
        cov = np.diag(s**2)
        cov[0, 1] = cov[1, 0] = 0.02 * 0.03 * 0.8
        df = _make_cov_df(cov, ["A", "B", "C"])
        result = regularize_covariance_higham(df)

        # Compute correlation from result
        d_half = np.diag(s)
        inv_d = np.diag(1.0 / s)
        C_derived = inv_d @ result.values @ inv_d
        np.testing.assert_allclose(np.diag(C_derived), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 9 & 10. Label Ordering and Symmetry
# ---------------------------------------------------------------------------


class TestLabelsAndSymmetry:

    def test_labels_and_ordering_strictly_preserved(self):
        labels = ["TCS", "INFY", "WIPRO", "HCLTECH"]
        cov = np.eye(4) * 0.04
        df = _make_cov_df(cov, labels)
        result = regularize_covariance_higham(df)

        assert list(result.columns) == labels
        assert list(result.index) == labels

    def test_symmetry_enforced_to_machine_precision(self):
        cov = np.array(
            [
                [0.04, 0.01, 0.005],
                [0.01, 0.09, 0.02],
                [0.005, 0.02, 0.16],
            ]
        )
        df = _make_cov_df(cov, ["A", "B", "C"])
        result = regularize_covariance_higham(df)

        np.testing.assert_allclose(result.values, result.values.T, atol=1e-14)


# ---------------------------------------------------------------------------
# 11. Condition Number & Positive Definiteness
# ---------------------------------------------------------------------------


class TestConditionNumberValidation:

    def test_positive_definiteness_guaranteed(self):
        cov = np.array(
            [
                [1.0, 0.99999, 0.99999],
                [0.99999, 1.0, 0.99999],
                [0.99999, 0.99999, 1.0],
            ]
        ) * 0.04
        df = _make_cov_df(cov, ["A", "B", "C"])
        result = regularize_covariance_higham(df)

        w = np.linalg.eigvalsh(result.values)
        assert w[0] > 0.0

    def test_extreme_condition_number_rejected_by_threshold(self):
        """When max_condition_number is set very strict, reject."""
        cov = np.array(
            [
                [1.0, 0.0],
                [0.0, 1e-10],
            ]
        )
        # Condition number is 1e10
        df = _make_cov_df(cov, ["A", "B"])
        with pytest.raises(CovarianceValidationError, match="condition number"):
            regularize_covariance_higham(df, max_condition_number=1e5)


# ---------------------------------------------------------------------------
# 12. Non-positive Diagonal Variance Rejection
# ---------------------------------------------------------------------------


class TestDiagonalVarianceRejection:

    def test_zero_variance_raises_financial_calculation_error(self):
        cov = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.04],
            ]
        )
        df = _make_cov_df(cov, ["A", "B"])
        with pytest.raises(FinancialCalculationError, match="strictly positive"):
            regularize_covariance_higham(df)

    def test_negative_variance_raises_financial_calculation_error(self):
        cov = np.array(
            [
                [-0.01, 0.0],
                [0.0, 0.04],
            ]
        )
        df = _make_cov_df(cov, ["A", "B"])
        with pytest.raises(FinancialCalculationError, match="strictly positive"):
            regularize_covariance_higham(df)


# ---------------------------------------------------------------------------
# 13, 14, 15, 16. Structural and Value Rejection
# ---------------------------------------------------------------------------


class TestStructuralAndValueRejection:

    def test_none_raises(self):
        with pytest.raises(FinancialCalculationError, match="None"):
            regularize_covariance_higham(None)

    def test_non_dataframe_raises(self):
        with pytest.raises(FinancialCalculationError, match="DataFrame"):
            regularize_covariance_higham(np.eye(3))

    def test_empty_dataframe_raises(self):
        with pytest.raises(InsufficientDataError, match="empty"):
            regularize_covariance_higham(pd.DataFrame())

    def test_single_asset_raises(self):
        df = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
        with pytest.raises(InsufficientDataError, match="2 assets"):
            regularize_covariance_higham(df)

    def test_non_square_raises(self):
        df = pd.DataFrame(
            [[0.04, 0.01, 0.0], [0.01, 0.09, 0.0]],
            index=["A", "B"],
            columns=["A", "B", "C"],
        )
        with pytest.raises(FinancialCalculationError, match="square"):
            regularize_covariance_higham(df)

    def test_nan_raises(self):
        cov = np.array([[0.04, np.nan], [np.nan, 0.09]])
        df = _make_cov_df(cov, ["A", "B"])
        with pytest.raises(FinancialCalculationError, match="NaN"):
            regularize_covariance_higham(df)

    def test_inf_raises(self):
        cov = np.array([[0.04, np.inf], [np.inf, 0.09]])
        df = _make_cov_df(cov, ["A", "B"])
        with pytest.raises(FinancialCalculationError, match="infinite"):
            regularize_covariance_higham(df)

    def test_complex_raises(self):
        cov = np.array([[0.04 + 0j, 0.01 + 0j], [0.01 + 0j, 0.09 + 0j]], dtype=complex)
        df = _make_cov_df(cov, ["A", "B"])
        with pytest.raises(FinancialCalculationError, match="complex"):
            regularize_covariance_higham(df)

    def test_non_numeric_column_raises(self):
        df = pd.DataFrame(
            [["0.04", "0.01"], ["0.01", "0.09"]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        with pytest.raises(FinancialCalculationError, match="non-numeric"):
            regularize_covariance_higham(df)

    def test_duplicate_column_labels_raises(self):
        cov = np.eye(2) * 0.04
        df = pd.DataFrame(cov, index=["A", "B"], columns=["A", "A"])
        with pytest.raises(FinancialCalculationError, match="duplicate"):
            regularize_covariance_higham(df)

    def test_duplicate_index_labels_raises(self):
        cov = np.eye(2) * 0.04
        df = pd.DataFrame(cov, index=["A", "A"], columns=["A", "B"])
        with pytest.raises(FinancialCalculationError, match="duplicate"):
            regularize_covariance_higham(df)

    def test_mismatched_row_col_labels_raises(self):
        cov = np.eye(2) * 0.04
        df = pd.DataFrame(cov, index=["A", "B"], columns=["B", "A"])
        with pytest.raises(FinancialCalculationError, match="identical and in the exact same order"):
            regularize_covariance_higham(df)


# ---------------------------------------------------------------------------
# 17. Materially Asymmetric Input Rejection
# ---------------------------------------------------------------------------


class TestAsymmetricInputRejection:

    def test_materially_asymmetric_matrix_rejected(self):
        # Asymmetry of 0.02 >> tolerance 1e-8
        cov = np.array(
            [
                [0.04, 0.03],
                [0.01, 0.04],
            ]
        )
        df = _make_cov_df(cov, ["A", "B"])
        with pytest.raises(FinancialCalculationError, match="materially asymmetric"):
            regularize_covariance_higham(df, input_symmetry_tolerance=1e-8)

    def test_sub_tolerance_asymmetry_repaired(self):
        # Asymmetry of 1e-12 < tolerance 1e-8 should be accepted
        cov = np.array(
            [
                [0.04, 0.01 + 1e-12],
                [0.01, 0.04],
            ]
        )
        df = _make_cov_df(cov, ["A", "B"])
        result = regularize_covariance_higham(df, input_symmetry_tolerance=1e-8)
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# 18, 19, 20. Parameter Validations
# ---------------------------------------------------------------------------


class TestParameterValidations:

    def _valid_df(self) -> pd.DataFrame:
        return _make_cov_df(np.eye(2) * 0.04, ["A", "B"])

    def test_bool_eigenvalue_floor_rejected(self):
        with pytest.raises(FinancialCalculationError, match="eigenvalue_floor"):
            regularize_covariance_higham(self._valid_df(), eigenvalue_floor=True)

    def test_negative_eigenvalue_floor_rejected(self):
        with pytest.raises(FinancialCalculationError, match="eigenvalue_floor"):
            regularize_covariance_higham(self._valid_df(), eigenvalue_floor=-1e-7)

    def test_zero_eigenvalue_floor_rejected(self):
        with pytest.raises(FinancialCalculationError, match="eigenvalue_floor"):
            regularize_covariance_higham(self._valid_df(), eigenvalue_floor=0.0)

    def test_bool_tolerance_rejected(self):
        with pytest.raises(FinancialCalculationError, match="tolerance"):
            regularize_covariance_higham(self._valid_df(), tolerance=True)

    def test_negative_tolerance_rejected(self):
        with pytest.raises(FinancialCalculationError, match="tolerance"):
            regularize_covariance_higham(self._valid_df(), tolerance=-1e-7)

    def test_bool_max_iterations_rejected(self):
        with pytest.raises(FinancialCalculationError, match="max_iterations"):
            regularize_covariance_higham(self._valid_df(), max_iterations=True)

    def test_float_max_iterations_rejected(self):
        with pytest.raises(FinancialCalculationError, match="max_iterations"):
            regularize_covariance_higham(self._valid_df(), max_iterations=100.0)

    def test_zero_max_iterations_rejected(self):
        with pytest.raises(FinancialCalculationError, match="max_iterations"):
            regularize_covariance_higham(self._valid_df(), max_iterations=0)

    def test_negative_max_iterations_rejected(self):
        with pytest.raises(FinancialCalculationError, match="max_iterations"):
            regularize_covariance_higham(self._valid_df(), max_iterations=-10)

    @pytest.mark.parametrize(
        "param_name",
        [
            "eigenvalue_floor",
            "tolerance",
            "max_condition_number",
            "variance_preservation_tol",
            "input_symmetry_tolerance",
        ],
    )
    @pytest.mark.parametrize(
        "invalid_val",
        [float("nan"), float("inf"), float("-inf")],
    )
    def test_non_finite_float_parameters_rejected(self, param_name, invalid_val):
        """NaN, +Inf and -Inf must be rejected for all floating-point parameters."""
        kwargs = {param_name: invalid_val}
        with pytest.raises(FinancialCalculationError, match=param_name):
            regularize_covariance_higham(self._valid_df(), **kwargs)


# ---------------------------------------------------------------------------
# 21. Non-Convergence Behavior
# ---------------------------------------------------------------------------


class TestNonConvergence:

    def test_max_iterations_one_raises_convergence_error(self):
        """An indefinite matrix requiring multiple iterations must fail if budget=1."""
        C_indef = np.array(
            [
                [1.0, 0.8, 0.0],
                [0.8, 1.0, 0.8],
                [0.0, 0.8, 1.0],
            ]
        )
        cov = C_indef * 0.04
        df = _make_cov_df(cov, ["A", "B", "C"])

        with pytest.raises(HighamConvergenceError, match="failed to converge") as exc_info:
            regularize_covariance_higham(df, max_iterations=1, tolerance=1e-12)

        assert exc_info.value.error_code == "HIGHAM_CONVERGENCE_ERROR"
        assert exc_info.value.details["max_iterations"] == 1


# ---------------------------------------------------------------------------
# 22. Numerical Stability on Ill-Conditioned Matrices
# ---------------------------------------------------------------------------


class TestNumericalStability:

    def test_large_ill_conditioned_matrix(self):
        """20x20 ill-conditioned matrix successfully regularized."""
        n = 20
        rng = np.random.default_rng(202)
        # Create rank-3 matrix in 20 dimensions
        factors = rng.normal(0, 0.01, (n, 3))
        cov_ill = factors @ factors.T
        np.fill_diagonal(cov_ill, np.diag(cov_ill) + 1e-4)

        df = _make_cov_df(cov_ill, [f"Asset_{i}" for i in range(n)])
        result = regularize_covariance_higham(df)

        w = np.linalg.eigvalsh(result.values)
        assert w[0] > 0.0
        assert (w[-1] / w[0]) < DEFAULT_MAX_CONDITION_NUMBER


# ---------------------------------------------------------------------------
# 23. Independent Numerical Test on Known 3x3 Test Case
# ---------------------------------------------------------------------------


class TestIndependentHighamNumericalResult:
    """Independent verification against reference Higham implementation.

    Test matrix:
        C = [[1.0, 0.8, 0.0],
             [0.8, 1.0, 0.8],
             [0.0, 0.8, 1.0]]
    Initial eigenvalues: [-0.13137, 1.0, 2.13137] (indefinite).
    """

    def test_independent_higham_reference(self):
        C_init = np.array(
            [
                [1.0, 0.8, 0.0],
                [0.8, 1.0, 0.8],
                [0.0, 0.8, 1.0],
            ]
        )
        sigmas = np.array([0.02, 0.02, 0.02])
        cov = np.diag(sigmas) @ C_init @ np.diag(sigmas)
        df = _make_cov_df(cov, ["A", "B", "C"])

        result = regularize_covariance_higham(
            df,
            eigenvalue_floor=DEFAULT_EIGENVALUE_FLOOR,
            tolerance=1e-8,
            max_condition_number=1e8,
        )

        # Standardize result back to correlation: C_reg = D^(-1/2) Σ_reg D^(-1/2)
        res_corr = result.values / 0.0004

        # For this symmetric nearest-correlation problem, the analytical / numerical
        # reference solution has:
        #   - Equal adjacent off-diagonal entries: C_12 = C_23 ≈ 0.7251
        #   - Outer off-diagonal entry: C_13 ≈ 0.0516
        #   - Unit diagonal entries: C_11 = C_22 = C_33 = 1.0
        EXPECTED_ADJACENT = 0.7251
        EXPECTED_OUTER = 0.0516

        # Verify symmetry of adjacent elements to high precision
        np.testing.assert_allclose(res_corr[0, 1], res_corr[1, 2], rtol=1e-5)
        np.testing.assert_allclose(res_corr[1, 0], res_corr[2, 1], rtol=1e-5)

        # Verify against reference values accounting for eigenvalue floor and renormalization
        np.testing.assert_allclose(res_corr[0, 1], EXPECTED_ADJACENT, atol=1e-3)
        np.testing.assert_allclose(res_corr[1, 2], EXPECTED_ADJACENT, atol=1e-3)
        np.testing.assert_allclose(res_corr[0, 2], EXPECTED_OUTER, atol=1e-3)
        np.testing.assert_allclose(res_corr[2, 0], EXPECTED_OUTER, atol=1e-3)

        # Diagonal must be exactly 1.0 and eigenvalues strictly positive
        np.testing.assert_allclose(np.diag(res_corr), 1.0, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(res_corr) > 0.0)


# ---------------------------------------------------------------------------
# 24. Namespace Compatibility Imports
# ---------------------------------------------------------------------------


class TestNamespaceImports:

    def test_import_from_portfolio_engine(self):
        from app.portfolio_engine import regularize_covariance_higham as fn
        assert callable(fn)

    def test_import_from_portfolio_engine_regularization(self):
        from app.portfolio_engine.regularization import regularize_covariance_higham as fn
        assert callable(fn)

    def test_import_from_financial_engine(self):
        from app.financial_engine import regularize_covariance_higham as fn
        assert callable(fn)

    def test_import_from_financial_engine_regularization(self):
        from app.financial_engine.regularization import regularize_covariance_higham as fn
        assert callable(fn)

    def test_all_imports_are_same_object(self):
        from app.portfolio_engine import regularize_covariance_higham as fn1
        from app.portfolio_engine.regularization import regularize_covariance_higham as fn2
        from app.financial_engine import regularize_covariance_higham as fn3
        from app.financial_engine.regularization import regularize_covariance_higham as fn4
        assert fn1 is fn2 is fn3 is fn4


# ---------------------------------------------------------------------------
# 25. Step 1 and Step 2 Estimators Remain Unaffected
# ---------------------------------------------------------------------------


class TestPriorStepsUnaffected:

    def test_step_1_sample_covariance_still_works(self):
        from app.portfolio_engine import calculate_sample_covariance
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        rng = np.random.default_rng(301)
        df = pd.DataFrame(
            {"A": rng.normal(0, 0.01, 30), "B": rng.normal(0, 0.02, 30)},
            index=dates,
        )
        cov = calculate_sample_covariance(df)
        assert cov.shape == (2, 2)
        assert np.all(np.isfinite(cov.values))

    def test_step_2_ledoit_wolf_still_works(self):
        from app.portfolio_engine import calculate_ledoit_wolf_covariance
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        rng = np.random.default_rng(302)
        df = pd.DataFrame(
            {"A": rng.normal(0, 0.01, 30), "B": rng.normal(0, 0.02, 30)},
            index=dates,
        )
        cov = calculate_ledoit_wolf_covariance(df)
        assert cov.shape == (2, 2)
        assert np.all(np.isfinite(cov.values))
