# Financial Methodology & Mathematical Foundations — Quantamental AI

**Author:** Senior Quantitative Developer & Financial Engineer
**Document:** Mathematical Specifications, Axioms, Edge Cases & Verification Protocols
**Target Market:** Indian Equities (National Stock Exchange - NSE)
**Status:** Canonical Financial Specification — Canonical Sign-off Edition (Approved Decisions)
**Version:** 1.4.1

---

## 1. Introduction and Architectural Principles

This document defines the mathematical foundations, formal formulas, input/output requirements, constraints, edge cases, and verification strategies for every financial calculation in **Quantamental AI**.

Quantamental AI is designed as a **quantitative finance research and decision-support platform**. The core premise of the platform is that **financial calculation engines must be deterministic, mathematically rigorous, and auditable**. Machine learning models provide predictive signals, but portfolio construction mathematics adheres strictly to Modern Portfolio Theory (Markowitz, 1952) and empirical risk management standards. The platform avoids promising perfect numerical or investment outcomes.

---

## 2. Annualization & Calendar Conventions

In financial calculations, inconsistent annualization conventions distort risk-adjusted metrics (such as Sharpe or Sortino ratios). Quantamental AI enforces the following conventions across all modules.

### 2.1 Indian Market Trading Days
- The National Stock Exchange of India (NSE) observes approximately **250 to 252 trading days per year** (excluding Saturdays, Sundays, and declared exchange holidays).
- **Canonical Standard:** The system uses $N = 252$ trading days as the annualization scalar.

### 2.2 Annualization Rules

| Metric | Daily Metric ($\theta_{daily}$) | Annualization Formula ($\theta_{ann}$) | Mathematical Assumption |
| :--- | :--- | :--- | :--- |
| **Arithmetic Expected Return** | Mean daily return $\bar{R}$ | $\mu_{ann} = 252 \times \bar{R}$ | Independent and identically distributed (i.i.d.) daily returns. Used strictly as input for single-period Markowitz optimization. |
| **Geometric Realized Return (CAGR)** | Cumulative return $\frac{V_T}{V_0} - 1$ | $\text{CAGR} = \left(\frac{V_T}{V_0}\right)^{\frac{252}{T}} - 1$ | Multi-period compounding over $T$ elapsed trading days. Mandatory for realized backtest performance reporting. |
| **Daily Volatility** | Sample standard deviation $\sigma_d$ | $\sigma_{ann} = \sqrt{252} \times \sigma_d$ | Square-root-of-time rule (assumes zero serial autocorrelation $\text{Cov}(R_t, R_{t-k}) = 0$). |
| **Covariance Matrix** | Daily covariance $\boldsymbol{\Sigma}_{daily}$ | $\boldsymbol{\Sigma}_{ann} = 252 \times \boldsymbol{\Sigma}_{daily}$ | Linear scaling under i.i.d. assumptions. |
| **Risk-Free Rate ($r_f$)** | Stated annual yield $r_{f, ann}$ (e.g. 6.50%) | $r_{f, daily} = (1 + r_{f, ann})^{\frac{1}{252}} - 1$ | Compounding daily discount rate calibrated to the 91-day Government of India Treasury Bill. |

> [!NOTE]
> **Risk-Free Rate Daily Conversion:** The system uses compounding over 252 trading days: $r_{f, daily} = (1 + r_{f, ann})^{1/252} - 1$. An alternative simple interest convention over calendar days ($r_{f, ann} / 365$) or trading days ($r_{f, ann} / 252$) is supported via configuration, but the compounding convention is the system default.
>
> [!IMPORTANT]
> **Separation of CAGR and Sharpe Numerator:** Realized CAGR is reported as an independent multi-period performance metric. It must **never** be substituted for the average daily excess return in the conventional historical Sharpe Ratio formula.

---

## 3. Return Calculations & Asset Moments

### 3.1 Simple vs. Logarithmic Returns

#### 3.1.1 Formulas
- **Simple Return ($R_t$):**
  $$R_t = \frac{P_{adj, t} - P_{adj, t-1}}{P_{adj, t-1}} = \frac{P_{adj, t}}{P_{adj, t-1}} - 1$$
- **Logarithmic Return ($r_t$):**
  $$r_t = \ln\left(\frac{P_{adj, t}}{P_{adj, t-1}}\right) = \ln(1 + R_t)$$

#### 3.1.2 Cross-Sectional vs. Time-Series Rules
- **Cross-Sectional Portfolio Aggregation:** Simple returns are strictly linear and additive across assets:
  $$R_{p, t} = \sum_{i=1}^N w_i R_{i, t} = \mathbf{w}^T \mathbf{R}_t$$
  *Log returns are NOT additive across assets* ($\ln \sum w_i P_i \ne \sum w_i \ln P_i$). Therefore, **all portfolio return and covariance calculations use simple returns**.
- **Time-Series Stationarity in ML/DL:** Log returns are time-additive ($r_{0, T} = \sum_{t=1}^T r_t$) and possess more symmetric statistical properties. Therefore, **all predictive machine learning models operate on log returns**.

### 3.2 Missing Data & Covariance Window Alignment
Unconditional forward-filling of prices is strictly forbidden because it corrupts return distributions and creates artificial zero returns. The system enforces domain-specific missing-data policies:

1. **Covariance Window Alignment (Listwise vs. Pairwise):**
   - For the initial 10-stock MVP, **the engine prefers a common aligned observation window using listwise-complete observations** (dropping dates where any constituent asset lacks a valid tick). This guarantees that the sample covariance matrix represents a mathematically consistent inner product across identical sample points.
   - **Pairwise Complete Caveat:** Estimating covariance using pairwise-complete observations evaluates each asset pair $(i, j)$ over different subsets of trading dates. This breaks transitivity and **can produce a non-positive-semidefinite (non-PSD) covariance matrix with negative eigenvalues**. If pairwise estimation is enabled as an option, the engine explicitly tests $\lambda_{min} > 0$ and checks the condition number $\kappa(\boldsymbol{\Sigma}) = \lambda_{max}/\lambda_{min}$. If validation fails, Higham's algorithm is applied; if the matrix remains degenerate ($\kappa > 10^6$), the matrix is rejected and an error is logged.
2. **Portfolio Valuation:** Assets not trading on date $t$ are carried at their last traded price, flagged with `is_stale=True` and documented in valuation audit metadata.
3. **Return Calculation:** Multi-day returns over missing intervals ($R_{t-k \to t} = P_t / P_{t-k} - 1$) are computed across observed ticks or unaligned observation pairs are excluded. Synthetic zero returns are never inserted.
4. **ML/DL Model Training:** Dates with missing features or targets are masked or dropped. Backward-filling (`bfill`) is strictly prohibited.
5. **Backtesting & Execution Simulation:** If an asset is suspended or unobserved on a scheduled rebalance date, execution orders in that asset are blocked; existing holding weights drift without stale-price trading.

---

## 4. Covariance Estimation & Numerical Regularization

### 4.1 Covariance Estimator Taxonomy

#### 1. Sample Covariance Matrix ($\mathbf{S}$)
$$\mathbf{S} = \frac{1}{T - 1} \sum_{t=1}^T (\mathbf{R}_t - \bar{\mathbf{R}})(\mathbf{R}_t - \bar{\mathbf{R}})^T$$
- *Characteristics:* Unbiased on listwise-aligned data, but numerically unstable when $N$ is large relative to $T$, resulting in extreme weights in Markowitz optimization.

#### 2. Standard `sklearn.covariance.LedoitWolf` Shrinkage (Default)
$$\boldsymbol{\Sigma}_{SKLW} = (1 - \hat{\delta}) \mathbf{S} + \hat{\delta} \mu_{var} \mathbf{I}$$
where $\mu_{var} = \frac{\text{Tr}(\mathbf{S})}{N}$ and $\hat{\delta} \in [0, 1]$ is the analytically optimal shrinkage intensity.
- *Target:* Scaled identity matrix.
- *Methodology:* Shrinkage generally improves matrix conditioning and reduces estimation noise. However, it does **not** unconditionally guarantee invertibility in every possible degenerate situation (e.g. constant prices or rank-zero inputs).
- *Mandatory Validation:* The resulting matrix must undergo explicit eigenvalue verification ($\lambda_{min} > 0$) and condition number checks ($\kappa(\boldsymbol{\Sigma}) < 10^6$).

#### 3. Constant-Correlation Shrinkage (Ledoit & Wolf, 2004)
$$\boldsymbol{\Sigma}_{CC} = (1 - \hat{\delta}) \mathbf{S} + \hat{\delta} \mathbf{F}$$
where $\mathbf{F}$ is the structured constant-correlation target:
$$F_{ii} = S_{ii}, \quad F_{ij} = \bar{\rho} \sqrt{S_{ii} S_{jj}} \quad (i \ne j), \quad \bar{\rho} = \frac{2}{N(N-1)} \sum_{i < j} \frac{S_{ij}}{\sqrt{S_{ii} S_{jj}}}$$
- *Target:* Preserves individual asset sample variances while shrinking pairwise correlations toward their cross-sectional mean $\bar{\rho}$.
- *Role:* Configurable specialized alternative; clearly distinguished from the scaled-identity target.

### 4.2 Positive Definite (PD) vs. Positive Semi-Definite (PSD) & Higham Covariance Handling

#### 4.2.1 Mathematical Distinction: PSD vs. Strictly PD
- **Positive Semi-Definite (PSD):** All eigenvalues satisfy $\lambda_i \ge 0$. While non-negative variance is guaranteed, eigenvalues equal to zero ($\lambda_{min} = 0$) signify a singular matrix with non-trivial null spaces, preventing matrix inversion.
- **Strictly Positive Definite (PD):** All eigenvalues satisfy $\lambda_i > 0$. In quadratic portfolio optimization, strict positive definiteness ($\lambda_{min} \ge \epsilon_{pd} = 10^{-7}$) is required to guarantee a strictly convex quadratic objective, a unique optimal solution, and numerical stability in Karush-Kuhn-Tucker (KKT) systems.

#### 4.2.2 Comprehensive Higham (2002) Projection & Positive-Definite Regularization Procedure
Higham's nearest correlation matrix algorithm operates strictly on correlation matrices (symmetric matrices with unit diagonal). To project an empirical or ill-conditioned covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{N \times N}$ onto the cone of strictly positive definite covariance matrices while preserving individual asset variances, the engine executes the following six-step procedure:

1. **Pre-Standardization Variance Validation & Asset Quarantine:**
   Before computing standard deviations, validate that all diagonal variance elements are finite and strictly positive:
   $$\sigma_i^2 = \boldsymbol{\Sigma}_{ii} > 0 \quad \text{and} \quad \text{isfinite}(\sigma_i^2) \quad \forall i \in \{1, \dots, N\}$$
   If any asset exhibits zero variance ($\sigma_i^2 = 0$, e.g., completely frozen price), negative variance, or non-finite values (`NaN`/`Inf`), that asset is quarantined or excluded from the universe prior to standardization. Division by zero in $\mathbf{D}^{-1/2}$ is strictly prevented.

2. **Standardization to Correlation:**
   Form the diagonal standard deviation matrix $\mathbf{D}^{1/2} = \text{diag}(\sigma_1, \dots, \sigma_N)$ and compute the standardized empirical correlation matrix:
   $$\mathbf{C} = \mathbf{D}^{-1/2} \boldsymbol{\Sigma} \mathbf{D}^{-1/2}, \quad C_{ij} = \frac{\Sigma_{ij}}{\sigma_i \sigma_j}$$

3. **Higham Alternating Dykstra Projection:**
   Apply Higham's (2002) alternating projection algorithm to $\mathbf{C}$. Higham's algorithm alternates projections between the closed convex cone of positive semi-definite matrices $\mathcal{S}_+$ and the affine subspace of symmetric matrices with unit diagonal $\mathcal{U} = \{ \mathbf{A} \in \mathcal{S}^N : A_{ii} = 1 \}$, yielding the nearest positive semi-definite correlation matrix:
   $$\mathbf{C}_{proj} = \text{Higham}(\mathbf{C})$$

4. **Controlled Positive-Definite Regularization (Eigenvalue Floor & Renormalization):**
   Higham's projection guarantees positive semi-definiteness ($\lambda_{min} \ge 0$), but the resulting matrix may lie on the boundary of the cone with singular zero eigenvalues ($\lambda_{min} = 0$). To guarantee strict positive definiteness ($\lambda_{min} \ge \epsilon_{floor}$) for non-singular quadratic optimization:
   a. Perform spectral decomposition: $\mathbf{C}_{proj} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^T$, where $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_N)$.
   b. Apply a controlled eigenvalue floor:
      $$\tilde{\boldsymbol{\Lambda}} = \text{diag}(\max(\lambda_i, \epsilon_{floor})), \quad \epsilon_{floor} = 10^{-7}$$
      $$\tilde{\mathbf{C}} = \mathbf{V} \tilde{\boldsymbol{\Lambda}} \mathbf{V}^T$$
   c. Renormalize the diagonal to restore exact unit diagonals:
      $$\mathbf{S}_{diag} = \text{diag}(\tilde{\mathbf{C}}), \quad \mathbf{C}_{reg} = \mathbf{S}_{diag}^{-1/2} \tilde{\mathbf{C}} \mathbf{S}_{diag}^{-1/2}$$
   *Numerical Impact:* The eigenvalue floor adds a minimal spectral perturbation $\sim 10^{-7}$ along degenerate eigenvectors while diagonal renormalization ensures $C_{reg, ii} \equiv 1.0$, producing a well-conditioned correlation matrix with minimal distortion to cross-asset correlation structure.

5. **Rescaling to Covariance:**
   Rescale the regularized correlation matrix using the original asset standard deviations:
   $$\boldsymbol{\Sigma}_{reg} = \mathbf{D}^{1/2} \mathbf{C}_{reg} \mathbf{D}^{1/2}$$
   Because $\mathbf{C}_{reg}$ has exact unit diagonals, the diagonal elements of $\boldsymbol{\Sigma}_{reg}$ preserve the original asset variances exactly: $\boldsymbol{\Sigma}_{reg, ii} = \sigma_i^2 \cdot 1.0 = \sigma_i^2$.

6. **Post-Rescaling Multi-Criteria Validation & Rejection Gate:**
   Before releasing $\boldsymbol{\Sigma}_{reg}$ to the optimizer, execute five mandatory validation assertions:
   a. **Symmetry:** $\|\boldsymbol{\Sigma}_{reg} - \boldsymbol{\Sigma}_{reg}^T\|_\infty \le 10^{-12}$.
   b. **Variance Preservation:** $\max_i |\boldsymbol{\Sigma}_{reg, ii} - \sigma_i^2| / \sigma_i^2 \le 10^{-6}$.
   c. **Strict Positive Definiteness:** $\lambda_{min}(\boldsymbol{\Sigma}_{reg}) \ge 10^{-7}$.
   d. **Condition Number:** $\kappa(\boldsymbol{\Sigma}_{reg}) = \frac{\lambda_{max}}{\lambda_{min}} < 10^6$.
   e. **Finite Tensors:** $\text{all}(\text{isfinite}(\boldsymbol{\Sigma}_{reg}))$.

   **Rejection Gate:** If any validation assertion fails, **the matrix is rejected and a detailed diagnostic exception is raised (`CovarianceValidationError`)**. It is **never** silently passed to the optimizer.

---

## 5. Modern Portfolio Theory Optimization (Module A)

### 5.1 Optimization Variables & Constraints
- Weight vector: $\mathbf{w} = [w_1, w_2, \dots, w_N]^T \in \mathbb{R}^N$.
- **Full Investment Constraint (Budget):**
  $$\sum_{i=1}^N w_i = \mathbf{1}^T \mathbf{w} = 1.0$$
- **Long-Only Constraint (No Short Selling, No Leverage):**
  $$0.0 \le w_i \le 1.0 \quad \forall i \in \{1, \dots, N\}$$

### 5.2 Global Minimum Variance (GMV) Portfolio
$$\min_{\mathbf{w}} \quad \frac{1}{2} \mathbf{w}^T \boldsymbol{\Sigma}_{ann} \mathbf{w} \quad \text{s.t.} \quad \mathbf{1}^T \mathbf{w} = 1, \quad 0 \le w_i \le 1 \quad \forall i$$
Analytical gradient supplied to solver:
$$\nabla_{\mathbf{w}} f(\mathbf{w}) = \boldsymbol{\Sigma}_{ann} \mathbf{w}$$

### 5.3 Maximum Sharpe Ratio (Tangency) Portfolio
$$\max_{\mathbf{w}} \quad \text{Sharpe}(\mathbf{w}) = \frac{\mathbf{w}^T \boldsymbol{\mu}_{ann} - r_{f, ann}}{\sqrt{\mathbf{w}^T \boldsymbol{\Sigma}_{ann} \mathbf{w}}}$$
The solver minimizes the negative Sharpe ratio using consistently annualized metrics and numerical stability constant $\epsilon = 10^{-10}$:
$$g(\mathbf{w}) = - \frac{\mathbf{w}^T \boldsymbol{\mu}_{ann} - r_{f, ann}}{\sigma_p(\mathbf{w})}$$
where $\sigma_p(\mathbf{w}) = \sqrt{\mathbf{w}^T \boldsymbol{\Sigma}_{ann} \mathbf{w} + \epsilon}$. The analytical Jacobian is:
$$\nabla_{\mathbf{w}} g(\mathbf{w}) = - \left[ \frac{\boldsymbol{\mu}_{ann}}{\sigma_p(\mathbf{w})} - \frac{(\mathbf{w}^T \boldsymbol{\mu}_{ann} - r_{f, ann}) \boldsymbol{\Sigma}_{ann} \mathbf{w}}{\sigma_p(\mathbf{w})^3} \right]$$
Analytical Jacobians improve gradient precision and avoid numerical difference truncation errors. However, analytical gradients do not unconditionally guarantee superlinear convergence across non-convex, flat, or ill-conditioned parameter surfaces.

### 5.4 Dual-Solver Validation & Calibrated Numerical Tolerances
Tolerances are not treated as universal assumptions across all asset universes or time horizons; they are treated as **illustrative initial defaults that must be calibrated against representative test datasets and solver behavior during implementation testing**:
1. **Constraint Satisfaction:** Strict tolerance $|\sum w_i - 1.0| \le 10^{-6}$ and $-10^{-8} \le w_i \le 1.0 + 10^{-8}$.
2. **Scale-Aware Objective Parity:** Evaluated using combined absolute and relative tolerances:
   $$|\text{Obj}^{custom} - \text{Obj}^{pypopt}| \le \text{tol}_{abs} + \text{tol}_{rel} \cdot |\text{Obj}^{pypopt}|$$
   - *Variance Objectives (GMV, Frontier):* Illustrative initial default: $\text{tol}_{abs} = 10^{-5}$, $\text{tol}_{rel} = 10^{-4}$ (reflecting typical variance scale $\sim 10^{-4}$ to $10^{-2}$).
   - *Sharpe Ratio Objectives:* Illustrative initial default: $\text{tol}_{abs} = 10^{-4}$, $\text{tol}_{rel} = 10^{-4}$ (reflecting typical Sharpe scale $\sim 0.5$ to $2.0$).
3. **Solver Convergence Status:** Verification that the optimization termination code indicates successful convergence.
4. **Feasibility & Numerical Stability:** Verification across perturbed initialization vectors.
5. **Weight Parity:** Evaluated only when the problem is strictly convex and the covariance matrix is strictly positive definite (guaranteeing a unique theoretical optimum).

### 5.5 Approved Optimizer Fallback Policy (Default: `RAISE_EXCEPTION`)
If numerical optimization fails to converge (e.g. collinear inputs, extreme market shocks):
- The system logs the specific failure code and diagnostics.
- The optimizer executes the configured fallback policy:
  1. **`RAISE_EXCEPTION` (Approved Default for Research/Optimization):** Aborts the optimization run and raises a detailed error containing matrix condition numbers and solver exit diagnostics.
  2. `EQUAL_WEIGHT`: Allocates $w_i = 1/N$.
  3. `PREVIOUS_WEIGHT`: Carries forward portfolio weights from the prior valid period.
  4. `HOLD_CASH`: Allocates 100% to cash.
- The result payload explicitly flags `fallback_executed: true`, naming the fallback policy and detailing the failure cause. **Silent substitution is strictly prohibited.**

---

## 6. Downside & Tail Risk Analytics

### 6.1 Value at Risk (VaR)
Measures the maximum potential loss at confidence level $1 - \alpha$ over a 1-day horizon.

#### 6.1.1 Parametric VaR (Normal Distribution)
$$\text{VaR}_\alpha^{\text{param}} = - \left( \mu_p + z_\alpha \sigma_p \right)$$
where $z_{0.05} = -1.6449$ (95%) and $z_{0.01} = -2.3263$ (99%).

#### 6.1.2 Historical VaR (Non-Parametric)
$$\text{VaR}_\alpha^{\text{hist}} = - \text{Quantile}_\alpha(\{ R_{p, 1}, R_{p, 2}, \dots, R_{p, T} \})$$

#### 6.1.3 Modified Cornish-Fisher VaR
Adjusts for empirical skewness ($S$) and excess kurtosis ($K$):
$$\tilde{z}_\alpha = z_\alpha + \frac{1}{6}(z_\alpha^2 - 1)S + \frac{1}{24}(z_\alpha^3 - 3z_\alpha)K - \frac{1}{36}(2z_\alpha^3 - 5z_\alpha)S^2$$
$$\text{VaR}_\alpha^{\text{CF}} = - (\mu_p + \tilde{z}_\alpha \sigma_p)$$

### 6.2 Conditional Value at Risk (CVaR / Expected Shortfall)
$$\text{CVaR}_\alpha = - \frac{1}{\sum_{t=1}^T \mathbb{I}(R_{p, t} \le -\text{VaR}_\alpha)} \sum_{t: R_{p, t} \le -\text{VaR}_\alpha} R_{p, t}$$

---

## 7. Performance & Backtesting Metrics (Module C)

### 7.1 Conventional Daily-Excess-Return Sharpe Ratio
In historical backtesting, the Sharpe Ratio is calculated from daily excess returns:
$$\text{Sharpe} = \sqrt{252} \cdot \frac{\text{mean}(R_{p, t} - r_{f, daily})}{\text{std}(R_{p, t} - r_{f, daily})}$$
where $r_{f, daily} = (1 + r_{f, ann})^{1/252} - 1$.

### 7.2 Sortino Ratio
Measures excess return per unit of downside risk relative to a Minimum Acceptable Return (MAR):
$$\text{Downside Semi-Deviation} = \sqrt{252} \cdot \sqrt{\frac{1}{T} \sum_{t=1}^T \min(0, R_{p, t} - \text{MAR})^2}$$
$$\text{Sortino} = \sqrt{252} \cdot \frac{\text{mean}(R_{p, t} - \text{MAR})}{\text{Downside Semi-Deviation}}$$
where MAR defaults to $r_{f, daily}$ (or $0.0$, configurable). Both numerator and denominator use consistent 252-day annualization.

### 7.3 Maximum Drawdown (MDD) & Calmar Ratio
$$HWM_t = \max_{0 \le s \le t} V_s, \quad DD_t = \frac{V_t - HWM_t}{HWM_t}, \quad \text{MDD} = \min_{0 \le t \le T} DD_t$$
$$\text{Calmar} = \frac{\text{CAGR}}{|\text{MDD}|}$$

### 7.4 Realized Compound Annual Growth Rate (CAGR)
$$\text{CAGR} = \left(\frac{V_T}{V_0}\right)^{\frac{252}{T}} - 1$$

### 7.5 Experiment Reproducibility Metadata
Every backtest records an immutable reproducibility manifest:
- `data_snapshot_id`: Version hash of historical prices used.
- `git_commit_sha`: Exact code version executing the backtest.
- `dependency_lock_hash`: Pinned environment package hash.
- `model_artifact_uri`: Serialized model checkpoint version.
- `random_seed`: Seed used for simulations and stochastic models.
- `optimizer_config`: Solver, tolerances, maximum iterations, and fallback policy.
- `execution_assumptions`: Execution latency, trade timing (Close $t-1 \to$ Open $t$).
- `cost_model_config`: Statutory rates, brokerage model, and slippage parameter $\gamma$.

---

## 8. Configurable Date-Aware Indian Transaction Cost Framework

Quantamental AI rejects hardcoding broker-specific rates as universal constants. All costs are separated into modular, configurable, date-aware components:

### 8.1 Transaction Cost Component Structure

| Cost Component | Initial Research Baseline | Applicable Leg | Date-Aware Context |
| :--- | :--- | :--- | :--- |
| **Securities Transaction Tax (STT)** | 0.10% (delivery) | Buy & Sell | Statutory rate set by Indian Finance Act. |
| **Stamp Duty** | 0.015% (₹1,500/crore) | Buy only | Unified Indian Stamp Act schedule (effective July 1, 2020). |
| **Exchange Transaction Charges** | 0.00345% | Buy & Sell | National Stock Exchange delivery tariff schedule. |
| **SEBI Turnover Fees** | ₹10 per crore (0.0001%) | Buy & Sell | SEBI Regulatory Fee Schedule. |
| **Brokerage** | Configurable (Default: flat ₹20/order or 0.05%) | Buy & Sell | Broker-specific model (zero, discount, or institutional bps). |
| **Goods & Services Tax (GST)** | 18% | Buy & Sell | Applicable on (Brokerage + Exchange Charges + SEBI Fees). |
| **Market Slippage** | $\Delta P = \gamma \cdot \left(\frac{\text{Trade Value}}{\text{ADV}}\right) \cdot P$ | Buy & Sell | Linear volume impact model calibrated to stock liquidity. |

---

## 9. Market Regime Detection & Anti-Leakage Protocols

### 9.1 Gaussian Hidden Markov Model (HMM) Methodology
1. **Arbitrary State Numbering:** HMM state IDs (0, 1, 2) are arbitrary random indices generated during Expectation-Maximization (EM) convergence.
2. **Post-Hoc Economic Interpretation:** The system sorts discovered states by their estimated empirical parameters (mean return, return variance) and dynamically assigns economic labels:
   - Lowest variance, positive drift $\to$ "Calm Bull Regime"
   - Highest variance, negative drift $\to$ "High-Volatility Crisis Regime"
   - Moderate variance, near-zero drift $\to$ "Sideways / Transitional Regime"
3. **Temporal Integrity (Anti-Leakage Requirements):**
   - During historical backtesting, trading decisions MUST use **filtered state probabilities** $P(S_t \mid y_{1:t})$ calculated strictly via the forward algorithm using data up to $t$.
   - **Smoothed probabilities** $P(S_t \mid y_{1:T})$, which incorporate future observations $y_{t+1:T}$, are strictly prohibited in historical strategy execution.
4. **Scope Differentiation:**
   - Initial Core Scope: 3-state Gaussian HMM with filtered inference (Stage 4 of AI Lab).
   - Optional Research Extensions: Non-homogeneous HMMs, autoregressive regime switching, Student's t-emission distributions.
