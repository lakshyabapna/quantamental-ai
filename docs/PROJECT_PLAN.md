# Comprehensive Project Plan & Implementation Roadmap — Quantamental AI

**Author:** Senior Quantitative Developer, Financial Engineer & Software Architect
**Project:** Quantamental AI
**Status:** Milestone Plan — Canonical Sign-off Edition (Approved Decisions)
**Version:** 1.4.0

---

## 1. Executive Summary & Phased Architecture

Quantamental AI is structured into **seven sequential, verifiable implementation phases**. Each phase contains explicit prerequisites, deliverables, validation criteria, and a formal **Definition of Done (DoD)**. No phase begins until the preceding phase satisfies its mathematical and engineering audit criteria.

```
Phase 0: Architecture & Mathematical Alignment [COMPLETED - CANONICAL SPECIFICATION]
   │
   ▼
Phase 1: Foundation, Persistent Storage & Incremental Ingestion Pipeline
   │
   ▼
Phase 2: Classical Financial Mathematics & Multi-Criteria Audited Optimization Core
   │
   ▼
Phase 3: Staged AI Risk Lab (Stage 1 Baselines -> Stage 2 XGBoost -> Stage 3 LSTM -> Stage 4 HMM)
   │
   ▼
Phase 4: Historical Backtesting, Date-Aware Indian Microstructure & Attribution
   │
   ▼
Phase 5: FastAPI Backend Services & REST API Layer
   │
   ▼
Phase 6: Interactive React / Vite Quantitative Dashboard
   │
   ▼
Phase 7: Cloud Deployment, Security Hardening & Continuous Integration
```

---

## 2. Detailed Phase Breakdown

### Phase 0: System Architecture & Alignment (Completed)
- **Goal:** Establish unassailable mathematical, architectural, and data governance specifications before writing application code.
- **Deliverables:**
  - `ARCHITECTURE.md` (System design, folder structure, API schemas, validation, storage architecture).
  - `docs/FINANCIAL_METHODOLOGY.md` (Complete mathematical formulas, covariance estimators, PSD/PD corrections, Sharpe conventions, Indian cost model).
  - `docs/DATA_STRATEGY.md` (10-stock NSE universe, rolling 5-year window, incremental refresh lifecycle, missing-data policies).
  - `docs/PROJECT_PLAN.md` (This document).
- **Validation Criteria:** Complete cross-document consistency; alignment with engineering and financial principles.

---

### Phase 1: Foundation, Persistent Storage & Incremental Ingestion Pipeline
- **Prerequisites:** Phase 0 canonical sign-off.
- **Tasks & Subtasks:**
  1. Initialize approved standard Python virtual environment (`python -m venv venv`), standard package installer (`pip`), and `pyproject.toml` project configuration.
  2. Implement `BaseDataProvider` abstract interface (`app/data_engine/base.py`).
  3. Implement `NSECalendar` (`app/data_engine/calendar.py`) managing official NSE trading calendars (sourced from annual NSE circulars, stored in `nse_trading_calendar`, and updated annually).
  4. Implement `YFinanceProvider` (`app/data_engine/yfinance_provider.py`) with provider-agnostic price normalization:
     - Map nominal OHLC and `Adj Close` separately when `auto_adjust=False`.
     - Record nominal OHLC as unavailable when `auto_adjust=True`; never populate nominal fields with adjusted prices.
     - Never infer split or dividend amounts from adjusted OHLC alone; capture from dedicated corporate action streams.
     - Record provider, adjustment mode, retrieval timestamp, and validation status.
  5. Implement `DataCleansingPipeline` (`app/data_engine/clean_pipeline.py`) enforcing configurable, asset-specific missing-data thresholds (distinguishing warnings, exclusions, and quarantine; strictly no `bfill`).
  6. Implement **persistent historical retention** and **incremental data fetching** (`app/data_engine/incremental.py`):
     - Identify latest successfully stored trading date per asset.
     - Fetch incremental batch with configurable overlap (5–10 trading days) to reconcile recent provider updates.
     - Upsert records by composite key `(symbol, trading_date)` preventing duplicates.
     - Preserve existing valid historical data if refresh fails.
  7. Implement broader historical reconciliation engine (`app/data_engine/reconciliation.py`) for periodic full-window checks, creating version snapshots for backtest reproducibility-supporting controls.
  8. Implement dynamic rolling five-year window slicer selecting the active research dataset from persistent storage.
  9. Implement `data_refresh_logs` metadata tracking.
- **Deliverables:**
  - Data ingestion engine with persistent storage, incremental update, and broader reconciliation capabilities.
  - Approved 10-stock historical dataset (`RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`, `INFY.NS`, `HINDUNILVR.NS`, `LT.NS`, `ITC.NS`, `BHARTIARTL.NS`, `MARUTI.NS`) and benchmark `^NSEI`.
  - PyTest suite verifying incremental refresh, overlap reconciliation, and upsert deduplication (`tests/integration/test_data_pipeline.py`).
- **Definition of Done (DoD):**
  - Incremental refresh correctly fetches only new/overlap data and prevents duplicates.
  - Dynamic 5-year window correctly resolves `T_end` (latest completed session) and `T_start`.
  - Stored dataset satisfies configurable asset quality thresholds with zero backward-filled values.

---

### Phase 2: Classical Financial Mathematics & Multi-Criteria Audited Optimization Core (Module A)
- **Prerequisites:** Phase 1 complete.
- **Tasks & Subtasks:**
  1. Implement `returns.py`: simple returns for portfolio aggregation, log returns for ML, arithmetic annualized return, and geometric CAGR.
  2. Implement `covariance.py`:
     - Standard listwise-aligned common window estimator as the default for the approved 10-stock MVP.
     - Pairwise-complete estimator with explicit strict positive definiteness ($\lambda_{min} \ge 10^{-7}$) and condition number checks ($\kappa < 10^6$).
     - Standard `sklearn.covariance.LedoitWolf` (scaled-identity target) with condition checks.
     - Constant-Correlation shrinkage (Ledoit & Wolf, 2004) as separately named configurable estimator.
  3. Implement `psd_projection.py`:
     - Pre-standardization validation: assert $\sigma_i^2 > 0$ and isfinite; quarantine/exclude zero-variance assets.
     - Standardize covariance to correlation $\mathbf{C} = \mathbf{D}^{-1/2} \boldsymbol{\Sigma} \mathbf{D}^{-1/2}$.
     - Apply Higham's (2002) alternating projection to obtain nearest correlation matrix $\mathbf{C}_{proj}$.
     - Controlled positive-definite regularization: apply eigenvalue floor $\tilde{\boldsymbol{\Lambda}} = \max(\lambda_i, 10^{-7})$ followed by diagonal renormalization $\mathbf{C}_{reg} = \text{diag}(\tilde{\mathbf{C}})^{-1/2} \tilde{\mathbf{C}} \text{diag}(\tilde{\mathbf{C}})^{-1/2}$ restoring exact unit diagonals.
     - Rescale $\boldsymbol{\Sigma}_{reg} = \mathbf{D}^{1/2} \mathbf{C}_{reg} \mathbf{D}^{1/2}$, exactly preserving original asset variances.
     - Post-rescaling validation gate: assert symmetry, variance preservation, strict positive definiteness ($\lambda_{min} \ge 10^{-7}$), and condition number ($\kappa < 10^6$); reject and raise diagnostic error if validation fails.
     - Fast eigenvalue clipping alternative.
  4. Implement `optimizer.py`:
     - Global Minimum Variance (GMV) portfolio via SciPy `SLSQP` with analytical gradients.
     - Maximum Sharpe Ratio (Tangency) portfolio with analytical Jacobian using consistent annualized metrics ($\boldsymbol{\mu}_{ann}, \boldsymbol{\Sigma}_{ann}, r_{f, ann}$) and stability constant $\epsilon = 10^{-10}$.
     - Target return constrained optimization for continuous Efficient Frontier mapping.
     - Capital Allocation Line (CAL) slope and coordinate generator.
     - Approved optimizer fallback handling: **`RAISE_EXCEPTION` as default**, with `EQUAL_WEIGHT`, `PREVIOUS_WEIGHT`, and `HOLD_CASH` explicitly configurable.
  5. Calibrate numerical tolerances against representative empirical test datasets, verifying constraint satisfaction ($|\sum w - 1| \le 10^{-6}$) and scale-aware objective parity ($|\Delta f| \le \text{tol}_{abs} + \text{tol}_{rel} \cdot |f|$) against `PyPortfolioOpt`.
  6. Implement `monte_carlo.py`: vectorized Dirichlet simulation generator for frontier scatter visualization.
  7. Implement `risk_metrics.py`: Parametric VaR, Historical VaR, Cornish-Fisher VaR, and Conditional Value at Risk (CVaR).
  8. PyTest suite:
     - 2-asset closed-form analytical validation ($|w - w^*| \le 10^{-5}$).
     - Dual-solver scale-aware objective parity checks across empirical test cases.
     - PSD/PD projection and condition checks.
- **Deliverables:**
  - Standalone, audited `app/financial_engine/` package.
  - Multi-criteria validation suite in `tests/unit/test_optimizer.py` and `tests/unit/test_validation.py`.
- **Definition of Done (DoD):**
  - All test cases pass within calibrated scale-aware numerical tolerances.
  - Optimizer explicitly logs and executes the approved fallback (`RAISE_EXCEPTION` by default) when supplied with singular/degenerate matrices.

---

### Phase 3: Staged AI Risk Lab (Module B)
- **Prerequisites:** Phase 2 complete.
- **Tasks & Subtasks:**
  Implementation is strictly partitioned into sequential validation gates:
  1. **Stage 3.1 — Historical Baselines:**
     - Implement `baselines.py`: Historical Rolling Volatility baseline, Rolling Mean, Ridge/OLS.
     - *Validation Gate 3.1:* Establish walk-forward benchmark error metrics (RMSE, MAE).
  2. **Stage 3.2 — Tree-Based Volatility Regressor:**
     - Implement `tree_models.py`: XGBoost regressor for forward realized volatility forecasting.
     - Implement `explainability.py`: TreeSHAP feature attributions for XGBoost.
     - *Validation Gate 3.2:* Demonstrate measurable error reduction relative to Stage 3.1 baseline on out-of-sample walk-forward test folds.
  3. **Stage 3.3 — Deep Learning Sequence Model:**
     - Implement `deep_models/lstm.py`: PyTorch LSTM sequence network.
     - *Validation Gate 3.3:* Achieve training convergence and sequential loss reduction without gradient explosion; cross-validate performance against Stage 3.2.
  4. **Stage 3.4 — Market Regime Detection:**
     - Implement `regime_detection.py`: Gaussian Hidden Markov Model (HMM) with post-hoc parameter labeling.
     - *Validation Gate 3.4:* Anti-leakage verified by asserting strictly **filtered state probabilities** $P(S_t \mid y_{1:t})$ during backtesting.
  5. Label advanced extensions (GRU, TCN, GARCH, higher-order HMMs) as optional future research modules.
- **Deliverables:**
  - Staged model evaluation leaderboard (Stage 1 Baselines vs Stage 2 XGBoost vs Stage 3 LSTM).
  - Filtered Stage 4 HMM regime timeline overlay on Nifty 50.
  - TreeSHAP summary charts.
- **Definition of Done (DoD):**
  - Staged validation gates pass sequentially prior to advancing.
  - Anti-leakage verified by automated assertions.

---

### Phase 4: Historical Backtesting, Date-Aware Indian Microstructure & Attribution Engine (Module C)
- **Prerequisites:** Phase 2 and Phase 3 complete.
- **Tasks & Subtasks:**
  1. Implement `cost_model.py`: modular, date-aware Indian equity cost engine (statutory STT, Stamp Duty, Exchange turnover fees, SEBI fees, configurable brokerage, GST, and volume slippage).
  2. Implement `simulator.py`: discrete-time historical rebalancing loop with 1-period execution latency (Close $t-1 \to$ Open/Close $t$):
     - Dynamic 5-year rolling slice from persistent store.
     - Daily mark-to-market accounting.
     - Turnover tracking and transaction cost deduction.
     - Manifest recording: data snapshot ID, git commit SHA, dependency lock hash, model artifact URI, random seed, optimizer config, execution assumptions, and cost model config.
  3. Implement `benchmark.py`: Benchmark integration (Nifty 50 TRI preferred subject to licensing; price index accurately labeled if TRI unavailable) and Equal-Weight (1/N) benchmark.
  4. Implement `performance.py`:
     - Conventional daily-excess-return Sharpe ratio: $\sqrt{252} \cdot \frac{\text{mean}(R_{p, t} - r_{f, daily})}{\text{std}(R_{p, t} - r_{f, daily})}$.
     - Sortino ratio with explicit MAR ($r_{f, daily}$ default or $0.0$).
     - Realized CAGR as a separate multi-period growth metric.
     - Maximum Drawdown (MDD) and Calmar ratio.
  5. PyTest suite (`tests/unit/test_cost_model.py`): assert date-aware fee calculations against verified statutory schedules.
- **Deliverables:**
  - Backtesting engine capable of side-by-side comparative simulation: Classical MPT vs AI-Regime Adaptive vs Equal Weight (1/N) vs Benchmark.
  - Complete rebalance audit ledger and drawdown series.
- **Definition of Done (DoD):**
  - All fees accurately deducted based on trade date and transaction leg.
  - Output metrics match hand-calculated test vectors on simulated trade sequences.

---

### Phase 5: FastAPI Backend Services & REST API Layer
- **Prerequisites:** Phases 1–4 complete.
- **Tasks & Subtasks:**
  1. Configure FastAPI app factory with CORS, security headers, exception handlers, and structured logging.
  2. Define Pydantic v2 schemas (`app/schemas/`) with strict validation boundaries.
  3. Implement API route controllers:
     - `/api/v1/data` (assets, prices within dynamic rolling window, refresh trigger, audit logs).
     - `/api/v1/optimization` (covariance selection, frontier, validation audit, explicit fallback status).
     - `/api/v1/ai` (staged forecasts, HMM regimes with filtered probabilities, SHAP).
     - `/api/v1/backtest` (run, compare strategies).
  4. Integration test suite (`tests/integration/test_api_endpoints.py`).
- **Deliverables:**
  - Interactive OpenAPI / Swagger documentation (`/docs`).
  - High-performance REST API with measurable benchmark targets ($p50 < 300\text{ ms}$, $p95 < 500\text{ ms}$ on 4-core hardware).
- **Definition of Done (DoD):**
  - 100% of endpoints conform to Pydantic v2 schemas.
  - Synchronous optimization requests meet benchmark latency targets.

---

### Phase 6: Interactive React / Vite Frontend Dashboard
- **Prerequisites:** Phase 5 complete.
- **Tasks & Subtasks:**
  1. Initialize Vite + React 18 + TypeScript project with Tailwind CSS.
  2. Implement state management with Zustand and server caching with `@tanstack/react-query`.
  3. Build Module A UI: Interactive Plotly Efficient Frontier, CAL, asset allocation charts, covariance heatmaps, and clear fallback indicators.
  4. Build Module B UI: Staged model leaderboard (Baseline vs XGBoost vs LSTM), filtered HMM regime timeline overlay, SHAP summary plots.
  5. Build Module C UI: Comparative equity curves, drawdown underwater chart, historical rebalance ledger with reproducibility manifest details.
- **Deliverables:**
  - Responsive, modern frontend SPA.
- **Definition of Done (DoD):**
  - Zero TypeScript compile errors.
  - Responsive chart interactions and parameter updates.

---

### Phase 7: Deployment, Containerization & CI/CD
- **Prerequisites:** Phase 6 complete.
- **Tasks & Subtasks:**
  1. Author production multi-stage Dockerfile for FastAPI backend.
  2. Deploy backend service to Render (decoupling persistent state from ephemeral container disks).
  3. Deploy frontend SPA to Vercel.
  4. Configure Supabase PostgreSQL production instance and PgBouncer connection pooling.
  5. Author end-to-end smoke test verifying client-to-backend-to-database communication.
- **Deliverables:**
  - Production web application accessible via HTTPS.
  - CI/CD workflow testing every commit.
- **Definition of Done (DoD):**
  - Both frontend and backend live and successfully communicating in cloud staging.

---

## 3. Risk Register & Technical Mitigation Matrix

| Risk ID | Description | Impact | Probability | Architectural Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| **R1** | **Covariance Matrix Singularity / Non-PD**<br>Collinear stock prices or pairwise estimation leads to non-invertible covariance or negative eigenvalues. | High | Medium | Prefer listwise-complete aligned window for MVP; implement standard `sklearn.covariance.LedoitWolf`, condition checks, and Higham standardize $\to$ project $\to$ rescale projection. |
| **R2** | **Optimizer Non-Convergence / Local Minima**<br>SLSQP optimizer fails to converge or stalls. | High | Low | Provide analytical Jacobians; evaluate multi-criteria validation against PyPortfolioOpt; implement approved default fallback (`RAISE_EXCEPTION`), with configurable alternatives. |
| **R3** | **Look-Ahead Bias in ML / Feature Engineering**<br>Features inadvertently leak future price returns, or smoothed HMM states are used in backtests. | Critical | Medium | Enforce anti-leakage temporal design ($\tau \le t - 1$); enforce **filtered probabilities** $P(S_t \mid y_{1:t})$ during backtesting; automated PyTest assertions. |
| **R4** | **Data Provider Throttling / Upstream Schema Changes**<br>`yfinance` undocumented endpoints get throttled or change schema. | High | High | Implement persistent local Parquet storage, incremental fetching with overlap, and abstract `BaseDataProvider` interface. |
| **R5** | **Cloud Container Ephemeral Disk Data Loss**<br>Render restarts clear local container disk. | Critical | Medium | Decouple durable state: store metadata/records in Supabase PostgreSQL; store model binaries in cloud object storage. |
| **R6** | **Corporate Action Split Distortions**<br>Stock splits create artificial $-50\%$ return drops. | Critical | Low | Exclusively calculate returns and models on `adj_close` prices; enforce validation gate 4 (split spike detector). |

---

## 4. Key Financial Assumptions

1. **Long-Only Portfolios:** Initial constraint enforces $w_i \ge 0$ (no short selling) and $\sum_{i=1}^N w_i = 1.0$ (no leverage, 100% equity invested at rebalance).
2. **Standard Annualization:** 252 trading days per year for Indian equity markets.
3. **Risk-Free Rate:** Baseline assumed at **6.50% annualized** ($r_f = 0.065$), calibrated to the 91-day Government of India Treasury Bill yield; converted daily via compounding $(1 + r_{f, ann})^{1/252} - 1$.
4. **Execution Latency:** Rebalancing decisions formulated at Close of Day $t-1$ execute at Open of Day $t$ (or Close of Day $t$ with 1-day lag), preventing same-moment execution look-ahead bias.
5. **Configurable Indian Transaction Microstructure:** Explicit deduction of STT, Stamp Duty, Exchange fees, SEBI fees, configurable brokerage, GST, and volume slippage.
6. **No Guaranteed Returns:** Quantamental AI is an academic and decision-support research platform; no strategy is presented as guaranteed profitable.

---

## 5. Approved Architectural Decisions

1. **Approved Initial Asset Universe:**
   The 10 NSE stocks (`RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`, `INFY.NS`, `HINDUNILVR.NS`, `LT.NS`, `ITC.NS`, `BHARTIARTL.NS`, `MARUTI.NS`) and benchmark `^NSEI` are formally recorded as **Approved**.
2. **Approved AI/ML Staged Scope:**
   Historical Volatility baseline (Stage 1), XGBoost volatility forecaster (Stage 2), PyTorch LSTM sequence model (Stage 3), and filtered Gaussian HMM (Stage 4) are formally recorded as **Approved**, developed through sequential validation gates (with GRU, TCN, GARCH, and advanced HMMs deferred to optional research extensions).
3. **Approved Default Optimizer Fallback Policy:**
   **`RAISE_EXCEPTION`** is formally recorded as the **Approved Default** fallback policy for research/optimization (with `EQUAL_WEIGHT`, `PREVIOUS_WEIGHT`, and `HOLD_CASH` explicitly configurable).
4. **Approved Packaging Standard:**
   Standard Python virtual environment (`venv`) + `pip` + `pyproject.toml` is formally recorded as the **Approved Standard**.
5. **Approved Benchmark Policy:**
   **Nifty 50 TRI** is the preferred benchmark subject to appropriate data licensing; otherwise, the price-index benchmark (`^NSEI`) is accurately labeled.

---

## 6. Recommended Next Implementation Task

Upon your final go-ahead, the exact first implementation task will be:

> **Phase 1, Task 1.1–1.3:** Setup the backend repository directory structure, author `pyproject.toml` with pinned dependencies, and implement the foundational `BaseDataProvider` interface (`app/data_engine/base.py`) and official `NSECalendar` module (`app/data_engine/calendar.py`).
