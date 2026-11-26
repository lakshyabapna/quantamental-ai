"""Quantamental AI - AI Risk Lab Engine (Module B).

Step 3.0 Infrastructure & Anti-Leakage Core:
    - ``feature_pipeline``: Strictly causal lagged features (tau <= t-1) on log returns.
    - ``splitters``: Chronological expanding/rolling walk-forward splitters with purge gaps.
    - ``registry``: Local artifact persistence with SHA-256 integrity & manifests.

Step 3.1 Historical Baseline Models:
    - ``baselines``: RollingVolBaseline, RollingMeanBaseline, OLSBaseline, RidgeBaseline,
      and explicit ForecastFloorPolicy / apply_volatility_floor post-processing.
    - ``metrics``: RMSE, MAE, R², normalized Patton (2011) QLIKE, and BaselineMetrics.
    - ``evaluation``: Walk-forward evaluation harness (``evaluate_baselines``, ``aggregate_metrics``).

Step 3.2 XGBoost Volatility Forecasting + TreeSHAP Explainability:
    - ``tree_models``: XGBoostVolatilityRegressor (fixed), TunedXGBoostVolatilityRegressor
      (nested purged chronological tuning), and XGBoostHyperparameters.
    - ``explainability``: TreeSHAP global and local explanations with additivity checks.
"""

from app.ai_engine.baselines import (
    DEFAULT_MIN_ANNUALIZED_VOL,
    BaselinePredictor,
    ForecastFloorPolicy,
    OLSBaseline,
    PostProcessedForecast,
    RidgeBaseline,
    RollingMeanBaseline,
    RollingVolBaseline,
    apply_volatility_floor,
)
from app.ai_engine.evaluation import (
    DEFAULT_GATE32_MAX_MAE_DETERIORATION_PCT,
    DEFAULT_GATE32_MAX_QLIKE_DETERIORATION,
    STAGE1_BASELINE_NAMES,
    Gate32Report,
    aggregate_metrics,
    evaluate_baselines,
    evaluate_stage2_gate,
)
from app.ai_engine.explainability import (
    GlobalShapSummary,
    LocalShapExplanation,
    compute_global_shap,
    explain_single_prediction,
)
from app.ai_engine.feature_pipeline import (
    DEFAULT_HORIZON_DAYS,
    DEFAULT_RETURN_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    FeatureDataset,
    build_volatility_features,
)
from app.ai_engine.metrics import (
    BaselineMetrics,
    mae,
    qlike,
    r_squared,
    rmse,
)
from app.ai_engine.registry import (
    ArtifactFormat,
    ModelManifest,
    ModelRegistry,
)
from app.ai_engine.splitters import (
    ScaledFoldData,
    SplitMode,
    WalkForwardSplit,
    WalkForwardSplitter,
    scale_fold_features,
)
from app.ai_engine.tree_models import (
    DEFAULT_XGBOOST_PARAM_GRID,
    TunedXGBoostVolatilityRegressor,
    XGBoostHyperparameters,
    XGBoostVolatilityRegressor,
)

__all__ = [
    "ArtifactFormat",
    "BaselineMetrics",
    "BaselinePredictor",
    "DEFAULT_GATE32_MAX_MAE_DETERIORATION_PCT",
    "DEFAULT_GATE32_MAX_QLIKE_DETERIORATION",
    "DEFAULT_HORIZON_DAYS",
    "DEFAULT_MIN_ANNUALIZED_VOL",
    "DEFAULT_RETURN_LAGS",
    "DEFAULT_ROLLING_WINDOWS",
    "DEFAULT_XGBOOST_PARAM_GRID",
    "FeatureDataset",
    "ForecastFloorPolicy",
    "Gate32Report",
    "GlobalShapSummary",
    "LocalShapExplanation",
    "ModelManifest",
    "ModelRegistry",
    "OLSBaseline",
    "PostProcessedForecast",
    "RidgeBaseline",
    "RollingMeanBaseline",
    "RollingVolBaseline",
    "ScaledFoldData",
    "STAGE1_BASELINE_NAMES",
    "SplitMode",
    "TunedXGBoostVolatilityRegressor",
    "WalkForwardSplit",
    "WalkForwardSplitter",
    "XGBoostHyperparameters",
    "XGBoostVolatilityRegressor",
    "aggregate_metrics",
    "apply_volatility_floor",
    "build_volatility_features",
    "compute_global_shap",
    "evaluate_baselines",
    "evaluate_stage2_gate",
    "explain_single_prediction",
    "mae",
    "qlike",
    "r_squared",
    "rmse",
    "scale_fold_features",
]
