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
    aggregate_metrics,
    evaluate_baselines,
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

__all__ = [
    "ArtifactFormat",
    "BaselineMetrics",
    "BaselinePredictor",
    "DEFAULT_HORIZON_DAYS",
    "DEFAULT_MIN_ANNUALIZED_VOL",
    "DEFAULT_RETURN_LAGS",
    "DEFAULT_ROLLING_WINDOWS",
    "FeatureDataset",
    "ForecastFloorPolicy",
    "ModelManifest",
    "ModelRegistry",
    "OLSBaseline",
    "PostProcessedForecast",
    "RidgeBaseline",
    "RollingMeanBaseline",
    "RollingVolBaseline",
    "ScaledFoldData",
    "SplitMode",
    "WalkForwardSplit",
    "WalkForwardSplitter",
    "aggregate_metrics",
    "apply_volatility_floor",
    "build_volatility_features",
    "evaluate_baselines",
    "mae",
    "qlike",
    "r_squared",
    "rmse",
    "scale_fold_features",
]
