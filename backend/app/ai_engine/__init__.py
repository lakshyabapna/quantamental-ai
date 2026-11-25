"""Quantamental AI - AI Risk Lab Engine (Module B).

Step 3.0 Infrastructure & Anti-Leakage Core:
    - ``feature_pipeline``: Strictly causal lagged features (tau <= t-1) on log returns.
    - ``splitters``: Chronological expanding/rolling walk-forward splitters with purge gaps.
    - ``registry``: Local artifact persistence with SHA-256 integrity & manifests.
"""

from app.ai_engine.feature_pipeline import (
    DEFAULT_HORIZON_DAYS,
    DEFAULT_RETURN_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    FeatureDataset,
    build_volatility_features,
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
    "DEFAULT_HORIZON_DAYS",
    "DEFAULT_RETURN_LAGS",
    "DEFAULT_ROLLING_WINDOWS",
    "FeatureDataset",
    "ModelManifest",
    "ModelRegistry",
    "ScaledFoldData",
    "SplitMode",
    "WalkForwardSplit",
    "WalkForwardSplitter",
    "build_volatility_features",
    "scale_fold_features",
]
