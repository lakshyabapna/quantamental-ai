"""Unit & integrity tests for ``app.ai_engine.registry``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from app.ai_engine import ModelManifest, ModelRegistry
from app.core.exceptions import ArtifactIntegrityError


class TestModelRegistry:
    """Verify artifact persistence, reproducibility manifests, and integrity checks."""

    def test_save_and_load_joblib_artifact_with_manifest_hashes(
        self, tmp_path: Path
    ) -> None:
        """Save and reload a scikit-learn artifact, verifying hashes and metadata."""
        registry = ModelRegistry(artifact_dir=tmp_path)

        rng = np.random.default_rng(42)
        X = rng.normal(size=(40, 4))
        y = X @ np.array([0.1, -0.2, 0.3, 0.05])
        model = Ridge(alpha=1.0).fit(X, y)

        manifest = registry.save_artifact(
            model,
            model_id="vol_ridge_baseline",
            version="1.0.0",
            stage="STAGE_1_BASELINE",
            artifact_format="joblib",
            hyperparameters={"alpha": 1.0},
            random_seed=42,
            data_snapshot_hash="abc123def456",
            train_start_date="2020-01-01",
            train_end_date="2024-12-31",
        )

        assert isinstance(manifest, ModelManifest)
        assert len(manifest.artifact_sha256) == 64
        assert len(manifest.manifest_sha256) == 64
        assert "scikit-learn" in manifest.dependencies
        assert "numpy" in manifest.dependencies

        loaded_model, loaded_manifest = registry.load_artifact("vol_ridge_baseline", "1.0.0")
        assert loaded_manifest == manifest
        assert isinstance(loaded_model, Ridge)
        np.testing.assert_allclose(
            np.asarray(loaded_model.predict(X), dtype=np.float64),
            np.asarray(model.predict(X), dtype=np.float64),
            atol=1e-12,
        )

    def test_save_and_load_state_dict_npz_without_torch(self, tmp_path: Path) -> None:
        """Verify state_dict_npz persistence works cleanly without importing PyTorch."""
        registry = ModelRegistry(artifact_dir=tmp_path)
        state = {
            "lstm.weight_ih": np.ones((8, 4), dtype=np.float32),
            "fc.bias": np.zeros((1,), dtype=np.float32),
        }

        manifest = registry.save_artifact(
            state,
            model_id="vol_lstm",
            version="0.1.0",
            stage="STAGE_3_LSTM",
            artifact_format="state_dict_npz",
            hyperparameters={"hidden_size": 16},
            random_seed=42,
            data_snapshot_hash="snap001",
            train_start_date="2021-01-01",
            train_end_date="2025-01-01",
        )
        assert manifest.artifact_filename.endswith(".npz")

        loaded, _ = registry.load_artifact("vol_lstm", "0.1.0")
        assert isinstance(loaded, dict)
        np.testing.assert_array_equal(loaded["lstm.weight_ih"], state["lstm.weight_ih"])
        np.testing.assert_array_equal(loaded["fc.bias"], state["fc.bias"])

    def test_corrupted_artifact_and_tampered_manifest_rejected(
        self, tmp_path: Path
    ) -> None:
        """Corrupted artifact bytes or tampered manifest fields must raise ArtifactIntegrityError."""
        registry = ModelRegistry(artifact_dir=tmp_path)
        payload = {"mean_vol": 0.185, "window": 21}

        manifest = registry.save_artifact(
            payload,
            model_id="vol_hist",
            version="1.0.0",
            stage="STAGE_1_BASELINE",
            artifact_format="json",
            hyperparameters={"window": 21},
            random_seed=42,
            data_snapshot_hash="snap_hash",
            train_start_date="2020-01-01",
            train_end_date="2024-01-01",
        )

        # 1. Corrupt artifact file bytes
        artifact_path = tmp_path / manifest.artifact_filename
        artifact_path.write_text('{"mean_vol": 9.999, "window": 21}', encoding="utf-8")
        with pytest.raises(ArtifactIntegrityError, match="SHA-256 mismatch"):
            registry.load_artifact("vol_hist", "1.0.0")

        # Restore artifact and tamper with manifest JSON instead
        artifact_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        manifest_path = tmp_path / "vol_hist__1.0.0.manifest.json"
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_manifest["random_seed"] = 999
        manifest_path.write_text(json.dumps(raw_manifest), encoding="utf-8")

        with pytest.raises(ArtifactIntegrityError, match="Manifest integrity"):
            registry.load_artifact("vol_hist", "1.0.0")

    def test_path_traversal_and_untrusted_format_rejected(self, tmp_path: Path) -> None:
        """Reject path traversal identifiers and unknown artifact formats."""
        registry = ModelRegistry(artifact_dir=tmp_path)

        with pytest.raises(ArtifactIntegrityError):
            registry.save_artifact(
                {"a": 1},
                model_id="../evil",
                version="1.0.0",
                stage="STAGE_1_BASELINE",
                artifact_format="json",
                data_snapshot_hash="x",
                train_start_date="2020-01-01",
                train_end_date="2024-01-01",
            )

        with pytest.raises(ArtifactIntegrityError):
            registry.load_artifact("nonexistent", "1.0.0")
