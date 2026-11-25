"""Quantamental AI - Local Model Artifact Registry & Reproducibility Manifest (Module B).

Manages versioned local persistence under configurable ``MODEL_ARTIFACT_DIR`` with:
    - Reproducibility manifest (model_id, version, config, seed, data snapshot hash,
      training date range, dependency versions, and SHA-256 artifact digest).
    - Integrity verification via constant-time SHA-256 comparison prior to loading.
    - Safe path validation (rejects directory traversal or unexpected formats).
    - Support for ``joblib`` and numpy/JSON state-dict formats without importing PyTorch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import importlib.metadata
import json
from pathlib import Path
import platform
import re
from typing import Any, Dict, Literal, Optional, Tuple, Union

import joblib
import numpy as np

from app.config import settings
from app.core.exceptions import ArtifactIntegrityError


ArtifactFormat = Literal["joblib", "json", "state_dict_npz"]
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_FORMAT_EXTENSIONS: Dict[str, str] = {
    "joblib": ".joblib",
    "json": ".json",
    "state_dict_npz": ".npz",
}


@dataclass(frozen=True)
class ModelManifest:
    """Immutable reproducibility manifest stored alongside every serialized model artifact."""

    model_id: str
    version: str
    stage: str
    artifact_format: ArtifactFormat
    artifact_filename: str
    hyperparameters: Dict[str, Any]
    random_seed: int
    data_snapshot_hash: str
    train_start_date: str
    train_end_date: str
    dependencies: Dict[str, str]
    artifact_sha256: str
    manifest_sha256: str
    created_at: str


def _collect_dependency_versions() -> Dict[str, str]:
    """Capture runtime Python and core library versions without importing optional ML libs."""
    versions: Dict[str, str] = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "pydantic"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def _sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file in streaming 64 KiB chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_manifest_digest(payload: Dict[str, Any]) -> str:
    """Compute canonical SHA-256 digest over manifest metadata excluding ``manifest_sha256``."""
    canonical = {k: v for k, v in payload.items() if k != "manifest_sha256"}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_identifier(value: str, label: str) -> None:
    """Reject empty strings, path separators, or traversal characters."""
    if not _SAFE_NAME_RE.match(value) or ".." in value:
        raise ArtifactIntegrityError(
            f"Invalid {label} {value!r}: must match ^[A-Za-z0-9][A-Za-z0-9._-]{{0,63}}$ "
            "without '..'."
        )


class ModelRegistry:
    """Local artifact store with SHA-256 integrity checks and reproducibility manifests."""

    def __init__(self, artifact_dir: Optional[Union[str, Path]] = None) -> None:
        base_dir = Path(artifact_dir if artifact_dir is not None else settings.MODEL_ARTIFACT_DIR)
        self.artifact_dir: Path = base_dir.resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_inside_root(self, filename: str) -> Path:
        """Ensure resolved path stays strictly inside ``self.artifact_dir``."""
        candidate = (self.artifact_dir / filename).resolve()
        if candidate.parent != self.artifact_dir and self.artifact_dir not in candidate.parents:
            raise ArtifactIntegrityError(
                f"Path traversal detected for artifact filename {filename!r}."
            )
        return candidate

    def save_artifact(
        self,
        artifact: object,
        *,
        model_id: str,
        version: str,
        stage: str,
        artifact_format: ArtifactFormat = "joblib",
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_seed: int = 42,
        data_snapshot_hash: str,
        train_start_date: str,
        train_end_date: str,
    ) -> ModelManifest:
        """Serialize a model artifact and write its companion reproducibility manifest."""
        _validate_identifier(model_id, "model_id")
        _validate_identifier(version, "version")
        if artifact_format not in _FORMAT_EXTENSIONS:
            raise ArtifactIntegrityError(
                f"Unsupported artifact_format {artifact_format!r}."
            )

        ext = _FORMAT_EXTENSIONS[artifact_format]
        artifact_filename = f"{model_id}__{version}{ext}"
        manifest_filename = f"{model_id}__{version}.manifest.json"

        artifact_path = self._resolve_inside_root(artifact_filename)
        manifest_path = self._resolve_inside_root(manifest_filename)

        if artifact_format == "joblib":
            joblib.dump(artifact, artifact_path)
        elif artifact_format == "json":
            if not isinstance(artifact, (dict, list)):
                raise ArtifactIntegrityError(
                    "JSON artifact_format requires a dict or list payload."
                )
            manifest_json = json.dumps(artifact, sort_keys=True, indent=2)
            artifact_path.write_text(manifest_json, encoding="utf-8")
        elif artifact_format == "state_dict_npz":
            if not isinstance(artifact, dict):
                raise ArtifactIntegrityError(
                    "state_dict_npz artifact_format requires a Dict[str, np.ndarray] mapping."
                )
            arrays: Dict[str, np.ndarray] = {}
            for key, val in artifact.items():
                arrays[str(key)] = np.asarray(val)
            np.savez(artifact_path, allow_pickle=False, **arrays)

        artifact_sha256 = _sha256_file(artifact_path)
        created_at = datetime.now(timezone.utc).isoformat()
        payload: Dict[str, Any] = {
            "model_id": model_id,
            "version": version,
            "stage": stage,
            "artifact_format": artifact_format,
            "artifact_filename": artifact_filename,
            "hyperparameters": hyperparameters or {},
            "random_seed": int(random_seed),
            "data_snapshot_hash": data_snapshot_hash,
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "dependencies": _collect_dependency_versions(),
            "artifact_sha256": artifact_sha256,
            "created_at": created_at,
        }
        payload["manifest_sha256"] = _compute_manifest_digest(payload)

        manifest_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8"
        )
        return ModelManifest(**payload)

    def load_manifest(self, model_id: str, version: str) -> ModelManifest:
        """Load and verify a model's reproducibility manifest."""
        _validate_identifier(model_id, "model_id")
        _validate_identifier(version, "version")
        manifest_path = self._resolve_inside_root(f"{model_id}__{version}.manifest.json")
        if not manifest_path.is_file():
            raise ArtifactIntegrityError(
                f"Manifest not found for {model_id} version {version}."
            )

        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ArtifactIntegrityError(
                f"Corrupted manifest JSON for {model_id} version {version}."
            ) from exc

        if not isinstance(raw, dict):
            raise ArtifactIntegrityError("Manifest payload must be a JSON object.")

        expected_digest = _compute_manifest_digest(raw)
        recorded_digest = str(raw.get("manifest_sha256", ""))
        if not hmac.compare_digest(expected_digest, recorded_digest):
            raise ArtifactIntegrityError(
                f"Manifest integrity check failed for {model_id} version {version}."
            )

        try:
            return ModelManifest(**raw)
        except TypeError as exc:
            raise ArtifactIntegrityError(
                f"Manifest schema mismatch for {model_id} version {version}."
            ) from exc

    def load_artifact(self, model_id: str, version: str) -> Tuple[object, ModelManifest]:
        """Verify manifest & SHA-256 digest before deserializing an artifact."""
        manifest = self.load_manifest(model_id, version)

        if manifest.artifact_format not in _FORMAT_EXTENSIONS:
            raise ArtifactIntegrityError(
                f"Untrusted or unsupported artifact format: {manifest.artifact_format!r}."
            )

        expected_ext = _FORMAT_EXTENSIONS[manifest.artifact_format]
        if not manifest.artifact_filename.endswith(expected_ext):
            raise ArtifactIntegrityError(
                f"Artifact filename {manifest.artifact_filename!r} does not match "
                f"expected extension {expected_ext!r}."
            )

        artifact_path = self._resolve_inside_root(manifest.artifact_filename)
        if not artifact_path.is_file():
            raise ArtifactIntegrityError(
                f"Artifact file missing: {manifest.artifact_filename!r}."
            )

        actual_sha256 = _sha256_file(artifact_path)
        if not hmac.compare_digest(actual_sha256, manifest.artifact_sha256):
            raise ArtifactIntegrityError(
                f"Artifact SHA-256 mismatch for {manifest.artifact_filename!r}; "
                "file may be corrupted or tampered with."
            )

        if manifest.artifact_format == "joblib":
            try:
                loaded = joblib.load(artifact_path)
            except Exception as exc:
                raise ArtifactIntegrityError(
                    f"Failed to deserialize joblib artifact {manifest.artifact_filename!r}."
                ) from exc
            return loaded, manifest

        if manifest.artifact_format == "json":
            try:
                loaded_json = json.loads(artifact_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ArtifactIntegrityError(
                    f"Failed to parse JSON artifact {manifest.artifact_filename!r}."
                ) from exc
            return loaded_json, manifest

        # state_dict_npz: load strictly with allow_pickle=False
        try:
            with np.load(artifact_path, allow_pickle=False) as npz_data:
                state_dict = {key: np.array(npz_data[key]) for key in npz_data.files}
        except Exception as exc:
            raise ArtifactIntegrityError(
                f"Failed to load NPZ state dict {manifest.artifact_filename!r}."
            ) from exc
        return state_dict, manifest
