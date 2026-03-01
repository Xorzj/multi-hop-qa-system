from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from src.common.exceptions import NotFoundError, ValidationError
from src.common.logger import get_logger

_LOGGER = get_logger(__name__)
_SEMVER_PATTERN = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
_ADAPTER_TYPES: set[Literal["dapt", "sft"]] = {"dapt", "sft"}


@dataclass
class CheckpointMetadata:
    """Metadata for a single adapter checkpoint."""

    version: str
    created_at: datetime
    base_model: str
    adapter_type: Literal["dapt", "sft"]
    training_steps: int
    eval_loss: float | None = None
    config: dict[str, Any] = field(default_factory=dict)
    description: str = ""


class CheckpointManager:
    """Manage LoRA adapter checkpoints with versioning and metadata."""

    def __init__(self, base_dir: Path | str, max_checkpoints: int = 5) -> None:
        """Initialize the checkpoint manager.

        Args:
            base_dir: Base directory that stores adapter checkpoints.
            max_checkpoints: Maximum checkpoints to keep per adapter type.
        """

        self._base_dir = Path(base_dir)
        self._max_checkpoints = max_checkpoints

    def save_checkpoint(
        self, adapter_path: Path | str, metadata: CheckpointMetadata
    ) -> Path:
        """Save an adapter checkpoint to a versioned directory.

        Args:
            adapter_path: Path to the adapter directory to copy.
            metadata: Metadata describing the checkpoint.

        Returns:
            Path to the saved checkpoint directory.
        """

        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            raise NotFoundError(f"Adapter path not found: {adapter_dir}")
        if not adapter_dir.is_dir():
            raise ValidationError(f"Adapter path is not a directory: {adapter_dir}")

        if metadata.adapter_type not in _ADAPTER_TYPES:
            raise ValidationError(f"Unsupported adapter type: {metadata.adapter_type}")

        if not metadata.version.strip():
            metadata.version = self._generate_version()

        checkpoint_dir = self._base_dir / metadata.adapter_type / metadata.version
        if checkpoint_dir.exists():
            raise ValidationError(f"Checkpoint already exists: {checkpoint_dir}")

        checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(adapter_dir, checkpoint_dir)
        self._save_metadata(checkpoint_dir, metadata)

        _LOGGER.info(
            "Saved adapter checkpoint",
            extra={
                "version": metadata.version,
                "adapter_type": metadata.adapter_type,
                "path": str(checkpoint_dir),
            },
        )

        if self._max_checkpoints > 0:
            self.cleanup_old()

        return checkpoint_dir

    def load_checkpoint(self, version: str) -> tuple[Path, CheckpointMetadata]:
        """Load a checkpoint by version.

        Args:
            version: Version identifier to load.

        Returns:
            Tuple of checkpoint directory path and metadata.
        """

        for adapter_type in _ADAPTER_TYPES:
            checkpoint_dir = self._base_dir / adapter_type / version
            if checkpoint_dir.exists() and checkpoint_dir.is_dir():
                metadata = self._load_metadata(checkpoint_dir)
                return checkpoint_dir, metadata

        raise NotFoundError(f"Checkpoint version not found: {version}")

    def list_checkpoints(self) -> list[CheckpointMetadata]:
        """List all checkpoint metadata sorted by version."""

        checkpoints = []
        for adapter_type in _ADAPTER_TYPES:
            adapter_dir = self._base_dir / adapter_type
            if not adapter_dir.exists():
                continue
            for checkpoint_dir in adapter_dir.iterdir():
                if not checkpoint_dir.is_dir():
                    continue
                try:
                    metadata = self._load_metadata(checkpoint_dir)
                except (NotFoundError, ValidationError) as exc:
                    _LOGGER.warning(
                        "Skipping checkpoint with invalid metadata",
                        extra={"path": str(checkpoint_dir), "error": str(exc)},
                    )
                    continue
                checkpoints.append(metadata)

        checkpoints.sort(key=lambda item: self._version_key(item.version))
        return checkpoints

    def get_latest(
        self, adapter_type: Literal["dapt", "sft"] | None = None
    ) -> tuple[Path, CheckpointMetadata] | None:
        """Get the most recent checkpoint, optionally filtered by adapter type.

        Args:
            adapter_type: Optional adapter type filter.

        Returns:
            Tuple of checkpoint path and metadata, or None if none exist.
        """

        candidates: list[tuple[Path, CheckpointMetadata]] = []
        types_to_search = (
            {adapter_type} if adapter_type is not None else set(_ADAPTER_TYPES)
        )

        for current_type in types_to_search:
            if current_type not in _ADAPTER_TYPES:
                raise ValidationError(f"Unsupported adapter type: {current_type}")
            adapter_dir = self._base_dir / current_type
            if not adapter_dir.exists():
                continue
            for checkpoint_dir in adapter_dir.iterdir():
                if not checkpoint_dir.is_dir():
                    continue
                try:
                    metadata = self._load_metadata(checkpoint_dir)
                except (NotFoundError, ValidationError):
                    continue
                candidates.append((checkpoint_dir, metadata))

        if not candidates:
            return None

        candidates.sort(key=lambda item: self._version_key(item[1].version))
        return candidates[-1]

    def delete_checkpoint(self, version: str) -> bool:
        """Delete a checkpoint by version.

        Args:
            version: Version to delete.

        Returns:
            True if deleted, False if not found.
        """

        for adapter_type in _ADAPTER_TYPES:
            checkpoint_dir = self._base_dir / adapter_type / version
            if checkpoint_dir.exists() and checkpoint_dir.is_dir():
                shutil.rmtree(checkpoint_dir)
                _LOGGER.info(
                    "Deleted adapter checkpoint",
                    extra={"version": version, "path": str(checkpoint_dir)},
                )
                return True
        return False

    def cleanup_old(self, keep: int | None = None) -> int:
        """Delete old checkpoints, keeping the most recent ones.

        Args:
            keep: Number of checkpoints to keep per adapter type.

        Returns:
            Count of deleted checkpoints.
        """

        keep_count = self._max_checkpoints if keep is None else keep
        if keep_count < 0:
            raise ValidationError("keep must be zero or positive")

        deleted = 0
        for adapter_type in _ADAPTER_TYPES:
            adapter_dir = self._base_dir / adapter_type
            if not adapter_dir.exists():
                continue
            checkpoints = []
            for checkpoint_dir in adapter_dir.iterdir():
                if not checkpoint_dir.is_dir():
                    continue
                try:
                    metadata = self._load_metadata(checkpoint_dir)
                except (NotFoundError, ValidationError):
                    continue
                checkpoints.append((checkpoint_dir, metadata))

            checkpoints.sort(key=lambda item: self._version_key(item[1].version))
            if keep_count == 0:
                to_delete = checkpoints
            else:
                to_delete = (
                    checkpoints[:-keep_count] if len(checkpoints) > keep_count else []
                )

            for checkpoint_dir, metadata in to_delete:
                shutil.rmtree(checkpoint_dir)
                deleted += 1
                _LOGGER.info(
                    "Cleaned up checkpoint",
                    extra={
                        "version": metadata.version,
                        "adapter_type": metadata.adapter_type,
                        "path": str(checkpoint_dir),
                    },
                )

        return deleted

    def _generate_version(self) -> str:
        """Generate the next semantic version string."""

        versions: list[tuple[int, int, int]] = []
        if self._base_dir.exists():
            for adapter_type in _ADAPTER_TYPES:
                adapter_dir = self._base_dir / adapter_type
                if not adapter_dir.exists():
                    continue
                for checkpoint_dir in adapter_dir.iterdir():
                    if not checkpoint_dir.is_dir():
                        continue
                    match = _SEMVER_PATTERN.match(checkpoint_dir.name)
                    if match:
                        major = int(match.group(1))
                        minor = int(match.group(2))
                        patch = int(match.group(3))
                        versions.append((major, minor, patch))

        if not versions:
            return "v1.0.0"

        major, minor, patch = max(versions)
        return f"v{major}.{minor}.{patch + 1}"

    def _save_metadata(
        self, checkpoint_dir: Path, metadata: CheckpointMetadata
    ) -> None:
        """Save metadata for a checkpoint to disk."""

        data = {
            "version": metadata.version,
            "created_at": metadata.created_at.isoformat(),
            "base_model": metadata.base_model,
            "adapter_type": metadata.adapter_type,
            "training_steps": metadata.training_steps,
            "eval_loss": metadata.eval_loss,
            "config": metadata.config,
            "description": metadata.description,
        }
        metadata_path = checkpoint_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _load_metadata(self, checkpoint_dir: Path) -> CheckpointMetadata:
        """Load metadata for a checkpoint from disk."""

        metadata_path = checkpoint_dir / "metadata.json"
        if not metadata_path.exists():
            raise NotFoundError(f"Missing metadata.json in {checkpoint_dir}")

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        required_fields = {
            "version",
            "created_at",
            "base_model",
            "adapter_type",
            "training_steps",
        }
        missing = required_fields - payload.keys()
        if missing:
            raise ValidationError(
                f"Checkpoint metadata missing fields: {sorted(missing)}"
            )

        adapter_type = payload["adapter_type"]
        if adapter_type not in _ADAPTER_TYPES:
            raise ValidationError(f"Unsupported adapter type: {adapter_type}")

        try:
            created_at = datetime.fromisoformat(payload["created_at"])
        except (TypeError, ValueError) as exc:
            raise ValidationError("Invalid created_at format in metadata") from exc

        return CheckpointMetadata(
            version=str(payload["version"]),
            created_at=created_at,
            base_model=str(payload["base_model"]),
            adapter_type=adapter_type,
            training_steps=int(payload["training_steps"]),
            eval_loss=payload.get("eval_loss"),
            config=payload.get("config", {}),
            description=str(payload.get("description", "")),
        )

    def _version_key(self, version: str) -> tuple[int, int, int, int | str]:
        match = _SEMVER_PATTERN.match(version)
        if match:
            major, minor, patch = (int(part) for part in match.groups())
            return (0, major, minor, patch)
        return (1, 0, 0, version)
