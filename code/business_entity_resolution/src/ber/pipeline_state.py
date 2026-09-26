"""Durable stage and shard checkpoints for long-running pipeline jobs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_VERSION = 1


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def signature(value: Any) -> str:
    """Return a stable signature for configuration relevant to a checkpoint."""

    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Replace a JSON file atomically so an interruption cannot corrupt it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(value), handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class PipelineState:
    """Small manifest recording completed stages and individual shards."""

    def __init__(self, work_dir: Path) -> None:
        self.path = work_dir / "pipeline_state.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("state_version") != STATE_VERSION:
                raise RuntimeError(
                    f"unsupported checkpoint version in {self.path}; "
                    "use a new --work-dir"
                )
        else:
            self.data = {
                "state_version": STATE_VERSION,
                "created_at": self._now(),
                "updated_at": self._now(),
                "stages": {},
            }
            self.save()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def save(self) -> None:
        self.data["updated_at"] = self._now()
        atomic_write_json(self.path, self.data)

    def stage_complete(
        self, stage: str, stage_signature: str, outputs: tuple[Path, ...]
    ) -> bool:
        record = self.data["stages"].get(stage)
        return bool(
            record
            and record.get("status") == "complete"
            and record.get("signature") == stage_signature
            and all(path.exists() for path in outputs)
        )

    def begin_stage(self, stage: str, stage_signature: str) -> None:
        previous = self.data["stages"].get(stage, {})
        self.data["stages"][stage] = {
            "status": "running",
            "signature": stage_signature,
            "started_at": self._now(),
            "completed_at": None,
            "shards": previous.get("shards", {}),
        }
        self.save()

    def complete_stage(self, stage: str, outputs: tuple[Path, ...]) -> None:
        missing = [str(path) for path in outputs if not path.exists()]
        if missing:
            raise RuntimeError(
                f"stage {stage!r} cannot complete; missing outputs: {missing}"
            )
        record = self.data["stages"][stage]
        record["status"] = "complete"
        record["completed_at"] = self._now()
        record["outputs"] = [str(path) for path in outputs]
        self.save()

    def shard_complete(
        self,
        stage: str,
        shard: str,
        shard_signature: str,
        outputs: tuple[Path, ...],
    ) -> bool:
        record = (
            self.data["stages"].get(stage, {}).get("shards", {}).get(shard)
        )
        return bool(
            record
            and record.get("status") == "complete"
            and record.get("signature") == shard_signature
            and all(path.exists() for path in outputs)
        )

    def complete_shard(
        self,
        stage: str,
        shard: str,
        shard_signature: str,
        outputs: tuple[Path, ...],
        statistics: dict[str, Any] | None = None,
    ) -> None:
        missing = [str(path) for path in outputs if not path.exists()]
        if missing:
            raise RuntimeError(
                f"shard {stage}/{shard} cannot complete; missing outputs: {missing}"
            )
        stage_record = self.data["stages"].setdefault(stage, {"shards": {}})
        stage_record.setdefault("shards", {})[shard] = {
            "status": "complete",
            "signature": shard_signature,
            "completed_at": self._now(),
            "outputs": [str(path) for path in outputs],
            "statistics": _jsonable(statistics or {}),
        }
        self.save()
