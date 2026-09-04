"""Persistent, atomic calibration datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .transforms import validate_transform


@dataclass(frozen=True)
class CalibrationSample:
    name: str
    captured_at_unix_s: float
    base_from_tcp: np.ndarray
    mocap_from_target: np.ndarray
    message_count: int = 0
    translation_std_m: float = 0.0
    rotation_std_deg: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("sample name must not be empty")
        if not np.isfinite(self.captured_at_unix_s):
            raise ValueError("sample timestamp must be finite")
        object.__setattr__(
            self,
            "base_from_tcp",
            validate_transform(self.base_from_tcp, "base_from_tcp"),
        )
        object.__setattr__(
            self,
            "mocap_from_target",
            validate_transform(self.mocap_from_target, "mocap_from_target"),
        )


@dataclass
class CalibrationDataset:
    robot_ip: str
    base_frame: str
    mocap_frame: str
    target_name: str
    samples: list[CalibrationSample] = field(default_factory=list)
    complete: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def _sample_to_dict(sample: CalibrationSample) -> dict[str, Any]:
    return {
        "name": sample.name,
        "captured_at_unix_s": float(sample.captured_at_unix_s),
        "base_from_tcp": sample.base_from_tcp.tolist(),
        "mocap_from_target": sample.mocap_from_target.tolist(),
        "message_count": int(sample.message_count),
        "translation_std_m": float(sample.translation_std_m),
        "rotation_std_deg": float(sample.rotation_std_deg),
    }


def _dataset_to_dict(dataset: CalibrationDataset) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": dataset.created_at,
        "complete": bool(dataset.complete),
        "robot_ip": dataset.robot_ip,
        "frames": {
            "base": dataset.base_frame,
            "mocap": dataset.mocap_frame,
            "target": dataset.target_name,
        },
        "samples": [_sample_to_dict(sample) for sample in dataset.samples],
    }


def save_dataset(path: str | Path, dataset: CalibrationDataset) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(_dataset_to_dict(dataset), sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_dataset(path: str | Path) -> CalibrationDataset:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("calibration dataset must use schema_version 1")
    frames = raw.get("frames")
    if not isinstance(frames, dict):
        raise TypeError("calibration dataset must define frames")
    raw_samples = raw.get("samples")
    if not isinstance(raw_samples, list):
        raise TypeError("calibration dataset samples must be a list")
    samples = [
        CalibrationSample(
            name=str(item["name"]),
            captured_at_unix_s=float(item["captured_at_unix_s"]),
            base_from_tcp=np.asarray(item["base_from_tcp"], dtype=float),
            mocap_from_target=np.asarray(item["mocap_from_target"], dtype=float),
            message_count=int(item.get("message_count", 0)),
            translation_std_m=float(item.get("translation_std_m", 0.0)),
            rotation_std_deg=float(item.get("rotation_std_deg", 0.0)),
        )
        for item in raw_samples
    ]
    return CalibrationDataset(
        robot_ip=str(raw["robot_ip"]),
        base_frame=str(frames["base"]),
        mocap_frame=str(frames["mocap"]),
        target_name=str(frames["target"]),
        samples=samples,
        complete=bool(raw.get("complete", False)),
        created_at=str(raw.get("created_at", "")),
    )
