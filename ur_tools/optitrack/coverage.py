"""Offline coverage-map construction for a hand-swept OptiTrack wand."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class CoverageSample:
    frame_number: int
    received_monotonic_s: float
    source_time_s: float
    tracking_valid: bool
    position_world: Point3 | None
    position_base: Point3 | None
    observed_marker_count: int
    expected_marker_count: int
    rigid_body_error: float | None = None
    marker_residual_mean: float | None = None
    marker_residual_max: float | None = None
    interpolated_position: bool = False
    localized_gap_s: float = 0.0


@dataclass(frozen=True)
class VoxelCoverage:
    index: tuple[int, int, int]
    center_base: Point3
    sample_count: int
    valid_count: int
    interpolated_invalid_count: int
    detection_fraction: float
    complete_marker_fraction: float
    mean_marker_fraction: float
    mean_rigid_body_error: float | None
    maximum_rigid_body_error: float | None
    mean_marker_residual: float | None
    maximum_marker_residual: float | None
    maximum_localized_gap_s: float
    grade: str


def _point(value: Sequence[float], name: str) -> Point3:
    point = tuple(float(component) for component in value)
    if len(point) != 3 or not all(math.isfinite(component) for component in point):
        raise ValueError(f"{name} must contain three finite values")
    return point  # type: ignore[return-value]


def interpolate_short_gaps(
    samples: Sequence[CoverageSample],
    maximum_gap_s: float,
) -> list[CoverageSample]:
    """Localize bounded invalid runs by interpolation between valid poses."""
    if not math.isfinite(maximum_gap_s) or maximum_gap_s < 0:
        raise ValueError("maximum_gap_s must be finite and non-negative")
    result = list(samples)
    index = 0
    while index < len(result):
        if result[index].position_base is not None:
            index += 1
            continue
        start = index
        while index < len(result) and result[index].position_base is None:
            index += 1
        end = index
        if start == 0 or end >= len(result):
            continue
        before = result[start - 1]
        after = result[end]
        if before.position_base is None or after.position_base is None:
            continue
        duration = after.received_monotonic_s - before.received_monotonic_s
        if duration <= 0 or duration > maximum_gap_s:
            continue
        for missing_index in range(start, end):
            sample = result[missing_index]
            fraction = (
                sample.received_monotonic_s - before.received_monotonic_s
            ) / duration
            base = tuple(
                first + fraction * (second - first)
                for first, second in zip(before.position_base, after.position_base)
            )
            world = None
            if before.position_world is not None and after.position_world is not None:
                world = tuple(
                    first + fraction * (second - first)
                    for first, second in zip(
                        before.position_world, after.position_world
                    )
                )
            result[missing_index] = replace(
                sample,
                position_base=base,  # type: ignore[arg-type]
                position_world=world,  # type: ignore[arg-type]
                interpolated_position=True,
                localized_gap_s=duration,
            )
    return result


def _optional_mean(values: Iterable[float | None]) -> float | None:
    finite = [
        float(value) for value in values if value is not None and math.isfinite(value)
    ]
    return None if not finite else sum(finite) / len(finite)


def _optional_max(values: Iterable[float | None]) -> float | None:
    finite = [
        float(value) for value in values if value is not None and math.isfinite(value)
    ]
    return None if not finite else max(finite)


def voxelize_samples(
    samples: Sequence[CoverageSample],
    *,
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    voxel_size_m: float,
    minimum_samples: int = 5,
    reliable_detection_fraction: float = 0.99,
    marginal_detection_fraction: float = 0.95,
    reliable_marker_fraction: float = 0.95,
    marginal_marker_fraction: float = 0.67,
) -> list[VoxelCoverage]:
    """Aggregate localized samples into conservative UR-base voxels."""
    lower = _point(bounds_min, "bounds_min")
    upper = _point(bounds_max, "bounds_max")
    if any(high <= low for low, high in zip(lower, upper)):
        raise ValueError("bounds_max must be greater than bounds_min")
    if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive and finite")
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    for value, name in (
        (reliable_detection_fraction, "reliable_detection_fraction"),
        (marginal_detection_fraction, "marginal_detection_fraction"),
        (reliable_marker_fraction, "reliable_marker_fraction"),
        (marginal_marker_fraction, "marginal_marker_fraction"),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be from zero to one")

    grouped: dict[tuple[int, int, int], list[CoverageSample]] = defaultdict(list)
    for sample in samples:
        point = sample.position_base
        if point is None or any(
            value < low or value >= high
            for value, low, high in zip(point, lower, upper)
        ):
            continue
        index = tuple(
            math.floor((value - low) / voxel_size_m) for value, low in zip(point, lower)
        )
        grouped[index].append(sample)  # type: ignore[arg-type]

    voxels = []
    for index, values in sorted(grouped.items()):
        sample_count = len(values)
        valid_count = sum(sample.tracking_valid for sample in values)
        marker_fractions = [
            min(
                1.0,
                sample.observed_marker_count / max(1, sample.expected_marker_count),
            )
            for sample in values
        ]
        complete_count = sum(
            sample.expected_marker_count > 0
            and sample.observed_marker_count >= sample.expected_marker_count
            for sample in values
        )
        detection = valid_count / sample_count
        complete = complete_count / sample_count
        mean_marker = sum(marker_fractions) / sample_count
        if sample_count < minimum_samples:
            grade = "insufficient"
        elif (
            detection >= reliable_detection_fraction
            and complete >= reliable_marker_fraction
        ):
            grade = "reliable"
        elif (
            detection >= marginal_detection_fraction
            and complete >= marginal_marker_fraction
        ):
            grade = "marginal"
        else:
            grade = "blind"
        center = tuple(
            low + (voxel_index + 0.5) * voxel_size_m
            for low, voxel_index in zip(lower, index)
        )
        voxels.append(
            VoxelCoverage(
                index=index,
                center_base=center,  # type: ignore[arg-type]
                sample_count=sample_count,
                valid_count=valid_count,
                interpolated_invalid_count=sum(
                    sample.interpolated_position and not sample.tracking_valid
                    for sample in values
                ),
                detection_fraction=detection,
                complete_marker_fraction=complete,
                mean_marker_fraction=mean_marker,
                mean_rigid_body_error=_optional_mean(
                    sample.rigid_body_error for sample in values
                ),
                maximum_rigid_body_error=_optional_max(
                    sample.rigid_body_error for sample in values
                ),
                mean_marker_residual=_optional_mean(
                    sample.marker_residual_mean for sample in values
                ),
                maximum_marker_residual=_optional_max(
                    sample.marker_residual_max for sample in values
                ),
                maximum_localized_gap_s=max(
                    sample.localized_gap_s for sample in values
                ),
                grade=grade,
            )
        )
    return voxels


def coverage_summary(
    samples: Sequence[CoverageSample],
    voxels: Sequence[VoxelCoverage],
    voxel_size_m: float,
) -> dict[str, object]:
    localized = [sample for sample in samples if sample.position_base is not None]
    valid = sum(sample.tracking_valid for sample in samples)
    grade_counts = {
        grade: sum(voxel.grade == grade for voxel in voxels)
        for grade in ("reliable", "marginal", "blind", "insufficient")
    }
    points = [sample.position_base for sample in localized]
    return {
        "frame_count": len(samples),
        "valid_frame_count": valid,
        "overall_detection_fraction": (valid / len(samples) if samples else None),
        "localized_frame_count": len(localized),
        "unlocalized_frame_count": len(samples) - len(localized),
        "voxel_count": len(voxels),
        "voxel_grade_counts": grade_counts,
        "voxel_volume_m3": voxel_size_m**3,
        "estimated_reliable_volume_m3": (grade_counts["reliable"] * voxel_size_m**3),
        "estimated_marginal_volume_m3": (grade_counts["marginal"] * voxel_size_m**3),
        "observed_bounds_min_base_m": (
            None
            if not points
            else [min(point[axis] for point in points) for axis in range(3)]
        ),
        "observed_bounds_max_base_m": (
            None
            if not points
            else [max(point[axis] for point in points) for axis in range(3)]
        ),
    }


def write_samples_csv(path: Path, samples: Sequence[CoverageSample]) -> None:
    fields = [
        "frame_number",
        "received_monotonic_s",
        "source_time_s",
        "tracking_valid",
        "world_x",
        "world_y",
        "world_z",
        "base_x",
        "base_y",
        "base_z",
        "observed_marker_count",
        "expected_marker_count",
        "rigid_body_error",
        "marker_residual_mean",
        "marker_residual_max",
        "interpolated_position",
        "localized_gap_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            world = sample.position_world or (None, None, None)
            base = sample.position_base or (None, None, None)
            writer.writerow(
                {
                    "frame_number": sample.frame_number,
                    "received_monotonic_s": sample.received_monotonic_s,
                    "source_time_s": sample.source_time_s,
                    "tracking_valid": sample.tracking_valid,
                    "world_x": world[0],
                    "world_y": world[1],
                    "world_z": world[2],
                    "base_x": base[0],
                    "base_y": base[1],
                    "base_z": base[2],
                    "observed_marker_count": sample.observed_marker_count,
                    "expected_marker_count": sample.expected_marker_count,
                    "rigid_body_error": sample.rigid_body_error,
                    "marker_residual_mean": sample.marker_residual_mean,
                    "marker_residual_max": sample.marker_residual_max,
                    "interpolated_position": sample.interpolated_position,
                    "localized_gap_s": sample.localized_gap_s,
                }
            )


def write_voxels(path: Path, voxels: Sequence[VoxelCoverage]) -> None:
    records = [asdict(voxel) for voxel in voxels]
    path.with_suffix(".json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not records:
        path.with_suffix(".csv").write_text("", encoding="utf-8")
        return
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def write_voxel_ply(path: Path, voxels: Sequence[VoxelCoverage]) -> None:
    colors = {
        "reliable": (0, 200, 0),
        "marginal": (255, 200, 0),
        "blind": (220, 0, 0),
        "insufficient": (128, 128, 128),
    }
    with path.open("w", encoding="utf-8") as output:
        output.write("ply\nformat ascii 1.0\n")
        output.write(f"element vertex {len(voxels)}\n")
        output.write("property float x\nproperty float y\nproperty float z\n")
        output.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        output.write("property float detection_fraction\nend_header\n")
        for voxel in voxels:
            red, green, blue = colors[voxel.grade]
            x, y, z = voxel.center_base
            output.write(
                f"{x:.6f} {y:.6f} {z:.6f} {red} {green} {blue} "
                f"{voxel.detection_fraction:.6f}\n"
            )


def render_projection(
    path: Path,
    voxels: Sequence[VoxelCoverage],
    axes: tuple[int, int],
    axis_labels: tuple[str, str],
    scale: int = 12,
) -> None:
    """Render the worst sampled detection fraction along the omitted axis."""
    if not voxels:
        return
    indices = [voxel.index for voxel in voxels]
    minimum = [min(index[axis] for index in indices) for axis in axes]
    maximum = [max(index[axis] for index in indices) for axis in axes]
    width = maximum[0] - minimum[0] + 1
    height = maximum[1] - minimum[1] + 1
    grid = np.full((height, width), np.nan, dtype=np.float32)
    for voxel in voxels:
        x = voxel.index[axes[0]] - minimum[0]
        y = maximum[1] - voxel.index[axes[1]]
        current = grid[y, x]
        grid[y, x] = (
            voxel.detection_fraction
            if np.isnan(current)
            else min(current, voxel.detection_fraction)
        )
    normalized = np.nan_to_num(grid, nan=0.0)
    colored = cv2.applyColorMap(
        np.asarray(np.clip(normalized, 0.0, 1.0) * 255, dtype=np.uint8),
        cv2.COLORMAP_TURBO,
    )
    colored[np.isnan(grid)] = (96, 96, 96)
    colored = cv2.resize(
        colored, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST
    )
    cv2.putText(
        colored,
        f"{axis_labels[0]} horizontal / {axis_labels[1]} vertical; gray=unsampled",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), colored)
