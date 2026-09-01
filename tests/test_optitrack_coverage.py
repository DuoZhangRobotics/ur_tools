import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ur_tools.optitrack.coverage import (
    CoverageSample,
    coverage_summary,
    interpolate_short_gaps,
    render_projection,
    voxelize_samples,
    write_samples_csv,
    write_voxel_ply,
    write_voxels,
)
from ur_tools.optitrack.coverage_cli import NatNetCoverageCollector, main

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "config" / "optitrack_to_ur_base.yaml"


def sample(
    timestamp: float,
    *,
    valid: bool,
    position=(0.1, 0.1, 0.1),
    marker_count: int = 3,
) -> CoverageSample:
    point = position if valid else None
    return CoverageSample(
        frame_number=int(timestamp * 100),
        received_monotonic_s=timestamp,
        source_time_s=timestamp,
        tracking_valid=valid,
        position_world=point,
        position_base=point,
        observed_marker_count=marker_count,
        expected_marker_count=3,
        rigid_body_error=0.001 if valid else None,
        marker_residual_mean=0.002 if valid else None,
        marker_residual_max=0.003 if valid else None,
    )


def test_short_invalid_gap_is_spatially_localized() -> None:
    samples = [
        sample(0.0, valid=True, position=(0.0, 0.0, 0.0)),
        sample(0.1, valid=False),
        sample(0.2, valid=True, position=(0.2, 0.4, 0.6)),
    ]

    localized = interpolate_short_gaps(samples, maximum_gap_s=0.25)

    assert localized[1].position_base == pytest.approx((0.1, 0.2, 0.3))
    assert localized[1].position_world == pytest.approx((0.1, 0.2, 0.3))
    assert localized[1].interpolated_position
    assert not localized[1].tracking_valid
    assert localized[1].localized_gap_s == pytest.approx(0.2)


def test_unbounded_or_long_gap_remains_unlocalized() -> None:
    unbounded = interpolate_short_gaps(
        [sample(0.0, valid=False), sample(0.1, valid=True)], 0.25
    )
    long_gap = interpolate_short_gaps(
        [sample(0.0, valid=True), sample(0.5, valid=False), sample(1.0, valid=True)],
        0.25,
    )

    assert unbounded[0].position_base is None
    assert long_gap[1].position_base is None


def test_voxel_grades_include_marker_completeness_and_dropouts() -> None:
    reliable = [sample(index / 100, valid=True) for index in range(10)]
    blind = [
        sample(1.0 + index / 100, valid=index < 8, position=(0.6, 0.1, 0.1))
        for index in range(10)
    ]
    blind = interpolate_short_gaps(
        [
            *blind,
            sample(1.11, valid=True, position=(0.6, 0.1, 0.1)),
        ],
        maximum_gap_s=0.25,
    )

    voxels = voxelize_samples(
        [*reliable, *blind],
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(1.0, 1.0, 1.0),
        voxel_size_m=0.5,
        minimum_samples=5,
    )

    by_index = {voxel.index: voxel for voxel in voxels}
    assert by_index[(0, 0, 0)].grade == "reliable"
    assert by_index[(1, 0, 0)].grade == "blind"
    assert by_index[(1, 0, 0)].detection_fraction < 0.95
    assert by_index[(1, 0, 0)].interpolated_invalid_count == 2


def test_exports_write_machine_and_visual_artifacts(tmp_path: Path) -> None:
    samples = [sample(index / 100, valid=True) for index in range(10)]
    voxels = voxelize_samples(
        samples,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(1.0, 1.0, 1.0),
        voxel_size_m=0.5,
    )

    write_samples_csv(tmp_path / "samples.csv", samples)
    write_voxels(tmp_path / "voxels", voxels)
    write_voxel_ply(tmp_path / "coverage.ply", voxels)
    render_projection(tmp_path / "xy.png", voxels, (0, 1), ("X", "Y"))
    summary = coverage_summary(samples, voxels, 0.5)

    assert (tmp_path / "samples.csv").stat().st_size > 0
    assert json.loads((tmp_path / "voxels.json").read_text())
    assert (tmp_path / "voxels.csv").stat().st_size > 0
    assert (tmp_path / "coverage.ply").read_text().startswith("ply\n")
    assert (tmp_path / "xy.png").stat().st_size > 0
    assert summary["estimated_reliable_volume_m3"] == pytest.approx(0.125)


def test_coverage_cli_is_hardware_free_by_default(capsys) -> None:
    sys.modules.pop("natnet", None)

    result = main(["--calibration", str(CALIBRATION)])

    assert result == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert "natnet" not in sys.modules


def test_marker_observation_excludes_occluded_or_model_only_markers() -> None:
    observed = SimpleNamespace(
        param=2,
        occluded=False,
        point_cloud_solved=True,
    )
    occluded = SimpleNamespace(
        param=3,
        occluded=True,
        point_cloud_solved=True,
    )
    model_only = SimpleNamespace(
        param=4,
        occluded=False,
        point_cloud_solved=False,
    )

    assert NatNetCoverageCollector._marker_is_observed(observed)
    assert not NatNetCoverageCollector._marker_is_observed(occluded)
    assert not NatNetCoverageCollector._marker_is_observed(model_only)
