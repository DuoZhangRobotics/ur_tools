from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from ur_tools.optitrack.config import load_config
from ur_tools.optitrack.dataset import CalibrationDataset, CalibrationSample
from ur_tools.optitrack.solver import solve_calibration, write_result
from ur_tools.optitrack.transforms import (
    invert,
    pose_error,
    transform_from_rotvec_translation,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "optitrack_ur5e_197.yaml")


def _synthetic_dataset(
    *, noise_translation_m: float = 0.0, noise_rotation_deg: float = 0.0
):
    rng = np.random.default_rng(7)
    base_from_mocap = transform_from_rotvec_translation(
        [0.25, -0.12, 0.31], [0.72, -0.36, 0.18]
    )
    tcp_from_target = transform_from_rotvec_translation(
        [-0.18, 0.09, 0.22], [0.03, -0.015, 0.12]
    )
    samples = []
    for index in range(25):
        base_from_tcp = transform_from_rotvec_translation(
            rng.uniform(-0.8, 0.8, 3),
            rng.uniform([-0.25, -0.75, 0.15], [0.25, -0.35, 0.55]),
        )
        mocap_from_target = invert(base_from_mocap) @ base_from_tcp @ tcp_from_target
        if noise_translation_m or noise_rotation_deg:
            noise = transform_from_rotvec_translation(
                rng.normal(0.0, np.deg2rad(noise_rotation_deg), 3),
                rng.normal(0.0, noise_translation_m, 3),
            )
            mocap_from_target = mocap_from_target @ noise
        samples.append(
            CalibrationSample(
                name=f"pose_{index + 1:02d}",
                captured_at_unix_s=float(index),
                base_from_tcp=base_from_tcp,
                mocap_from_target=mocap_from_target,
            )
        )
    return (
        CalibrationDataset(
            robot_ip="172.16.90.197",
            base_frame="base",
            mocap_frame="world",
            target_name="ur_calib",
            samples=samples,
            complete=True,
        ),
        base_from_mocap,
        tcp_from_target,
    )


def _assert_transform_close(actual, expected, translation_m, rotation_deg):
    translation, rotation = pose_error(expected, actual)
    assert np.linalg.norm(translation) < translation_m
    assert np.rad2deg(np.linalg.norm(rotation)) < rotation_deg


def test_exact_solver_recovers_transform_directions() -> None:
    dataset, expected_base_from_mocap, expected_tcp_from_target = _synthetic_dataset()
    result = solve_calibration(dataset, CONFIG.solver)

    assert result.accepted
    _assert_transform_close(
        result.base_from_mocap, expected_base_from_mocap, 1e-6, 1e-4
    )
    _assert_transform_close(
        result.tcp_from_target, expected_tcp_from_target, 1e-6, 1e-4
    )


def test_noisy_solver_meets_default_acceptance() -> None:
    dataset, expected_base_from_mocap, _ = _synthetic_dataset(
        noise_translation_m=0.0005, noise_rotation_deg=0.05
    )
    result = solve_calibration(dataset, CONFIG.solver)

    assert result.accepted, result.reasons
    _assert_transform_close(
        result.base_from_mocap, expected_base_from_mocap, 0.003, 0.5
    )


def test_validation_rejects_large_held_out_error() -> None:
    dataset, _, _ = _synthetic_dataset()
    # Every fifth sample is held out; corrupt one without influencing training.
    bad = dataset.samples[4]
    corrupted = bad.mocap_from_target.copy()
    corrupted[:3, 3] += [0.05, 0.0, 0.0]
    dataset.samples[4] = replace(bad, mocap_from_target=corrupted)

    result = solve_calibration(dataset, CONFIG.solver)

    assert not result.accepted
    assert any("translation" in reason for reason in result.reasons)


def test_robust_refinement_tolerates_one_training_outlier() -> None:
    dataset, expected_base_from_mocap, _ = _synthetic_dataset(
        noise_translation_m=0.0002, noise_rotation_deg=0.02
    )
    bad = dataset.samples[0]
    corrupted = bad.mocap_from_target.copy()
    corrupted[:3, 3] += [0.015, -0.01, 0.0]
    dataset.samples[0] = replace(bad, mocap_from_target=corrupted)

    result = solve_calibration(dataset, CONFIG.solver)

    assert result.accepted, result.reasons
    _assert_transform_close(
        result.base_from_mocap, expected_base_from_mocap, 0.004, 0.6
    )


def test_incomplete_dataset_can_never_be_accepted() -> None:
    dataset, _, _ = _synthetic_dataset()
    dataset.complete = False

    with pytest.raises(ValueError, match="incomplete"):
        solve_calibration(dataset, CONFIG.solver)


def test_accepted_result_writes_matrices_and_metrics(tmp_path: Path) -> None:
    dataset, _, _ = _synthetic_dataset()
    result = solve_calibration(dataset, CONFIG.solver)
    output = write_result(tmp_path, result, dataset)
    payload = yaml.safe_load(output.read_text())

    assert output.name == "calibration.yaml"
    assert payload["accepted"]
    assert "static_tf_command" not in payload
    for name in ("base_from_mocap.txt", "mocap_from_base.txt", "tcp_from_target.txt"):
        assert (tmp_path / name).is_file()


def test_degenerate_rotations_are_rejected() -> None:
    dataset, _, _ = _synthetic_dataset()
    for index, sample in enumerate(dataset.samples):
        transform = sample.base_from_tcp.copy()
        transform[:3, :3] = np.eye(3)
        dataset.samples[index] = replace(sample, base_from_tcp=transform)

    with pytest.raises(ValueError, match="rotations"):
        solve_calibration(dataset, CONFIG.solver)
