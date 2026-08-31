"""Robot-world/hand-eye initialization, refinement, and validation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .config import SolverConfig
from .dataset import CalibrationDataset, CalibrationSample
from .transforms import (
    invert,
    params_from_transform,
    pose_error,
    transform_from_params,
)


@dataclass(frozen=True)
class ErrorMetrics:
    sample_count: int
    translation_rms_m: float
    translation_max_m: float
    rotation_rms_deg: float
    rotation_max_deg: float


@dataclass(frozen=True)
class CalibrationResult:
    accepted: bool
    initializer: str
    base_from_mocap: np.ndarray
    tcp_from_target: np.ndarray
    training_metrics: ErrorMetrics
    validation_metrics: ErrorMetrics
    all_sample_metrics: ErrorMetrics
    reasons: tuple[str, ...]


def _check_excitation(samples: Sequence[CalibrationSample]) -> None:
    if len(samples) < 6:
        raise ValueError("at least six calibration samples are required")
    positions = np.array([sample.base_from_tcp[:3, 3] for sample in samples])
    if np.linalg.matrix_rank(positions - positions.mean(axis=0), tol=1e-4) < 2:
        raise ValueError("TCP positions are degenerate; use non-collinear positions")
    reference = samples[0].base_from_tcp[:3, :3]
    relative_rotvecs = np.array(
        [
            Rotation.from_matrix(reference.T @ sample.base_from_tcp[:3, :3]).as_rotvec()
            for sample in samples[1:]
        ]
    )
    if not len(relative_rotvecs) or np.max(
        np.linalg.norm(relative_rotvecs, axis=1)
    ) < np.deg2rad(10):
        raise ValueError("TCP rotations lack at least 10 degrees of excitation")
    if np.linalg.matrix_rank(relative_rotvecs, tol=np.deg2rad(1)) < 2:
        raise ValueError("TCP rotations must excite at least two rotation axes")


def _opencv_initializer(
    samples: Sequence[CalibrationSample], method: int
) -> tuple[np.ndarray, np.ndarray]:
    target_from_mocap = [invert(sample.mocap_from_target) for sample in samples]
    tcp_from_base = [invert(sample.base_from_tcp) for sample in samples]
    result = cv2.calibrateRobotWorldHandEye(
        [pose[:3, :3] for pose in target_from_mocap],
        [pose[:3, 3].reshape(3, 1) for pose in target_from_mocap],
        [pose[:3, :3] for pose in tcp_from_base],
        [pose[:3, 3].reshape(3, 1) for pose in tcp_from_base],
        method=method,
    )
    mocap_from_base = np.eye(4)
    mocap_from_base[:3, :3] = np.asarray(result[0], dtype=float)
    mocap_from_base[:3, 3] = np.asarray(result[1], dtype=float).reshape(3)
    target_from_tcp = np.eye(4)
    target_from_tcp[:3, :3] = np.asarray(result[2], dtype=float)
    target_from_tcp[:3, 3] = np.asarray(result[3], dtype=float).reshape(3)
    return invert(mocap_from_base), invert(target_from_tcp)


def _prediction(
    sample: CalibrationSample,
    base_from_mocap: np.ndarray,
    tcp_from_target: np.ndarray,
) -> np.ndarray:
    mocap_from_base = invert(base_from_mocap)
    return mocap_from_base @ sample.base_from_tcp @ tcp_from_target


def _residuals(
    params: np.ndarray,
    samples: Sequence[CalibrationSample],
    rotation_scale: float,
) -> np.ndarray:
    mocap_from_base = transform_from_params(params[:6])
    tcp_from_target = transform_from_params(params[6:])
    residuals: list[float] = []
    for sample in samples:
        predicted = mocap_from_base @ sample.base_from_tcp @ tcp_from_target
        translation, rotation = pose_error(sample.mocap_from_target, predicted)
        residuals.extend(translation)
        residuals.extend(rotation_scale * rotation)
    return np.asarray(residuals, dtype=float)


def _fit(
    samples: Sequence[CalibrationSample], config: SolverConfig
) -> tuple[str, np.ndarray, np.ndarray]:
    initializers = (
        ("SHAH", cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH),
        ("LI", cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI),
    )
    candidates: list[tuple[float, str, np.ndarray]] = []
    errors: list[str] = []
    for name, method in initializers:
        try:
            base_from_mocap, tcp_from_target = _opencv_initializer(samples, method)
            params = np.concatenate(
                [
                    params_from_transform(invert(base_from_mocap)),
                    params_from_transform(tcp_from_target),
                ]
            )
            score = float(
                np.mean(
                    _residuals(
                        params,
                        samples,
                        config.rotation_residual_scale_m_per_rad,
                    )
                    ** 2
                )
            )
            candidates.append((score, name, params))
        except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
            errors.append(f"{name}: {exc}")
    if not candidates:
        raise ValueError("all OpenCV initializers failed: " + "; ".join(errors))
    _, initializer, initial_params = min(candidates, key=lambda item: item[0])
    optimized = least_squares(
        _residuals,
        initial_params,
        args=(samples, config.rotation_residual_scale_m_per_rad),
        loss="soft_l1",
        f_scale=config.robust_loss_scale,
        max_nfev=5000,
    )
    if not optimized.success or not np.all(np.isfinite(optimized.x)):
        raise ValueError(f"nonlinear refinement failed: {optimized.message}")
    mocap_from_base = transform_from_params(optimized.x[:6])
    tcp_from_target = transform_from_params(optimized.x[6:])
    return initializer, invert(mocap_from_base), tcp_from_target


def calculate_metrics(
    samples: Iterable[CalibrationSample],
    base_from_mocap: np.ndarray,
    tcp_from_target: np.ndarray,
) -> ErrorMetrics:
    translations: list[float] = []
    rotations: list[float] = []
    for sample in samples:
        translation, rotation = pose_error(
            sample.mocap_from_target,
            _prediction(sample, base_from_mocap, tcp_from_target),
        )
        translations.append(float(np.linalg.norm(translation)))
        rotations.append(float(np.rad2deg(np.linalg.norm(rotation))))
    if not translations:
        return ErrorMetrics(0, float("nan"), float("nan"), float("nan"), float("nan"))
    return ErrorMetrics(
        sample_count=len(translations),
        translation_rms_m=sqrt(float(np.mean(np.square(translations)))),
        translation_max_m=max(translations),
        rotation_rms_deg=sqrt(float(np.mean(np.square(rotations)))),
        rotation_max_deg=max(rotations),
    )


def _acceptance_reasons(metrics: ErrorMetrics, config: SolverConfig) -> tuple[str, ...]:
    reasons: list[str] = []
    comparisons = (
        (
            "translation RMS",
            metrics.translation_rms_m,
            config.translation_rms_max_m,
            "m",
        ),
        (
            "translation maximum",
            metrics.translation_max_m,
            config.translation_max_m,
            "m",
        ),
        ("rotation RMS", metrics.rotation_rms_deg, config.rotation_rms_max_deg, "deg"),
        ("rotation maximum", metrics.rotation_max_deg, config.rotation_max_deg, "deg"),
    )
    for name, actual, maximum, unit in comparisons:
        if not np.isfinite(actual) or actual > maximum:
            reasons.append(f"{name} {actual:.6g} {unit} exceeds {maximum:.6g} {unit}")
    return tuple(reasons)


def solve_calibration(
    dataset: CalibrationDataset, config: SolverConfig
) -> CalibrationResult:
    if not dataset.complete:
        raise ValueError(
            "dataset is incomplete; only a fully collected dataset can be accepted"
        )
    samples = tuple(dataset.samples)
    if len(samples) < config.minimum_samples:
        raise ValueError(
            f"dataset has {len(samples)} samples; at least {config.minimum_samples} are required"
        )
    _check_excitation(samples)
    validation_indices = {
        index
        for index in range(len(samples))
        if (index + 1) % config.holdout_every == 0
    }
    if not validation_indices:
        raise ValueError("holdout split produced no validation samples")
    training = tuple(
        sample
        for index, sample in enumerate(samples)
        if index not in validation_indices
    )
    validation = tuple(
        sample for index, sample in enumerate(samples) if index in validation_indices
    )
    initializer, trial_base_from_mocap, trial_tcp_from_target = _fit(training, config)
    training_metrics = calculate_metrics(
        training, trial_base_from_mocap, trial_tcp_from_target
    )
    validation_metrics = calculate_metrics(
        validation, trial_base_from_mocap, trial_tcp_from_target
    )
    reasons = _acceptance_reasons(validation_metrics, config)
    if reasons:
        return CalibrationResult(
            accepted=False,
            initializer=initializer,
            base_from_mocap=trial_base_from_mocap,
            tcp_from_target=trial_tcp_from_target,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            all_sample_metrics=calculate_metrics(
                samples, trial_base_from_mocap, trial_tcp_from_target
            ),
            reasons=reasons,
        )
    final_initializer, base_from_mocap, tcp_from_target = _fit(samples, config)
    return CalibrationResult(
        accepted=True,
        initializer=final_initializer,
        base_from_mocap=base_from_mocap,
        tcp_from_target=tcp_from_target,
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
        all_sample_metrics=calculate_metrics(samples, base_from_mocap, tcp_from_target),
        reasons=(),
    )


def _metrics_dict(metrics: ErrorMetrics) -> dict[str, float | int]:
    return {
        "sample_count": metrics.sample_count,
        "translation_rms_m": metrics.translation_rms_m,
        "translation_max_m": metrics.translation_max_m,
        "rotation_rms_deg": metrics.rotation_rms_deg,
        "rotation_max_deg": metrics.rotation_max_deg,
    }


def write_result(
    output_directory: str | Path,
    result: CalibrationResult,
    dataset: CalibrationDataset,
) -> Path:
    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mocap_from_base = invert(result.base_from_mocap)
    target_from_tcp = invert(result.tcp_from_target)
    payload = {
        "schema_version": 1,
        "accepted": result.accepted,
        "initializer": result.initializer,
        "frames": {
            "base": dataset.base_frame,
            "mocap": dataset.mocap_frame,
            "target": dataset.target_name,
            "tcp": "tcp",
        },
        "transforms": {
            "base_from_mocap": result.base_from_mocap.tolist(),
            "mocap_from_base": mocap_from_base.tolist(),
            "tcp_from_target": result.tcp_from_target.tolist(),
            "target_from_tcp": target_from_tcp.tolist(),
        },
        "metrics": {
            "training": _metrics_dict(result.training_metrics),
            "validation": _metrics_dict(result.validation_metrics),
            "all_samples": _metrics_dict(result.all_sample_metrics),
        },
        "rejection_reasons": list(result.reasons),
    }
    filename = "calibration.yaml" if result.accepted else "calibration_candidate.yaml"
    result_path = directory / filename
    result_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (directory / "metrics.json").write_text(
        json.dumps(payload["metrics"], indent=2) + "\n", encoding="utf-8"
    )
    np.savetxt(directory / "base_from_mocap.txt", result.base_from_mocap)
    np.savetxt(directory / "mocap_from_base.txt", mocap_from_base)
    np.savetxt(directory / "tcp_from_target.txt", result.tcp_from_target)
    return result_path
