"""Small, explicit SE(3) helpers used by the calibration pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def validate_transform(
    value: np.ndarray | Sequence[Sequence[float]], name: str
) -> np.ndarray:
    """Return a checked 4x4 rigid transform."""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation must have determinant +1")
    return matrix.copy()


def transform_from_rotvec_translation(
    rotvec: Sequence[float], translation: Sequence[float]
) -> np.ndarray:
    rotvec_array = np.asarray(rotvec, dtype=float)
    translation_array = np.asarray(translation, dtype=float)
    if rotvec_array.shape != (3,) or translation_array.shape != (3,):
        raise ValueError(
            "rotation vector and translation must each contain three values"
        )
    if not np.all(np.isfinite(rotvec_array)) or not np.all(
        np.isfinite(translation_array)
    ):
        raise ValueError("rotation vector and translation must be finite")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotvec_array).as_matrix()
    result[:3, 3] = translation_array
    return result


def transform_from_pose_vector(pose: Sequence[float]) -> np.ndarray:
    """Convert an RTDE [x, y, z, rx, ry, rz] pose to a transform."""
    values = np.asarray(pose, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("RTDE pose must contain six finite values")
    return transform_from_rotvec_translation(values[3:], values[:3])


def transform_from_quaternion_translation(
    quaternion_xyzw: Sequence[float], translation: Sequence[float]
) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    translation_array = np.asarray(translation, dtype=float)
    if quaternion.shape != (4,) or translation_array.shape != (3,):
        raise ValueError("quaternion must contain four values and translation three")
    if not np.all(np.isfinite(quaternion)) or not np.all(
        np.isfinite(translation_array)
    ):
        raise ValueError("quaternion and translation must be finite")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-9:
        raise ValueError("quaternion norm must be non-zero")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    result[:3, 3] = translation_array
    return result


def invert(transform: np.ndarray) -> np.ndarray:
    matrix = validate_transform(transform, "transform")
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def params_from_transform(transform: np.ndarray) -> np.ndarray:
    matrix = validate_transform(transform, "transform")
    return np.concatenate(
        [Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3]]
    )


def transform_from_params(params: Sequence[float]) -> np.ndarray:
    values = np.asarray(params, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("transform parameters must contain six finite values")
    return transform_from_rotvec_translation(values[:3], values[3:])


def pose_error(
    observed: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return translation and rotation-vector errors in the observed frame."""
    delta = invert(observed) @ validate_transform(predicted, "predicted")
    return delta[:3, 3], Rotation.from_matrix(delta[:3, :3]).as_rotvec()


def quaternion_xyzw(transform: np.ndarray) -> np.ndarray:
    matrix = validate_transform(transform, "transform")
    return Rotation.from_matrix(matrix[:3, :3]).as_quat()
