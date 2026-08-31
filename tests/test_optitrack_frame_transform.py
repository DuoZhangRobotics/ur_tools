from pathlib import Path

import numpy as np
import pytest
import yaml

from ur_tools.optitrack.frame_transform import (
    DEFAULT_CALIBRATION,
    WorldBaseTransform,
)
from ur_tools.optitrack.transforms import transform_from_rotvec_translation


def test_repository_calibration_converts_state_and_goal_round_trip() -> None:
    transform = WorldBaseTransform.load()
    point_world = np.array([0.25, -0.1, 0.4])

    point_base = transform.world_to_base(point_world)
    restored_world = transform.base_to_world(point_base)

    assert transform.base_frame == "base"
    assert transform.world_frame == "world"
    assert transform.source_path == DEFAULT_CALIBRATION.resolve()
    assert transform.base_from_world[0, 3] == pytest.approx(-0.0710346707)
    assert restored_world == pytest.approx(point_world, abs=1e-12)


def test_vectors_do_not_receive_translation() -> None:
    transform = WorldBaseTransform.load()
    zero = transform.world_vector_to_base([0.0, 0.0, 0.0])
    vector = np.array([0.1, -0.2, 0.3])

    assert zero == pytest.approx([0.0, 0.0, 0.0])
    assert transform.base_vector_to_world(
        transform.world_vector_to_base(vector)
    ) == pytest.approx(vector, abs=1e-12)


def test_pose_conversion_round_trip() -> None:
    transform = WorldBaseTransform.load()
    world_from_object = transform_from_rotvec_translation(
        [0.1, -0.2, 0.3], [0.4, 0.5, 0.6]
    )

    base_from_object = transform.world_pose_to_base(world_from_object)
    restored = transform.base_pose_to_world(base_from_object)

    assert restored == pytest.approx(world_from_object, abs=1e-12)


def test_unaccepted_calibration_is_rejected(tmp_path: Path) -> None:
    source = yaml.safe_load(DEFAULT_CALIBRATION.read_text())
    source["accepted"] = False
    path = tmp_path / "unaccepted.yaml"
    path.write_text(yaml.safe_dump(source))

    with pytest.raises(ValueError, match="accepted"):
        WorldBaseTransform.load(path)
