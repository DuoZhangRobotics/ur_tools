from pathlib import Path

import numpy as np
import pytest
import yaml
from builtin_interfaces.msg import Time

from ur_tools.optitrack.static_transform import (
    DEFAULT_CALIBRATION,
    load_static_transform,
    transform_message,
)


def test_repository_calibration_loads_in_tf_direction() -> None:
    stored = load_static_transform(DEFAULT_CALIBRATION)
    message = transform_message(stored, Time(sec=123))

    assert stored.parent_frame == "base"
    assert stored.child_frame == "world"
    assert stored.matrix.shape == (4, 4)
    assert message.header.frame_id == "base"
    assert message.child_frame_id == "world"
    assert message.header.stamp.sec == 123
    assert message.transform.translation.x == pytest.approx(-0.0710302263)
    quaternion = np.array(
        [
            message.transform.rotation.x,
            message.transform.rotation.y,
            message.transform.rotation.z,
            message.transform.rotation.w,
        ]
    )
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)


def test_unaccepted_calibration_is_rejected(tmp_path: Path) -> None:
    source = yaml.safe_load(DEFAULT_CALIBRATION.read_text())
    source["accepted"] = False
    path = tmp_path / "unaccepted.yaml"
    path.write_text(yaml.safe_dump(source))

    with pytest.raises(ValueError, match="accepted"):
        load_static_transform(path)
