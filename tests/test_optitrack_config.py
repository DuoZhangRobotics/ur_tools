from pathlib import Path

import pytest
import yaml

from ur_tools.optitrack.config import ConfigError, load_config
from ur_tools.optitrack.dataset import (
    CalibrationDataset,
    CalibrationSample,
    load_dataset,
    save_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config" / "optitrack_ur5e_197.yaml"


def test_default_profile_is_explicit_and_reviewable() -> None:
    config = load_config(PROFILE)

    assert config.robot_ip == "172.16.90.197"
    assert config.motive_ip == "172.16.90.213"
    assert config.target_name == "ur_calib"
    assert config.base_frame == "base"
    assert config.mocap_frame == "world"
    assert len(config.motion.poses) == 25
    assert len({name for name, _ in config.motion.poses}) == 25


def test_invalid_ip_is_rejected(tmp_path: Path) -> None:
    source = yaml.safe_load(PROFILE.read_text())
    source["robot"]["ip"] = "not-an-ip"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(source))

    with pytest.raises(ConfigError, match="valid IP"):
        load_config(path)


def test_duplicate_pose_name_is_rejected(tmp_path: Path) -> None:
    source = yaml.safe_load(PROFILE.read_text())
    source["motion"]["poses"][1]["name"] = source["motion"]["poses"][0]["name"]
    path = tmp_path / "duplicate.yaml"
    path.write_text(yaml.safe_dump(source))

    with pytest.raises(ConfigError, match="duplicate"):
        load_config(path)


def test_dataset_is_atomically_round_tripped(tmp_path: Path) -> None:
    import numpy as np

    dataset = CalibrationDataset(
        robot_ip="172.16.90.197",
        base_frame="base",
        mocap_frame="world",
        target_name="ur_calib",
        samples=[
            CalibrationSample(
                name="pose_01",
                captured_at_unix_s=123.0,
                base_from_tcp=np.eye(4),
                mocap_from_target=np.eye(4),
                message_count=60,
            )
        ],
        complete=False,
    )
    path = save_dataset(tmp_path / "samples.yaml", dataset)
    restored = load_dataset(path)

    assert not restored.complete
    assert restored.samples[0].name == "pose_01"
    assert restored.samples[0].message_count == 60
    assert not (tmp_path / "samples.yaml.tmp").exists()
