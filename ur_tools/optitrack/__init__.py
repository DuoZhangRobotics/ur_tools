"""OptiTrack-to-UR calibration tools.

The solver and data model intentionally have no ROS or RTDE imports so recorded
datasets can be inspected and solved on any development machine.
"""

from .config import CalibrationConfig, ConfigError, load_config
from .dataset import CalibrationDataset, CalibrationSample, load_dataset, save_dataset
from .solver import CalibrationResult, solve_calibration, write_result

__all__ = [
    "CalibrationConfig",
    "CalibrationDataset",
    "CalibrationResult",
    "CalibrationSample",
    "ConfigError",
    "load_config",
    "load_dataset",
    "save_dataset",
    "solve_calibration",
    "write_result",
]
