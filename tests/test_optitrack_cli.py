import sys
from pathlib import Path

from ur_tools.optitrack.cli import main

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config" / "optitrack_ur5e_197.yaml"


def test_default_mode_is_a_hardware_free_dry_run(capsys) -> None:
    sys.modules.pop("rtde_control", None)
    sys.modules.pop("rclpy", None)

    result = main(["--config", str(PROFILE)])

    output = capsys.readouterr().out
    assert result == 0
    assert "DRY RUN" in output
    assert "pose_25" in output
    assert "rtde_control" not in sys.modules
    assert "rclpy" not in sys.modules


def test_execute_requires_an_interactive_terminal(monkeypatch, capsys) -> None:
    sys.modules.pop("rtde_control", None)
    sys.modules.pop("rclpy", None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    result = main(["--config", str(PROFILE), "--execute"])

    assert result == 2
    assert "interactive terminal" in capsys.readouterr().err
    assert "rtde_control" not in sys.modules
    assert "rclpy" not in sys.modules
