"""Command-line entry point for dry-run, collection, and offline solving."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import ConfigError, load_config
from .dataset import load_dataset
from .solver import solve_calibration, write_result

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "optitrack_ur5e_197.yaml"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ur-optitrack-calibrate",
        description="Calibrate the OptiTrack world frame to a UR base frame.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="reviewed calibration YAML profile",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="permit the reviewed automatic robot motions",
    )
    parser.add_argument(
        "--solve-only",
        metavar="DATASET",
        help="solve a recorded samples.yaml without ROS, RTDE, or robot motion",
    )
    return parser


def _print_dry_run(config) -> None:
    print("DRY RUN: no ROS connection and no robot command will be made.")
    print(f"Robot:       {config.robot_ip}")
    print(f"Motive:      {config.motive_ip}")
    print(f"Pose input:  {config.poses_topic} / {config.target_name}")
    print(f"Frames:      {config.mocap_frame} -> {config.base_frame}")
    print(f"Joint home:  {config.motion.home_joints}")
    print("Reviewed Cartesian motions:")
    for index, (name, pose) in enumerate(config.motion.poses, start=1):
        print(f"  {index:02d}. {name}: {pose}")
    print(
        "Run again with --execute only after checking the mounted target, "
        "pose list, and robot workspace."
    )


def _timestamped_directory(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / timestamp
    suffix = 1
    while destination.exists():
        destination = root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    destination.mkdir(parents=True)
    return destination


def _solve(dataset, config, output_directory: Path) -> int:
    result = solve_calibration(dataset, config.solver)
    result_path = write_result(output_directory, result, dataset)
    print(f"Calibration result: {result_path}")
    print(
        "Held-out errors: "
        f"{result.validation_metrics.translation_rms_m * 1000:.3f} mm RMS, "
        f"{result.validation_metrics.translation_max_m * 1000:.3f} mm max; "
        f"{result.validation_metrics.rotation_rms_deg:.3f} deg RMS, "
        f"{result.validation_metrics.rotation_max_deg:.3f} deg max"
    )
    if not result.accepted:
        print("REJECTED: " + "; ".join(result.reasons), file=sys.stderr)
        return 2
    print("ACCEPTED")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.solve_only:
            dataset = load_dataset(args.solve_only)
            destination = _timestamped_directory(config.output_root)
            shutil.copy2(config.source_path, destination / "profile.yaml")
            shutil.copy2(Path(args.solve_only).resolve(), destination / "samples.yaml")
            return _solve(dataset, config, destination)

        if not args.execute:
            _print_dry_run(config)
            return 0

        phrase = f"MOVE UR5E {config.robot_ip}"
        if not sys.stdin.isatty():
            raise ConfigError("--execute requires an interactive terminal")
        entered = input(f"Type {phrase!r} to start automatic motion: ").strip()
        if entered != phrase:
            print("Confirmation did not match; no motion was sent.", file=sys.stderr)
            return 2

        from .collector import collect_dataset

        destination = _timestamped_directory(config.output_root)
        shutil.copy2(config.source_path, destination / "profile.yaml")
        dataset_path = destination / "samples.yaml"
        dataset = collect_dataset(config, dataset_path)
        return _solve(dataset, config, destination)
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "Interrupted; robot stop and dataset preservation requested.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
