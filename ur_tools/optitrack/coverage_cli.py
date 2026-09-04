"""Map OptiTrack coverage by hand-sweeping an OptiTrack CW-250 wand."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from threading import Event, Lock
from typing import Any

from .coverage import (
    CoverageSample,
    coverage_summary,
    interpolate_short_gaps,
    render_projection,
    voxelize_samples,
    write_samples_csv,
    write_voxel_ply,
    write_voxels,
)
from .frame_transform import DEFAULT_CALIBRATION, WorldBaseTransform

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "coverage_results"


class CoverageError(RuntimeError):
    pass


class NatNetCoverageCollector:
    """Collect wand validity and per-marker quality from Motive NatNet."""

    def __init__(
        self,
        *,
        server_ip: str,
        local_ip: str,
        multicast_ip: str,
        target_name: str,
        transform: WorldBaseTransform,
        expected_marker_count: int | None,
        use_unmodeled_centroid: bool,
    ) -> None:
        try:
            from natnet import NatNetClient
        except ImportError as exc:
            raise CoverageError("the natnet package is not installed") from exc
        self.target_name = target_name
        self.transform = transform
        self.requested_marker_count = expected_marker_count
        self.use_unmodeled_centroid = use_unmodeled_centroid
        self._lock = Lock()
        self._ready = Event()
        self._target_id: int | None = None
        self._expected_marker_count = expected_marker_count or 0
        self._available_models: dict[int, str] = {}
        self._samples: list[CoverageSample] = []
        self._maximum_unmodeled_marker_count = 0
        self._maximum_target_marker_count = 0
        if use_unmodeled_centroid and expected_marker_count is not None:
            self._ready.set()
        self.client = NatNetClient(
            server_ip_address=server_ip,
            local_ip_address=local_ip,
            multicast_address=multicast_ip,
            use_multicast=True,
        )
        self.client.on_data_description_received_event.handlers.append(
            self._description_callback
        )
        self.client.on_data_frame_received_event.handlers.append(self._frame_callback)
        self.client.connect()
        self.client.request_modeldef()
        self.client.run_async()

    def _description_callback(self, descriptions: Any) -> None:
        available = {
            int(item.id_num): str(item.name)
            for item in descriptions.rigid_bodies
            if item.name
        }
        target = next(
            (
                item
                for item in descriptions.rigid_bodies
                if item.name == self.target_name
            ),
            None,
        )
        with self._lock:
            self._available_models = available
            if target is not None:
                self._target_id = int(target.id_num)
                described_count = len(target.markers)
                if self.requested_marker_count is None:
                    self._expected_marker_count = described_count
                self._ready.set()

    @staticmethod
    def _marker_is_observed(marker: Any) -> bool:
        return bool(
            marker.param is not None
            and not marker.occluded
            and marker.point_cloud_solved
        )

    def _frame_callback(self, frame: Any) -> None:
        if frame.suffix.tracked_models_changed:
            self.client.request_modeldef()
        with self._lock:
            target_id = self._target_id
            expected = self._expected_marker_count
        if target_id is None and not self.use_unmodeled_centroid:
            return
        body = next(
            (
                item
                for item in frame.rigid_bodies
                if target_id is not None and item.id_num == target_id
            ),
            None,
        )
        markers = [
            marker
            for marker in frame.labeled_markers
            if target_id is not None and marker.model_id == target_id
        ]
        unmodeled = [
            marker
            for marker in frame.labeled_markers
            if marker.model_id == 0 and self._marker_is_observed(marker)
        ]
        observed = [marker for marker in markers if self._marker_is_observed(marker)]
        selected_markers = unmodeled if self.use_unmodeled_centroid else observed
        residuals = [
            float(marker.residual)
            for marker in selected_markers
            if marker.residual is not None and math.isfinite(marker.residual)
        ]
        if self.use_unmodeled_centroid:
            valid = bool(
                expected > 0
                and len(unmodeled) == expected
                and all(
                    len(marker.pos) == 3
                    and all(math.isfinite(float(value)) for value in marker.pos)
                    for marker in unmodeled
                )
            )
        else:
            valid = bool(body is not None and body.tracking_valid)
        world = None
        base = None
        rigid_error = None
        if self.use_unmodeled_centroid and valid:
            world = tuple(
                sum(float(marker.pos[axis]) for marker in unmodeled) / expected
                for axis in range(3)
            )
            base_array = self.transform.world_to_base(world)
            base = tuple(float(value) for value in base_array)
        elif body is not None:
            rigid_error = (
                None if body.marker_error is None else float(body.marker_error)
            )
            if valid:
                world = tuple(float(value) for value in body.pos)
                base_array = self.transform.world_to_base(world)
                base = tuple(float(value) for value in base_array)
        sample = CoverageSample(
            frame_number=int(frame.prefix.frame_number),
            received_monotonic_s=time.monotonic(),
            source_time_s=float(frame.suffix.timestamp),
            tracking_valid=valid,
            position_world=world,  # type: ignore[arg-type]
            position_base=base,  # type: ignore[arg-type]
            observed_marker_count=len(selected_markers),
            expected_marker_count=expected,
            rigid_body_error=rigid_error,
            marker_residual_mean=(
                None if not residuals else sum(residuals) / len(residuals)
            ),
            marker_residual_max=None if not residuals else max(residuals),
        )
        with self._lock:
            self._samples.append(sample)
            self._maximum_unmodeled_marker_count = max(
                self._maximum_unmodeled_marker_count, len(unmodeled)
            )
            self._maximum_target_marker_count = max(
                self._maximum_target_marker_count, len(observed)
            )

    def wait_for_target(self, timeout_s: float) -> None:
        if self._ready.wait(timeout_s):
            return
        with self._lock:
            names = sorted(self._available_models.values())
        raise CoverageError(
            f"Motive rigid body {self.target_name!r} was not found; available: "
            + (", ".join(names) if names else "none")
        )

    @property
    def target_id(self) -> int | None:
        with self._lock:
            if self._target_id is None and not self.use_unmodeled_centroid:
                raise CoverageError("target model is not ready")
            return self._target_id

    @property
    def expected_marker_count(self) -> int:
        with self._lock:
            return self._expected_marker_count

    def snapshot(self) -> list[CoverageSample]:
        with self._lock:
            return list(self._samples)

    def wait_for_valid_tracking(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if any(sample.tracking_valid for sample in self._samples):
                    return
            time.sleep(0.05)
        with self._lock:
            unmodeled = self._maximum_unmodeled_marker_count
            target_markers = self._maximum_target_marker_count
        if self.use_unmodeled_centroid:
            raise CoverageError(
                "The unmodeled-marker centroid never became valid; "
                f"expected exactly {self.expected_marker_count} reconstructed "
                f"markers but observed at most {unmodeled}. Remove or cover all "
                "other passive markers and keep all CW-250 markers visible."
            )
        raise CoverageError(
            f"Motive rigid body {self.target_name!r} never became valid; "
            f"maximum target-labeled markers={target_markers}, maximum "
            f"unmodeled reconstructed markers={unmodeled}. Enable or recreate "
            "the asset from the current CW-250 markers before sweeping, or use "
            "--use-unmodeled-centroid with no other passive markers present."
        )

    def close(self) -> None:
        self.client.shutdown()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--server-ip", default="172.16.90.213")
    parser.add_argument("--local-ip", default="172.16.90.195")
    parser.add_argument("--multicast-ip", default="239.255.42.99")
    parser.add_argument("--target-name", default="coverage_wand")
    parser.add_argument("--expected-marker-count", type=int)
    parser.add_argument(
        "--use-unmodeled-centroid",
        action="store_true",
        help=(
            "track the centroid of exactly the expected number of unmodeled "
            "reconstructed markers instead of the Motive rigid-body pose"
        ),
    )
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--voxel-size-m", type=float, default=0.05)
    parser.add_argument("--bounds-min", nargs=3, type=float, default=(-1.5, -2.0, 0.0))
    parser.add_argument("--bounds-max", nargs=3, type=float, default=(1.5, 0.0, 1.5))
    parser.add_argument("--maximum-interpolated-gap-s", type=float, default=0.25)
    parser.add_argument("--minimum-voxel-samples", type=int, default=5)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--target-timeout-s", type=float, default=10.0)
    parser.add_argument("--initial-valid-timeout-s", type=float, default=5.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.duration_s) or args.duration_s < 0:
        raise CoverageError("duration must be finite and non-negative")
    if not math.isfinite(args.voxel_size_m) or args.voxel_size_m <= 0:
        raise CoverageError("voxel size must be positive and finite")
    if args.expected_marker_count is not None and args.expected_marker_count < 1:
        raise CoverageError("expected marker count must be positive")
    if args.minimum_voxel_samples < 1:
        raise CoverageError("minimum voxel samples must be positive")
    if not math.isfinite(args.target_timeout_s) or args.target_timeout_s <= 0:
        raise CoverageError("target timeout must be positive and finite")
    if (
        not math.isfinite(args.initial_valid_timeout_s)
        or args.initial_valid_timeout_s <= 0
    ):
        raise CoverageError("initial valid timeout must be positive and finite")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_results(
    output: Path,
    *,
    args: argparse.Namespace,
    transform: WorldBaseTransform,
    target_id: int | None,
    expected_marker_count: int,
    samples: list[CoverageSample],
) -> dict[str, object]:
    localized = interpolate_short_gaps(samples, args.maximum_interpolated_gap_s)
    voxels = voxelize_samples(
        localized,
        bounds_min=args.bounds_min,
        bounds_max=args.bounds_max,
        voxel_size_m=args.voxel_size_m,
        minimum_samples=args.minimum_voxel_samples,
    )
    output.mkdir(parents=True, exist_ok=False)
    write_samples_csv(output / "samples.csv", localized)
    write_voxels(output / "voxels", voxels)
    write_voxel_ply(output / "coverage_voxels.ply", voxels)
    render_projection(output / "coverage_xy.png", voxels, (0, 1), ("X", "Y"))
    render_projection(output / "coverage_xz.png", voxels, (0, 2), ("X", "Z"))
    render_projection(output / "coverage_yz.png", voxels, (1, 2), ("Y", "Z"))
    summary = coverage_summary(localized, voxels, args.voxel_size_m)
    metadata = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": {
            "name": args.target_name,
            "rigid_body_id": target_id,
            "expected_marker_count": expected_marker_count,
            "tracking_mode": (
                "unmodeled_marker_centroid"
                if args.use_unmodeled_centroid
                else "motive_rigid_body"
            ),
            "probe": (
                "centroid of exactly the expected number of unmodeled CW-250 markers"
                if args.use_unmodeled_centroid
                else "OptiTrack CW-250 configured as a Motive rigid body"
            ),
        },
        "network": {
            "server_ip": args.server_ip,
            "local_ip": args.local_ip,
            "multicast_ip": args.multicast_ip,
        },
        "frames": {"source": transform.world_frame, "output": transform.base_frame},
        "calibration": {
            "path": str(transform.source_path),
            "sha256": _sha256(transform.source_path),
        },
        "settings": {
            "duration_s": args.duration_s,
            "voxel_size_m": args.voxel_size_m,
            "bounds_min_base_m": list(args.bounds_min),
            "bounds_max_base_m": list(args.bounds_max),
            "maximum_interpolated_gap_s": args.maximum_interpolated_gap_s,
            "minimum_voxel_samples": args.minimum_voxel_samples,
            "reliable_detection_fraction": 0.99,
            "reliable_complete_marker_fraction": 0.95,
            "marginal_detection_fraction": 0.95,
            "marginal_complete_marker_fraction": 0.67,
        },
        "summary": summary,
        "limitations": [
            "NatNet does not expose per-marker camera-ray counts in frame data.",
            "Unbounded leading/trailing dropouts cannot be spatially localized.",
            "Short dropout positions are linearly interpolated only between valid poses.",
            "A CW-250 rigid body is more observable than a single drone marker; use the per-marker completeness metric conservatively.",
            *(
                [
                    "Centroid mode has no rigid-body orientation or marker identity.",
                    "Centroid mode is valid only when every reconstructed unmodeled marker belongs to the CW-250.",
                ]
                if args.use_unmodeled_centroid
                else []
            ),
        ],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    collector = None
    try:
        _validate_args(args)
        transform = WorldBaseTransform.load(args.calibration)
        print(f"Coverage target: {args.target_name}")
        print(
            f"UR-base bounds: {tuple(args.bounds_min)} to {tuple(args.bounds_max)}, "
            f"voxel={args.voxel_size_m:.3f} m"
        )
        if args.use_unmodeled_centroid:
            print(
                "Centroid mode: every frame must contain exactly the expected "
                "number of unmodeled reconstructed markers. Remove or cover all "
                "drone markers and other passive reflections."
            )
        if not args.execute:
            if args.use_unmodeled_centroid:
                print(
                    "DRY RUN: enable labeled-marker streaming, leave only the "
                    "CW-250 markers visible, then add --execute."
                )
            else:
                print(
                    "DRY RUN: create a Motive rigid body named "
                    f"{args.target_name!r} from the CW-250 markers, enable "
                    "rigid-body and labeled-marker streaming, then add --execute."
                )
            return 0
        collector = NatNetCoverageCollector(
            server_ip=args.server_ip,
            local_ip=args.local_ip,
            multicast_ip=args.multicast_ip,
            target_name=args.target_name,
            transform=transform,
            expected_marker_count=args.expected_marker_count,
            use_unmodeled_centroid=args.use_unmodeled_centroid,
        )
        collector.wait_for_target(args.target_timeout_s)
        expected = collector.expected_marker_count
        if expected < 1:
            raise CoverageError(
                "Motive did not describe wand markers; pass --expected-marker-count"
            )
        collector.wait_for_valid_tracking(args.initial_valid_timeout_s)
        if args.use_unmodeled_centroid:
            reference = (
                "none" if collector.target_id is None else str(collector.target_id)
            )
            print(
                f"Connected in unmodeled-centroid mode "
                f"(reference model id={reference}, markers={expected})."
            )
        else:
            print(
                f"Connected to rigid body {args.target_name!r} "
                f"(id={collector.target_id}, markers={expected})."
            )
        print(
            "Sweep the CW-250 slowly through the flight volume. Press Ctrl+C to stop."
        )
        started = time.monotonic()
        next_progress = started + 5.0
        while args.duration_s == 0 or time.monotonic() - started < args.duration_s:
            time.sleep(0.1)
            now = time.monotonic()
            if now >= next_progress:
                samples = collector.snapshot()
                valid = sum(sample.tracking_valid for sample in samples)
                fraction = valid / len(samples) if samples else 0.0
                print(
                    f"Frames={len(samples)}, tracking={fraction * 100:.1f}%",
                    flush=True,
                )
                next_progress = now + 5.0
        samples = collector.snapshot()
    except KeyboardInterrupt:
        samples = [] if collector is None else collector.snapshot()
        print("Sweep stopped by operator; preserving collected data.")
    except (CoverageError, OSError, TypeError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    finally:
        if collector is not None:
            collector.close()

    if not samples:
        print("FAILED: no NatNet coverage frames were collected", file=sys.stderr)
        return 2
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output = Path(args.output_root).expanduser().resolve() / run_id
    try:
        metadata = _write_results(
            output,
            args=args,
            transform=transform,
            target_id=collector.target_id,
            expected_marker_count=collector.expected_marker_count,
            samples=samples,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"FAILED to write coverage results: {exc}", file=sys.stderr)
        return 2
    summary = metadata["summary"]
    print(f"Coverage map written to {output}")
    print(
        f"Detection={summary['overall_detection_fraction']}, "
        f"reliable voxels={summary['voxel_grade_counts']['reliable']}, "
        f"blind voxels={summary['voxel_grade_counts']['blind']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
