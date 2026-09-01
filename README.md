# RobotControl

UR5/UR5e control and calibration utilities for RealSense, ChArUco, and
OptiTrack-based lab setups.

## OptiTrack world to UR base calibration

The OptiTrack calibration tool pairs a Motive rigid-body pose with the UR5e TCP
pose at 25 reviewed robot configurations. It estimates both the OptiTrack-world
to UR-base transform and the unknown calibration-target mounting transform.
Motive 2.0 rigid bodies are received by the project’s direct NatNet 3.0
publisher and forwarded to the calibration collector on ROS `/poses`.
`ur_calib` and its marker board are temporary calibration equipment; normal
drone operation loads the promoted matrix directly with
`WorldBaseTransform`.

See [docs/OPTITRACK_CALIBRATION.md](docs/OPTITRACK_CALIBRATION.md) for the uv
environment, Motive setup, dry run, collection, offline solve, output files, and
ROS-free runtime state/goal conversions.

The default profile targets:

- UR5e: `172.16.90.197`
- Motive: `172.16.90.213`
- ROS topic: `/poses`
- Motive rigid body: `ur_calib`
- Runtime calibration: `config/optitrack_to_ur_base.yaml`

Running `ur-optitrack-calibrate` without `--execute` is always a no-motion
dry run. Automatic movement additionally requires typing the robot-IP
confirmation shown by the command.

## OptiTrack coverage mapping with the CW-250

`ur-optitrack-coverage` maps observed tracking reliability into 5 cm voxels in
the calibrated UR `base` frame. It connects directly to Motive over NatNet and
does not move the robot or control a drone.

In Motive:

1. Select the CW-250 markers and create a rigid body.
2. Rename it exactly `coverage_wand`.
3. Enable rigid-body and labeled-marker streaming.
4. Confirm the asset reports a valid full pose while the wand is stationary.

Review the hardware-free dry run:

```sh
source tools/activate_ros.sh
ur-optitrack-coverage
```

Collect a two-minute sweep over the active drone volume:

```sh
ur-optitrack-coverage --execute \
  --target-name coverage_wand \
  --duration-s 120 \
  --voxel-size-m 0.05 \
  --bounds-min -1.5 -2.0 0.0 \
  --bounds-max 1.5 0.0 1.5
```

If Motive reconstructs the three CW-250 spheres but its rigid body remains
invalid at the origin, use the marker-centroid fallback:

```sh
ur-optitrack-coverage --execute \
  --use-unmodeled-centroid \
  --expected-marker-count 3 \
  --duration-s 120 \
  --voxel-size-m 0.05 \
  --bounds-min -1.5 -2.0 0.0 \
  --bounds-max 1.5 0.0 1.5
```

Centroid mode accepts a frame only when exactly three unmodeled reconstructed
markers are present. Remove or cover all drone markers and other passive
reflectors during the sweep. It maps the triangle centroid and does not recover
wand orientation or marker identity. With `--expected-marker-count`, this mode
does not require a valid or correctly named Motive rigid-body asset.

Move slowly, cover multiple wand orientations in every region, and dwell in
suspected blind spots. Use `--duration-s 0` for an operator-stopped sweep.

Each timestamped directory under `coverage_results/` contains:

- `samples.csv`: every NatNet frame, validity, marker completeness, residual,
  rigid-body error, calibrated position, and localized short-gap state.
- `voxels.csv` and `voxels.json`: detection fraction, complete-marker
  fraction, residuals, gaps, and `reliable`/`marginal`/`blind` grades.
- `coverage_voxels.ply`: colored 3D voxel centers for CloudCompare or MeshLab.
- `coverage_xy.png`, `coverage_xz.png`, and `coverage_yz.png`: conservative
  minimum-detection projections; gray cells were not sampled.
- `metadata.json`: settings, calibration hash, model ID, summary volumes, and
  documented limitations.

Only short gaps bounded by valid poses are spatially interpolated. Leading,
trailing, or long dropouts remain unlocalized. NatNet does not expose Motive's
per-marker camera-ray count in frame data, so inspect **Marker Rays** in Motive
when diagnosing a marginal voxel. A three-marker CW-250 rigid body is more
observable than one drone marker; treat complete-marker fraction as the more
conservative signal and verify the final flight path with an actual drone
marker before use.

In the script `rtde_control_vg.script', make sure to use channel 2 to include both channel A and B, or the suction power isn't enough. Channel 0 is A, channel 1 is B and channel 2 is both. Refer to the manual for more detailed instruction.
