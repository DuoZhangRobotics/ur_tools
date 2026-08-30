# OptiTrack to UR5e base calibration

This workflow estimates `T_base_mocap`, which maps positions expressed in the
Motive `world` frame into the UR5e `base` frame. It also estimates the fixed
`T_tcp_target` mounting transform of a full-pose OptiTrack rigid body attached
to the tool.

## One-time uv environment

The environment is isolated from system Python while retaining the apt-installed
ROS 2 packages:

```sh
cd /home/duo/ur_tools
uv venv --python 3.12 --system-site-packages
source /home/duo/ur_tools/tools/activate_ros.sh
uv pip install -e '.[test]'
```

In each new terminal, run `source /home/duo/ur_tools/tools/activate_ros.sh`.

## Motive and ROS

1. Rigidly attach a three-or-more-marker fixture to the UR tool.
2. In Motive, create and name the rigid body `ur_calib`.
3. Keep OptiTrack frame streaming, rigid bodies, and Z-up enabled.
4. Start only the motion-capture receiver in a dedicated terminal:

```sh
source /home/duo/ur_tools/tools/activate_ros.sh
ur-optitrack-publisher \
  --server-ip 172.16.90.213 \
  --local-ip 172.16.90.195 \
  --target-name ur_calib
```

The publisher automatically loads
`config/optitrack_to_ur_base.yaml` and broadcasts the accepted
`base -> world` transform on `/tf_static`. Use `--no-static-tf` only while
producing a replacement calibration.

Confirm that the full pose is finite:

```sh
ros2 topic echo --once /poses
```

The orientation for `ur_calib` must be a finite quaternion. A single-marker
pose with NaN orientation is intentionally rejected.

## Review and run

First print the complete plan without connecting to ROS or the robot:

```sh
ur-optitrack-calibrate \
  --config /home/duo/ur_tools/config/optitrack_ur5e_197.yaml
```

Review the home joints, all Cartesian poses, fixture clearance, and robot
workspace. Then run:

```sh
ur-optitrack-calibrate \
  --config /home/duo/ur_tools/config/optitrack_ur5e_197.yaml \
  --execute
```

Execution requires typing `MOVE UR5E 172.16.90.197`. Ctrl+C requests `stopL`
and `stopJ`, preserves all completed samples, and never marks an incomplete
dataset as accepted.

## Offline solve and outputs

Every run creates a timestamped directory under `calibration_results/`.
Collection writes `samples.yaml` atomically after every pose. To solve it again
without ROS, RTDE, or motion:

```sh
ur-optitrack-calibrate \
  --config /home/duo/ur_tools/config/optitrack_ur5e_197.yaml \
  --solve-only /path/to/samples.yaml
```

Accepted runs contain `calibration.yaml`, matrices for both transform
directions, the tool/target transform, held-out metrics, and the exact
`static_transform_publisher` command. Failed acceptance thresholds produce
`calibration_candidate.yaml`, exit non-zero, and do not print a TF command.
