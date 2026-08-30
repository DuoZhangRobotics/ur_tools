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
drone operation uses only the promoted static transform.

See [docs/OPTITRACK_CALIBRATION.md](docs/OPTITRACK_CALIBRATION.md) for the uv
environment, Motive setup, dry run, collection, offline solve, output files, and
static-TF command.

The default profile targets:

- UR5e: `172.16.90.197`
- Motive: `172.16.90.213`
- ROS topic: `/poses`
- Motive rigid body: `ur_calib`
- Runtime calibration: `config/optitrack_to_ur_base.yaml`

Running `ur-optitrack-calibrate` without `--execute` is always a no-motion
dry run. Automatic movement additionally requires typing the robot-IP
confirmation shown by the command.

In the script `rtde_control_vg.script', make sure to use channel 2 to include both channel A and B, or the suction power isn't enough. Channel 0 is A, channel 1 is B and channel 2 is both. Refer to the manual for more detailed instruction.
