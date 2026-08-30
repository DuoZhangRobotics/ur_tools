# OptiTrack world to UR5e base calibration

This runbook calibrates the fixed transform between the OptiTrack/Motive
`world` frame and the UR5e `base` frame. The calibration target is a
temporary rigid body attached to the end effector.

The solver estimates:

- `T_base_mocap`: maps Motive `world` coordinates into UR `base`.
- `T_tcp_target`: the unknown rigid transform from the UR TCP to the
  temporary marker board.

The board and the Motive asset `ur_calib` are not required after calibration.
Normal drone experiments retain only the accepted `T_base_mocap` file.

## Lab configuration

| Item | Value |
| --- | --- |
| UR5e | `172.16.90.197` |
| Motive PC | `172.16.90.213` |
| Ubuntu wired interface | `172.16.90.195` |
| NatNet multicast | `239.255.42.99` |
| NatNet ports | command `1510`, data `1511` |
| ROS pose topic | `/poses` |
| Motive frame | `world` |
| UR frame | `base` |
| Calibration rigid body | `ur_calib` |
| Motion profile | `config/optitrack_ur5e_197.yaml` |
| Runtime transform | `config/optitrack_to_ur_base.yaml` |

## Software environment

Create the project-local uv environment once:

```sh
cd /home/duo/ur_tools
uv venv --python 3.12 --system-site-packages
source /home/duo/ur_tools/tools/activate_ros.sh
uv pip install -e '.[test]'
```

For later terminals:

```sh
cd /home/duo/ur_tools
source tools/activate_ros.sh
```

The environment is isolated from system Python. ROS is sourced only because the
live collector uses `/poses` as a local message bridge:

```text
Motive -> direct NatNet client -> ROS /poses -> calibration collector
UR5e  -> RTDE -------------------------------> calibration collector
```

RTDE control, sample storage, optimization, and offline solving do not depend on
ROS. The direct NatNet client is required because the installed
`motion_capture_tracking` Direct backend does not publish Motive 2.0 rigid
bodies, and its bundled NatNet 4.1 SDK fails against Motive 2.0 with
`NatNetSDK Error 3`.

## Build the calibration target

Use a stiff, lightweight, matte board attached rigidly to the end effector.
The lab board is approximately 10 cm by 10 cm.

- Prefer four spherical passive markers.
- Spread markers across the board.
- Use an asymmetric layout; do not form a square or equilateral triangle.
- Coplanar markers are acceptable, but all four must remain visible throughout
  the motion profile.
- Secure the board and every marker so the geometry cannot flex.
- Cover shiny tape, battery foil, and reflective hardware with matte material.

In Motive:

1. Select the four reconstructed markers.
2. Create a Rigid Body.
3. Rename the asset exactly `ur_calib`.
4. Enable the asset in the Assets pane.
5. Confirm a full 6DoF pose and coordinate axes are visible.

### Motive reconstruction settings

In Motive 2.x, open **Settings -> Live Reconstruction**.

- Keep frame data, rigid bodies, and Z-up streaming enabled.
- Use multicast transmission.
- Use the default NatNet ports 1510/1511.
- Set **Minimum Ray Count** to 3 when camera coverage permits.

If one physical marker becomes two reconstructed points, select the false point
and inspect its tracked rays. A two-ray ghost is normally removed by requiring
three rays. If real markers disappear after this change, improve camera
visibility rather than flying with ambiguous markers.

Check Motive masks after moving or remounting the board. A valid marker hidden
inside a camera mask can make `ur_calib` invalid at only some robot poses.

## Prepare the UR5e

1. Load the correct UR installation file and TCP definition.
2. Release the emergency stop, power the arm, and release the brakes.
3. Use Remote Control mode if required by PolyScope.
4. Attach the marker board securely.
5. Verify the calibration workspace is clear.

The reviewed profile first moves to:

```text
joint_home =
[1.1242085695, -1.2143486899, 2.4404674212,
 -1.2285430890, 1.1332809925, 0.0]
```

It then visits 25 Cartesian poses in UR base coordinates:

- X: `-0.05 .. +0.05 m`
- Y: `-0.70 .. -0.50 m`
- Z: `0.15 .. 0.25 m`
- Tool speed: `0.10 m/s`
- Tool acceleration: `0.50 m/s^2`
- Joint speed/acceleration: `0.50 rad/s`, `0.50 rad/s^2`

Run receive-only checks before enabling motion:

```sh
python -c "from rtde_receive import RTDEReceiveInterface as R; r=R('172.16.90.197', use_upper_range_registers=False); print('robot_mode', r.getRobotMode()); print('safety_mode', r.getSafetyMode()); print('tcp', r.getActualTCPPose()); r.disconnect()"
```

Expected values are robot mode 7 and safety mode 1.

## Run calibration

Use two terminals.

### Terminal 1: Motive 2.0 pose publisher

Do not publish the previously accepted static transform while creating a
replacement:

```sh
cd /home/duo/ur_tools
source tools/activate_ros.sh
ur-optitrack-publisher \
  --server-ip 172.16.90.213 \
  --local-ip 172.16.90.195 \
  --target-name ur_calib \
  --no-static-tf
```

The publisher should report:

```text
Motive rigid bodies: ur_calib (id=...)
Connected to Motive 2.0.0.0 (NatNet 3.0.0.0)
```

Verify one finite pose:

```sh
ros2 topic echo --once /poses
```

The message must contain `name: ur_calib`, finite position values, and a
finite normalized quaternion.

### Terminal 2: review and execute

Dry run first; it cannot connect to ROS or move the robot:

```sh
cd /home/duo/ur_tools
source tools/activate_ros.sh
ur-optitrack-calibrate \
  --config config/optitrack_ur5e_197.yaml
```

After reviewing all poses and clearances:

```sh
ur-optitrack-calibrate \
  --config config/optitrack_ur5e_197.yaml \
  --execute
```

Enter the exact confirmation:

```text
MOVE UR5E 172.16.90.197
```

At each pose the collector:

1. Waits for the blocking `moveL` command to complete.
2. Allows the arm to settle.
3. Averages a 0.5-second Motive window.
4. Verifies at least 20 full-pose samples.
5. Rejects translation jitter above 2 mm or rotation jitter above 1 degree.
6. Reads the stationary RTDE TCP pose.
7. Atomically saves the completed sample.

On Ctrl+C or any failure, the collector requests `stopL` and `stopJ`,
disconnects RTDE, and preserves the incomplete dataset. Incomplete datasets can
never be promoted as accepted calibrations.

Stop Terminal 1 with Ctrl+C after collection and solving finish.

## Outputs and acceptance

Each execution creates:

```text
calibration_results/YYYYMMDDTHHMMSSZ/
  profile.yaml
  samples.yaml
  calibration.yaml              # accepted result
  calibration_candidate.yaml    # rejected result, when applicable
  metrics.json
  base_from_mocap.txt
  mocap_from_base.txt
  tcp_from_target.txt
```

Every fifth sample is held out during fitting. Acceptance requires:

- Translation RMS <= 5 mm
- Translation maximum <= 10 mm
- Rotation RMS <= 1 degree
- Rotation maximum <= 2 degrees

The current accepted lab calibration is:

```text
calibration_results/20260830T075837Z/calibration.yaml
```

Its held-out metrics are:

- Translation RMS: 0.309 mm
- Translation maximum: 0.399 mm
- Rotation RMS: 0.179 degrees
- Rotation maximum: 0.235 degrees

The accepted runtime matrix is stored in:

```text
config/optitrack_to_ur_base.yaml
```

The `source_result` field must identify the timestamped result from which the
runtime matrix was promoted. Run the full test suite after promotion:

```sh
source tools/activate_ros.sh
python -m pytest -q
uvx ruff check ur_tools/optitrack tests
```

## Offline solve

Re-run optimization without ROS, Motive, RTDE, or robot motion:

```sh
ur-optitrack-calibrate \
  --config config/optitrack_ur5e_197.yaml \
  --solve-only calibration_results/TIMESTAMP/samples.yaml
```

## Normal operation after calibration

After accepting and promoting a calibration:

1. Remove the marker board from the UR5e.
2. Disable or remove the `ur_calib` asset in Motive.
3. Do not run the calibration-only NatNet publisher for normal drone tracking.
4. Keep `config/optitrack_to_ur_base.yaml` as the permanent artifact.

Motive/Crazyswarm continues to track and control drones in its native `world`
frame. The runtime coordination layer must:

- Transform observed drone positions from `world` into UR `base`.
- Accept planner goals in `base`.
- Transform those goals back into `world` before calling Crazyswarm.

Do not send UR-base coordinates directly to the existing Crazyswarm
`go_to` service; it currently interprets goals in Motive `world`.

For visualization or a temporary integration test, the stored matrix can be
broadcast as:

```sh
ros2 run tf2_ros static_transform_publisher \
  --x -0.0710346707 --y -0.683186996 --z 0.00744266017 \
  --qx -0.0124699739 --qy 0.00622861175 \
  --qz 0.0151664772 --qw 0.999787819 \
  --frame-id base --child-frame-id world
```

This command intentionally remains running until Ctrl+C. A future runtime launch
should load the YAML automatically instead of duplicating these numbers.

## Troubleshooting

### No `/poses` messages

- Confirm `ur_calib` is enabled and Rigid Bodies streaming is on.
- Use `ur-optitrack-publisher`, not the installed
  `motion_capture_tracking` vendor backend, for Motive 2.0.
- Confirm server/local IP addresses and multicast settings.

### `ur_calib` disappears at one pose

- Inspect all four marker reconstructions in Motive at that exact arm pose.
- Check camera masks and reflective tape.
- Confirm each true marker meets the Minimum Ray Count.
- Improve board/camera visibility; do not proceed with an invalid rigid body.

The collector will retry three times and then stop at the current pose.

### `NatNetSDK Error 3`

The installed closed-source driver bundles NatNet SDK 4.1 and is incompatible
with this Motive 2.0 system. Use the project’s pure-Python
`ur-optitrack-publisher`, which negotiates NatNet 3.0 successfully.

### Publisher shutdown traceback

The project publisher guards against a final NatNet callback arriving after ROS
shutdown and should exit quietly on Ctrl+C. If another mocap process is still
running, stop the exact process before restarting to avoid duplicate multicast
consumers.
