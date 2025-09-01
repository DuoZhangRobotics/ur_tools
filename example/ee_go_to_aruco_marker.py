import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from ur_tools.camera.realsense_camera import Camera
from ur_tools.ur_control.ur5_robotiq_2f_85_control import UR_Robotiq_2f_85_Controller 
import time

def yaw_only_projection(T_b1_b2):
    R = T_b1_b2[:3,:3]
    # Nearest Rz(theta) in least-squares sense for the top-left 2x2 block
    c = R[0,0] + R[1,1]
    s = R[1,0] - R[0,1]
    theta = np.arctan2(s, c)
    theta=180/180*np.pi
    cth, sth = np.cos(theta), np.sin(theta)
    Rz = np.array([[cth, -sth, 0],
                   [sth,  cth, 0],
                   [  0,    0, 1]])
    Tp = np.eye(4)
    Tp[:3,:3] = Rz
    Tp[:3,3]  = T_b1_b2[:3,3]   # zero z if same table height
    Tp[2,3] = 0
    return Tp, theta


def measure_marker_world_pos(camera: Camera, detector, marker_id: int, num_samples: int = 10, sleep_s: float = 0.02):
    """Average world position of a specific ArUco marker id over multiple frames.

    Args:
        camera: Camera instance providing get_data and projection utilities.
        detector: cv2.aruco.ArucoDetector initialized with the right dictionary.
        marker_id: The ArUco ID to sample.
        num_samples: Target number of valid samples to average.
        sleep_s: Small delay between frames.

    Returns:
        (avg: np.ndarray shape (3,), samples: np.ndarray shape (M,3))
        avg is None if no valid samples were collected.
    """
    collected = []
    attempts = 0
    max_attempts = num_samples * 5  # allow some misses
    while len(collected) < num_samples and attempts < max_attempts:
        attempts += 1
        color, depth = camera.get_data()
        if color is None or depth is None:
            time.sleep(sleep_s)
            continue
        try:
            gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        except Exception:
            # Fallback if color ordering differs
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            time.sleep(sleep_s)
            continue
        ids_flat = ids.flatten()
        matches = np.where(ids_flat == marker_id)[0]
        if len(matches) == 0:
            time.sleep(sleep_s)
            continue
        # Use the first match
        corner = corners[int(matches[0])]  # shape (1,4,2)
        center = np.mean(corner, axis=1).reshape(-1)  # (2,)
        x, y = float(center[0]), float(center[1])
        yi, xi = int(round(y)), int(round(x))
        if yi < 0 or xi < 0 or yi >= depth.shape[0] or xi >= depth.shape[1]:
            time.sleep(sleep_s)
            continue
        z = float(depth[yi, xi])
        if not np.isfinite(z) or z <= 0:
            time.sleep(sleep_s)
            continue

        wp = camera.pos_in_image_to_robot(np.array([[x, y]]), np.array([z]))[0]
        collected.append(wp)
        time.sleep(sleep_s)

    if len(collected) == 0:
        return None, np.empty((0, 3), dtype=float)
    samples = np.asarray(collected, dtype=float)
    avg = samples.mean(axis=0)
    return avg, samples

def debug_main_arm():
    dt = 0.002
    robot = UR_Robotiq_2f_85_Controller(
        dt=dt, robot_ip="172.17.139.100", init_gripper=True)
    init_q = robot.get_joint_state()
    robot.moveJ(
        [1.5708118677139282, -2.2, 1.9, -1.383, -1.5700505415545862, 0], speed=0.3, acceleration=0.3
    )
    init_ee_pos = robot.get_ee_state()
    print("Initial end-effector position:\n", init_ee_pos)
    robot.open_gripper()
    camera = Camera(on_hand=False)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    camera.show_video_realtime()
    colored_image, depth_image = camera.get_data()

    # Convert to grayscale for ArUco detection
    gray = cv2.cvtColor(colored_image, cv2.COLOR_RGB2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        print("No markers found.")
        return

    for marker_id in ids.flatten():
        avg_pos, samples = measure_marker_world_pos(camera, detector, int(marker_id), num_samples=10)
        if avg_pos is None:
            print(f"Could not average marker {marker_id} (no valid samples).")
            continue
        print(f"Marker {marker_id}: averaged world_pos over {len(samples)} samples -> {avg_pos}")

        input("Press Enter to Move EE to averaged marker position")
        pos = [avg_pos[0], avg_pos[1], avg_pos[2]/2, 0, -np.pi, 0]
        robot.moveL(pos, speed=0.3, acceleration=0.3)
        input("Press Enter to close gripper")
        robot.close_gripper()
        input("Press Enter to open gripper")
        robot.open_gripper()
        input("Press Enter to go home")
        robot.moveJ(
            [1.5708118677139282, -2.2, 1.9, -1.383, -1.5700505415545862, 0], speed=0.3, acceleration=0.3
        )

def make_base2base(x, y, theta):
    base2base = np.eye(4)
    base2base[:3, 3] = [x, y, 0]
    rot = R.from_euler('z', theta, degrees=True)
    base2base[:3, :3] = rot.as_matrix()
    return base2base

def debug_second_arm():
    dt = 0.002
    robot = UR_Robotiq_2f_85_Controller(
        dt=dt, robot_ip="172.17.139.103", init_gripper=True)
    init_ee_pos = robot.get_ee_state()
    robot.moveJ(
        [1.5708118677139282, -2.2, 1.9, -1.383, -1.5700505415545862, 0], speed=0.3, acceleration=0.3
    )
    print("Initial end-effector position:\n", init_ee_pos)
    robot.open_gripper()
    camera = Camera(on_hand=False)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    camera.show_video_realtime()
    colored_image, depth_image = camera.get_data()

    # Convert to grayscale for ArUco detection
    gray = cv2.cvtColor(colored_image, cv2.COLOR_RGB2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)
    base2base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ur_tools/ur_control", "base2base.txt")
    base2base = np.loadtxt(base2base_path, delimiter=" ")

    base2base, theta = yaw_only_projection(base2base)
    print("base2base:\n", base2base)
    print("Yaw angle (degrees): ", theta / np.pi * 180)

    cam2base_arm2_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ur_tools/camera", "cam2base_pose_arm2.txt")
    cam2base_arm2 = np.loadtxt(cam2base_arm2_path, delimiter=" ")
    cam2base_arm1 = camera.cam2robot
    arm1_z_in_cam = np.linalg.inv(cam2base_arm1[:3, :3]) @ np.array([[0], [0], [1]])
    arm2_z_in_cam = np.linalg.inv(cam2base_arm2[:3, :3]) @ np.array([[0], [0], [1]])
    print(f"Arm 1 Z in camera: {arm1_z_in_cam}")
    print(f"Arm 2 Z in camera: {arm2_z_in_cam}")
    dot_product = np.dot(arm1_z_in_cam.reshape(-1), arm2_z_in_cam.reshape(-1)) / (np.linalg.norm(arm1_z_in_cam) * np.linalg.norm(arm2_z_in_cam))
    angle_z_arm1_arm2 = np.arccos(dot_product) * 180 / np.pi
    print(f"Angle in degrees between Arm 1 and Arm 2 in camera: {angle_z_arm1_arm2}")
    rot = R.from_matrix(base2base[:3, :3])
    rpy_xyz = rot.as_euler('xyz', degrees=True)  # roll(x), pitch(y), yaw(z)
    rotvec = rot.as_rotvec()
    theta = np.linalg.norm(rotvec)
    axis = rotvec / theta if theta > 1e-12 else np.array([1, 0, 0])
    print(f"Rotation axis: {axis}, Angle: {theta}")
    print(f"RPY (degrees): {rpy_xyz}")
    # np.savetxt(base2base_path, base2base)

    if ids is None or len(ids) == 0:
        print("No markers found.")
        return

    for marker_id in ids.flatten():
        avg_pos, samples = measure_marker_world_pos(camera, detector, int(marker_id), num_samples=10)
        if avg_pos is None:
            print(f"Could not average marker {marker_id} (no valid samples).")
            continue

        print(f"Marker {marker_id}: averaged world_pos over {len(samples)} samples -> {avg_pos}")
        # Optional manual adjustment (existing logic)
        # world_pos = (-1 * avg_pos[0], -1.095 - avg_pos[1], avg_pos[2])
        base2base=make_base2base(0, -1.098, 178)
        world_pos = base2base @ np.array([avg_pos[0], avg_pos[1], avg_pos[2], 1])
        print(f"Transformed world coordinates: {world_pos}")

        input("Press Enter to Move EE to averaged marker position")
        pos = [world_pos[0], world_pos[1], world_pos[2]/2, 0, -np.pi, 0]
        robot.moveL(pos, speed=0.3, acceleration=0.3)
        input("Press Enter to close gripper")
        robot.close_gripper()
        input("Press Enter to open gripper")
        robot.open_gripper()
        input("Press Enter to go home")
        robot.moveJ(
            [1.5708118677139282, -2.2, 1.9, -1.383, -1.5700505415545862, 0], speed=0.3, acceleration=0.3
        )

        

if __name__ == "__main__":
    # print(3.109430060929261/np.pi*180)
    # print(178.17605049/180 * np.pi)
    # # convert 178.17605049 around [0, 0, 1] to rotation matrix
    # rot = R.from_euler('xyz', [0, 0, 178.17605049/180 * np.pi])
    # rot_matrix = rot.as_matrix()
    # print(f"Rotation matrix:\n{rot_matrix}")
    # debug_main_arm()
    debug_second_arm()