import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from ur_tools.camera.realsense_camera import Camera
from ur_tools.ur_control.ur5_robotiq_2f_85_control import UR_Robotiq_2f_85_Controller 

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
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    camera.show_video_realtime()
    colored_image, depth_image = camera.get_data()

    # Convert to grayscale for ArUco detection
    gray = cv2.cvtColor(colored_image, cv2.COLOR_RGB2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    center = np.mean(corners[0], axis=1)
    print(f"Center of detected marker: {center}")
    x, y = center[0][0], center[0][1]
    z = depth_image[int(y), int(x)]
    print(f"Depth of detected marker: {z}")
    x_cam = (x - camera.intrinsics[0, 2]) * z / camera.intrinsics[0, 0]
    y_cam = (y - camera.intrinsics[1, 2]) * z / camera.intrinsics[1, 1]
    print(f"Camera coordinates: ({x_cam}, {y_cam}, {z})")
    world_pos = camera.pos_in_camera_to_ee(np.array([[x_cam, y_cam, z]]))
    print(f"World coordinates: {world_pos}")

    world_pos = camera.pos_in_image_to_robot(np.array([[x, y]]), np.array([z]))[0]
    print(f"World coordinates: {world_pos}")

    input("Press Enter to Move EE to marker")

    pos = [world_pos[0], world_pos[1], world_pos[2]/2, 0, -np.pi, 0]
    robot.moveL(pos, speed=0.3, acceleration=0.3)
    input("Press Enter to close gripper")
    robot.close_gripper()
    input("Press Enter to open gripper")
    robot.open_gripper()

def debug_second_arm():
    dt = 0.002
    robot = UR_Robotiq_2f_85_Controller(
        dt=dt, robot_ip="172.17.139.103", init_gripper=True)
    init_q = robot.get_joint_state()
    robot.moveJ(
        [1.5708118677139282, -2.2, 1.9, -1.383, -1.5700505415545862, 0], speed=0.3, acceleration=0.3
    )
    init_ee_pos = robot.get_ee_state()
    print("Initial end-effector position:\n", init_ee_pos)
    robot.open_gripper()
    camera = Camera(on_hand=False)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    camera.show_video_realtime()
    colored_image, depth_image = camera.get_data()

    # Convert to grayscale for ArUco detection
    gray = cv2.cvtColor(colored_image, cv2.COLOR_RGB2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    center = np.mean(corners[0], axis=1)
    print(f"Center of detected marker: {center}")
    x, y = center[0][0], center[0][1]
    z = depth_image[int(y), int(x)]
    print(f"Depth of detected marker: {z}")
    x_cam = (x - camera.intrinsics[0, 2]) * z / camera.intrinsics[0, 0]
    y_cam = (y - camera.intrinsics[1, 2]) * z / camera.intrinsics[1, 1]
    print(f"Camera coordinates: ({x_cam}, {y_cam}, {z})")
    world_pos = camera.pos_in_camera_to_ee(np.array([[x_cam, y_cam, z]]))
    print(f"World coordinates: {world_pos}")

    world_pos = camera.pos_in_image_to_robot(np.array([[x, y]]), np.array([z]))[0]
    print(f"World coordinates: {world_pos}")
    
    base2base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ur_tools/ur_control", "base2base.txt")
    base2base = np.loadtxt(base2base_path, delimiter=" ")
    rot = R.from_matrix(base2base[:3, :3])
    rotvec = rot.as_rotvec()
    theta = np.linalg.norm(rotvec)
    axis = rotvec / theta if theta > 1e-12 else np.array([1, 0, 0])
    print(f"Rotation axis: {axis}, Angle: {theta}")
    # base2base[:, 3] = np.array([-0.0, -1.097, 0, 1])
    world_pos = base2base @ np.array([[world_pos[0]], [world_pos[1]], [world_pos[2]], [1]])
    world_pos = world_pos.flatten()[:3]
    print(f"Transformed world coordinates: {world_pos}")

    input("Press Enter to Move EE to marker")

    pos = [world_pos[0], world_pos[1], world_pos[2]/2, 0, -np.pi, 0]
    robot.moveL(pos, speed=0.3, acceleration=0.3)
    input("Press Enter to close gripper")
    robot.close_gripper()
    input("Press Enter to open gripper")
    robot.open_gripper()
    

if __name__ == "__main__":
    # debug_main_arm()
    debug_second_arm()