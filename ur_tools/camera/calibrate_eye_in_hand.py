# Author: Baichuan Huang
import time
import os

import numpy as np
import cv2
from scipy.optimize import least_squares

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

from ur_tools.camera.utils import generate_6d_pose, calibrate_eye_hand, reprojection_error_eye_in_hand, get_board_pose

from ur_tools.camera.realsense_camera import Camera


def calibrate_eye_in_hand():
    # NOTE: make sure this is safe and correct ===============================
    tool_vel = 0.1
    tool_acc = 0.5
    joint_vel = 0.5
    joint_acc = 0.5
    _ip = "172.17.139.100"

    aruco_dict_board = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    # charuco_board = cv2.aruco.CharucoBoard((7, 5), 0.035, 0.02625, aruco_dict_board) # 7x5 charuco board with 35mm square and 26.25mm marker
    charuco_board = cv2.aruco.CharucoBoard((7, 5), 0.04, 0.03, aruco_dict_board) # 7x5 charuco board with 40mm square and 30mm marker
    if _ip == "172.17.139.100":
        joint_home = [1.305213451385498, 
                    -1.592827936212057, 
                    1.6703251043902796, 
                    -1.647313734094137, 
                    -1.5624845663653772, 
                    -0.17452842393984014
                    ]   # robot will go here before starting the calibration
    elif _ip == "172.17.139.103":
        joint_home = [
            1.3088655471801758, 
            -1.559130892423429, 
            1.6379039923297327, 
            -1.6532632313170375, 
            -1.5686739126788538, 
            -1.7453292519943295
        ]
    else:
        raise RuntimeError("Invalid IP address. Please check the IP address of the robot.")

    target = [0, -0.5, 0]               # rough position of the target (calibration board)
    y_limits = [-0.6, -0.3]             # y limits for the end-effector
    x_limits = [-0.1, 0.1]              # x limits for the end-effector
    z_heights = [0.3, 0.35, 0.4]        # randomly pick z from this list
    up_axis = [0, -1, 0]                # camera up axis when looking at the target #Duo: This does not mean the direction of the lens. 
                                        # It literally means the up axis of a robot. If you put the camera on the table while the lens is facing you,
                                        # the up axis now is [0, 0, 1]
    offset_rpy = [0, np.pi, 0]          # adjust the rpy of the camera to match the robot's coordinate system
    randomness_up = 0.2                 # add randomness in the up axis
    x_num, y_num = 5, 5                 # sample points for the end-effector wihin the limits
    initial_guess_cam2ee = None         # provide this if calibrateHandEye is not good
    # =========================================================================

    camera = Camera(calibrated=False)

    # Connect robot
    rtde_c = RTDEControl(_ip)
    rtde_r = RTDEReceive(_ip, use_upper_range_registers=False)

    # aruco_params = cv2.aruco.DetectorParameters()
    # aruco_detector_board = cv2.aruco.ArucoDetector(aruco_dict_board, aruco_params)
    charuco_detector = cv2.aruco.CharucoDetector(charuco_board)

    # prepare ee configs
    x_values = np.linspace(x_limits[0], x_limits[1], x_num)
    y_values = np.linspace(y_limits[0], y_limits[1], y_num)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    calibration_ee_configs = []
    for x, y in zip(x_grid.flatten(), y_grid.flatten()):
        # select z randomly, and rpy randomly
        z_i = np.random.randint(0, len(z_heights))
        # add random noise to up_axis
        random_up_axis = np.array(up_axis) + np.random.uniform(-randomness_up, randomness_up, 3)
        random_up_axis = random_up_axis / np.linalg.norm(random_up_axis)
        pose = generate_6d_pose([x, y, z_heights[z_i]], target, random_up_axis, offset_rpy)
        print(f"pose = {pose}")
        calibration_ee_configs.append(pose)

    # Lists to store poses
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    # Collect marker pos
    # input("\nCollecting marker pos...\nPress Enter to continue...")
    rtde_c.moveJ(joint_home, joint_vel, joint_acc)

    for i, ee_config in enumerate(calibration_ee_configs):
        rtde_c.moveL(ee_config, tool_vel, tool_acc)
        print(f"Moving to {ee_config}...")
        time.sleep(0.5)

        # prepare marker pose
        color_img, depth_img = camera.get_data()
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        rvec, tvec, marker_pos, color_img = get_board_pose(color_img, charuco_detector, charuco_board, camera.intrinsics, camera.distortion_coeffs, plot=True)

        if rvec is not None and tvec is not None:
            # Store target (marker) to camera transformation
            R_target2cam.append(cv2.Rodrigues(rvec)[0])
            t_target2cam.append(tvec)

            # Get robot pose (gripper to base)
            robot_pose = np.array(rtde_r.getActualTCPPose())
            R_gripper2base.append(cv2.Rodrigues(robot_pose[3:])[0])
            t_gripper2base.append(robot_pose[:3].reshape(3, 1))
        else:
            print(f"Marker not detected. Skipping this pose {ee_config}...")

        cv2.imshow("frame color", color_img)
        cv2.imshow("frame depth", depth_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # input("waiting for next marker...")
    cv2.destroyAllWindows()
    print(f"Number of marker poses: {len(R_gripper2base)}")

    # Convert lists to numpy arrays
    R_gripper2base = np.array(R_gripper2base)
    t_gripper2base = np.array(t_gripper2base)
    R_target2cam = np.array(R_target2cam)
    t_target2cam = np.array(t_target2cam)

    # Prepare data for optimization
    ee_poses = np.zeros((len(R_gripper2base), 4, 4))
    ee_poses[:, :3, :3] = R_gripper2base
    ee_poses[:, :3, 3] = t_gripper2base.squeeze()
    ee_poses[:, 3, 3] = 1
    marker_poses = np.zeros((len(R_target2cam), 4, 4))
    marker_poses[:, :3, :3] = R_target2cam
    marker_poses[:, :3, 3] = t_target2cam.squeeze()
    marker_poses[:, 3, 3] = 1

    # Perform hand-eye calibration
    print("Performing hand-eye calibration...")
    results = calibrate_eye_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=False)
    R_cam, T_cam = results["PARK"]
    cam2ee_pose = np.eye(4)
    cam2ee_pose[:3, :3] = R_cam
    cam2ee_pose[:3, 3] = T_cam.flatten()
    print(cam2ee_pose)



    # Initial guess for optimization
    initial_guess = np.zeros(12)  # 6 for cam2ee pose, 6 for marker2base pose
    if initial_guess_cam2ee is None:
        initial_guess[:6] = np.concatenate([cv2.Rodrigues(R_cam)[0].flatten(), T_cam.flatten()])
    else:
        initial_guess[:6] = initial_guess_cam2ee
    # For marker2base, use the first observation as initial guess
    initial_marker2base = ee_poses[0] @ cam2ee_pose @ np.linalg.inv(marker_poses[0])
    initial_guess[6:9] = cv2.Rodrigues(initial_marker2base[:3, :3])[0].flatten()
    initial_guess[9:] = initial_marker2base[:3, 3]

    # Perform optimization
    print("Optimizing camera pose and marker offset...")
    result = least_squares(reprojection_error_eye_in_hand, initial_guess, args=(ee_poses, marker_poses), method='lm')

    # Extract optimized values
    optimized_params = result.x
    optimized_cam2ee_pose_params = optimized_params[:6]
    optimized_marker2base = optimized_params[6:]
    optimized_cam2ee_pose = np.eye(4)
    optimized_cam2ee_pose[:3, :3] = cv2.Rodrigues(optimized_cam2ee_pose_params[:3])[0]
    optimized_cam2ee_pose[:3, 3] = optimized_cam2ee_pose_params[3:]
    print("Optimized camera to end-effector pose:")
    print(optimized_cam2ee_pose)
    print("Optimized marker offset:")
    print(optimized_marker2base)
    print("Done.")

    # Calculate and display error metrics
    reprojection_errors = np.reshape(reprojection_error_eye_in_hand(optimized_params, ee_poses, marker_poses), (-1, 3))
    print("\nCalibration Error Metrics (in robot base frame):")
    print(f"Mean Reprojection Error: {np.mean(np.linalg.norm(reprojection_errors, axis=1)):.6f} m")
    print(f"Max Reprojection Error: {np.max(np.linalg.norm(reprojection_errors, axis=1)):.6f} m")

    # Calculate and display error metrics for each method from hand-eye calibration
    for result in results:
        R_cam, T_cam = results[result]
        optimized_params = np.zeros(12)
        optimized_params[:6] = np.concatenate([cv2.Rodrigues(R_cam)[0].flatten(), T_cam.flatten()])
        optimized_params[6:] = optimized_marker2base
        reprojection_errors = np.reshape(reprojection_error_eye_in_hand(optimized_params, ee_poses, marker_poses), (-1, 3))
        print(f"\nCalibration Error Metrics ({result} method):")
        print(f"Mean Reprojection Error: {np.mean(np.linalg.norm(reprojection_errors, axis=1)):.6f} m")
        print(f"Max Reprojection Error: {np.max(np.linalg.norm(reprojection_errors, axis=1)):.6f} m")

    print("Saving...")
    saving_path = os.path.join(os.path.dirname(__file__), "cam2ee_pose.txt")
    np.savetxt(saving_path, optimized_cam2ee_pose, delimiter=" ")
    return optimized_cam2ee_pose

if __name__ == "__main__":
    calibrate_eye_in_hand()
