# Author: Baichuan Huang
import time
import os

import numpy as np
import cv2
from scipy.optimize import least_squares

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

from ur_tools.camera.utils import *

from ur_tools.camera.realsense_camera import Camera


def calibrate_eye_to_hand(ip_address="172.17.139.103", camera=None, write_pose_to_file:bool=True, x_num=5, y_num=5):
    # NOTE: make sure this is safe and correct ===============================
    tool_vel = 0.1
    tool_acc = 0.5
    joint_vel = 0.5
    joint_acc = 0.5
    if ip_address not in [
        "172.17.139.100",
        "172.17.139.103"
    ]:
        raise RuntimeError("Invalid IP address. Please check the IP address of the robot.")

    _ip = ip_address

    # aruco_dict_board = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    # charuco_board = cv2.aruco.CharucoBoard((4, 4), 0.025, 0.01875, aruco_dict_board) # 4x4 charuco board with 30mm square and 22.5mm marker
    aruco_dict_board = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    charuco_board = cv2.aruco.CharucoBoard((7, 5), 0.04, 0.03, aruco_dict_board) # 7x5 charuco board with 40mm square and 30mm marker
    charuco_board.setLegacyPattern(True)  # the board use the old pattern, remove this if the board is created with the new pattern
    joint_home = [1.1242085695266724, 
                  -1.2143486899188538, 
                  2.440467421208517, 
                  -1.2285430890372773, 
                  1.1332809925079346, 
                  0]                    # robot will go here before starting the calibration
    
    y_limits = [-0.5, -0.6]             # y limits for the end-effector
    x_limits = [-0.2, 0.2]              # x limits for the end-effector
    z_heights = [0.08, 0.12, 0.18]      # randomly pick z from this list
    rpy = [[1.57, 0, 0],
           [1.56, 0.2, 0.2],
           [1.56, -0.2, -0.2],
           [1.56, -0.2, 0.2],
           [1.56, 0.2, -0.2],
           [1.76, 0, 0],
           [1.36, 0, 0]]                # randomly pick rpy from this list
    initial_guess_cam2base = None       # provide this if calibrateHandEye is not good
    
    # =========================================================================

    if camera is None:
        camera = Camera(calibrated=False, on_hand=False)

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
    # have the base layer to use the same z and rpy
    calibration_ee_configs = [[x, y, z_heights[0]] + rpy[0] for x, y in zip(x_grid.flatten(), y_grid.flatten())]
    for x, y in zip(x_grid.flatten(), y_grid.flatten()):
        # select z randomly, and rpy randomly
        z_i = np.random.randint(1, len(z_heights))
        rpy_i = np.random.randint(1, len(rpy))
        calibration_ee_configs.append([x, y, z_heights[z_i]] + rpy[rpy_i])
    
    # rand_indx = np.random.choice(len(calibration_ee_configs), 6, replace=False)
    # calibration_ee_configs = [calibration_ee_configs[i] for i in rand_indx]

    # Lists to store poses
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    # Collect marker pos
    input("\nCollecting marker pos...\nPress Enter to continue...")
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
    results = calibrate_eye_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True)
    R_cam, T_cam = results["PARK"]
    cam2base_pose = np.eye(4)
    cam2base_pose[:3, :3] = R_cam
    cam2base_pose[:3, 3] = T_cam.flatten()
    print(cam2base_pose)



    # Initial guess for optimization
    initial_guess = np.zeros(12)  # 6 for cam2base pose, 6 for marker2ee pose
    if initial_guess_cam2base is None:
        initial_guess[:6] = np.concatenate([cv2.Rodrigues(R_cam)[0].flatten(), T_cam.flatten()])
    else:
        initial_guess[:6] = initial_guess_cam2base
    # For marker2ee, use the first observation as initial guess
    initial_marker2ee = np.linalg.inv(ee_poses[0]) @ cam2base_pose @ marker_poses[0]
    initial_guess[6:9] = cv2.Rodrigues(initial_marker2ee[:3, :3])[0].flatten()
    initial_guess[9:] = initial_marker2ee[:3, 3]

    # Perform optimization
    print("Optimizing camera pose and marker offset...")
    result = least_squares(reprojection_error_eye_to_hand, initial_guess, args=(ee_poses, marker_poses), method="lm")

    # Extract optimized values
    optimized_params = result.x
    optimized_camera_pose_params = optimized_params[:6]
    optimized_marker2ee_params = optimized_params[6:]
    optimized_camera_pose = np.eye(4)
    optimized_camera_pose[:3, :3] = cv2.Rodrigues(optimized_camera_pose_params[:3])[0]
    optimized_camera_pose[:3, 3] = optimized_camera_pose_params[3:]
    print("Optimized camera to base pose:")
    print(optimized_camera_pose)
    print("Optimized marker offset:")
    print(optimized_marker2ee_params)
    print("Done.")

    # Calculate and display error metrics
    reprojection_errors = np.reshape(reprojection_error_eye_to_hand(optimized_params, ee_poses, marker_poses), (-1, 3))
    print("\nCalibration Error Metrics (in robot base frame):")
    print(f"Mean Reprojection Error: {np.mean(np.linalg.norm(reprojection_errors, axis=1)):.6f} m")
    print(f"Max Reprojection Error: {np.max(np.linalg.norm(reprojection_errors, axis=1)):.6f} m")

    # Calculate and display error metrics for each method from hand-eye calibration
    for result in results:
        R_cam, T_cam = results[result]
        optimized_params = np.zeros(12)
        optimized_params[:6] = np.concatenate([cv2.Rodrigues(R_cam)[0].flatten(), T_cam.flatten()])
        optimized_params[6:] = optimized_marker2ee_params
        reprojection_errors = np.reshape(reprojection_error_eye_to_hand(optimized_params, ee_poses, marker_poses), (-1, 3))
        print(f"\nCalibration Error Metrics ({result} method):")
        print(f"Mean Reprojection Error: {np.mean(np.linalg.norm(reprojection_errors, axis=1)):.6f} m")
        print(f"Max Reprojection Error: {np.max(np.linalg.norm(reprojection_errors, axis=1)):.6f} m")

    camera_rotation = cv2.Rodrigues(optimized_camera_pose_params[:3])[0]
    camera_translation = optimized_camera_pose_params[3:]
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = camera_rotation
    camera_pose[:3, 3] = camera_translation

    if write_pose_to_file:
        print("Saving...")
        saving_path = os.path.join(os.path.dirname(__file__), "cam2base_pose.txt")
        np.savetxt(saving_path, camera_pose, delimiter=" ")
    return camera_pose 

if __name__ == "__main__":
    ip = "172.17.139.100"
    calibrate_eye_to_hand(ip_address=ip)