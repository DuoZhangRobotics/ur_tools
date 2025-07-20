# Author: Baichuan Huang
import time

import numpy as np
import cv2
from scipy.optimize import least_squares

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

from utils import *

from realsense_camera import Camera


if __name__ == "__main__":

    # NOTE: make sure this is safe and correct ===============================
    tool_vel = 0.1
    tool_acc = 0.5
    joint_vel = 0.5
    joint_acc = 0.5
    _ip = "172.17.139.103"

    aruco_dict_board = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    charuco_board = cv2.aruco.CharucoBoard((4, 4), 0.025, 0.01875, aruco_dict_board) # 4x4 charuco board with 30mm square and 22.5mm marker
    charuco_board.setLegacyPattern(True)  # the board use the old pattern, remove this if the board is created with the new pattern
    joint_home = [1.1242085695266724, 
                  -1.2143486899188538, 
                  2.440467421208517, 
                  -1.2285430890372773, 
                  1.1332809925079346, 
                  0]                    # robot will go here before starting the calibration
    joint_home = [52.11*3.14/180, 
                  -130*3.14/180, 
                  102*3.14/180, 
                  -89*3.14/180, 
                  -71*3.14/180, 
                  148*3.14/180]         # robot will go here before starting the calibration
    
    y_limits = [-0.9, -0.5]             # y limits for the end-effector
    x_limits = [-0.2, 0.2]              # x limits for the end-effector
    z_heights = [0.08, 0.12, 0.18]      # randomly pick z from this list
    rpy = [[1.57, 0, 0],
           [1.56, 0.2, 0.2],
           [1.56, -0.2, -0.2],
           [1.56, -0.2, 0.2],
           [1.56, 0.2, -0.2],
           [1.76, 0, 0],
           [1.36, 0, 0]]                # randomly pick rpy from this list
    x_num, y_num = 4, 4                 # sample points for the end-effector wihin the limits
    initial_guess_cam2base = None       # provide this if calibrateHandEye is not good
    
    # =========================================================================

    camera = Camera()

    # Connect robot
    rtde_c = RTDEControl(_ip)
    rtde_r = RTDEReceive(_ip, use_upper_range_registers=False)

    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector_board = cv2.aruco.ArucoDetector(aruco_dict_board, aruco_params)
    
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
    
    # Waste:    #   [0.98757, -0.19770, 0.41266, 1.823, 1.625, 0.932] - 1
    #   [0.98757, -0.19770, 0.41266, 1.823, 1.625, 0.932] - 5
    #   [0.92030, -0.46118, 0.52680, 1.584, 1.197, 0.865] - 6
    # [INFO@KOWNDI]: side-view-hanging-ext-cam-until Jan 15, 2025 - Working config
    # calibration_ee_configs = [
    #                           [0.92030, -0.46118, 0.52680, 1.584, 1.197, 0.865],
    #                           [0.98658, -0.37980, 0.49076, 1.432, 1.241, 0.943],
    #                           [0.79595, -0.67713, 0.52630, 1.068, 1.531, 0.114],
    #                           [0.67751, -0.73273, 0.62963, 1.388, 1.103, 0.347],
    #                           [0.84195, -0.52638, 0.62341, 1.604, 0.928, 0.935]
    #                           ]
    # calibration_ee_configs = [ 
    #     # [-0.22353, -0.97908, 0.55591, 1.885, -0.1, -0.087],
    #     [-0.02528, -0.87823, 0.67080, 1.916, -0.944, 0.536],
    #     [-0.19785, -1.00760, 0.53068, 1.849, -0.434, -0.040],
    #     [-0.23840, -1.0086, 0.57720, 1.673, -0.467, 0.306],
    #     [-0.13769, -0.96116, 0.54246, 1.577, -0.524, 0.532],
    #     [-0.12707, -0.96083, 0.59375, 1.934, -0.611, 0.725]
    # ]
    calibration_ee_configs = [
        [-0.00864, -0.35145, 0.46070, 2.528, -0.516, -0.144], # 8
        [-0.00864, -0.49469, 0.48178, 2.528, -0.516, -0.144], # 8
        [-0.00864, -0.56910, 0.48178, 2.446, -0.861, -0.245], # 7
        [-0.27640, -0.56913, 0.48175, 2.539, -0.733, -0.241], # 8
        [-0.27640, -0.49158, 0.48175, 2.471, -0.657, -0.480], # 8
        [-0.09620, -0.34092, 0.48178, 2.471, -0.657, -0.480], # 8
    ]
    rand_indx = np.random.choice(len(calibration_ee_configs), 6, replace=False)
    print('random sequence:', rand_indx)
    calibration_ee_configs = [calibration_ee_configs[i] for i in rand_indx]
    # rand_idx = np.random.choice(len(calibration_ee_configs), 4, replace=False)
    # calibration_ee_configs = [calibration_ee_configs[i] for i in rand_idx]
    # # Load camera pose and marker offset
    # rvec = cv2.Rodrigues(camera.camera_pose[:3, :3])[0].flatten().tolist()
    # tvec = camera.camera_pose[:3, 3]
    # camera_pose_esitimate = [*rvec, *tvec]
    # marker_offset_esitimate = np.loadtxt("marker_offset.txt", delimiter=" ")

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
        rvec, tvec, marker_pos, color_img = get_board_pose(color_img, aruco_detector_board, charuco_board, camera.intrinsics, camera.distortion_coeffs, plot=True)

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

    print("Saving...")
    np.savetxt("kowndi_camera_pose_rand1_jan15.txt", optimized_camera_pose, delimiter=" ")