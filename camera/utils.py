# Author: Baichuan Huang

import numpy as np
import cv2

def generate_6d_pose(eye, target, up, offset_RPY):
    # Generate 6D pose from eye and target, with up vector, looking at target from eye
    # eye: 3D position of the camera
    # target: 3D position of the target
    # up: 3D vector pointing upwards
    # offset_RPY: 3D vector of roll, pitch, yaw adjustments

    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)
    offset_RPY = np.array(offset_RPY)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    
    side = np.cross(up, forward)
    side /= np.linalg.norm(side)

    up = np.cross(side, forward)
    up /= np.linalg.norm(up)

    look_at_R = np.array([side, up, -forward]).T

    # Adjust the rotation matrix by the offset_RPY
    R = look_at_R @ cv2.Rodrigues(offset_RPY)[0]

    # Return the 6D pose [x, y, z, roll, pitch, yaw]
    return np.concatenate([eye, cv2.Rodrigues(R)[0].flatten()])

def calibrate_eye_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True):
    if eye_to_hand:
        # For eye-to-hand, we need base2gripper instead of gripper2base
        R_base2gripper, t_base2gripper = [], []
        for R, t in zip(R_gripper2base, t_gripper2base):
            # Invert the transformation
            R_b2g = R.T  # Transpose for rotation matrix inversion
            t_b2g = -R_b2g @ t  # Transform the translation vector
            R_base2gripper.append(R_b2g)
            t_base2gripper.append(t_b2g)
        
        # Update the variables for the calibration function
        R_gripper2base = R_base2gripper
        t_gripper2base = t_base2gripper

    results = {}
    
    # Calibrate using OpenCV's function
    R, t = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,  # Now contains base2gripper for eye-to-hand
        t_gripper2base=t_gripper2base,  # Now contains base2gripper for eye-to-hand
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    results["TSAI"] = (R, t)

    R, t = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,  # Now contains base2gripper for eye-to-hand
        t_gripper2base=t_gripper2base,  # Now contains base2gripper for eye-to-hand
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    results["PARK"] = (R, t)

    R, t = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,  # Now contains base2gripper for eye-to-hand
        t_gripper2base=t_gripper2base,  # Now contains base2gripper for eye-to-hand
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_HORAUD,
    )
    results["HORAUD"] = (R, t)

    R, t = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,  # Now contains base2gripper for eye-to-hand
        t_gripper2base=t_gripper2base,  # Now contains base2gripper for eye-to-hand
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_ANDREFF,
    )
    results["ANDREFF"] = (R, t)

    R, t = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,  # Now contains base2gripper for eye-to-hand
        t_gripper2base=t_gripper2base,  # Now contains base2gripper for eye-to-hand
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS,
    )
    results["DANIILIDIS"] = (R, t)
    
    return results

def get_board_pose(color_img, aruco_detector, charuco_board, camera_matrix, dist_coeffs, plot=False):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected_img_points = aruco_detector.detectMarkers(gray)
    number_markers = (charuco_board.getChessboardSize()[0] - 1) * (charuco_board.getChessboardSize()[1] - 1)
    print("Number of IDs: ", len(ids), "number of markers: ", number_markers)

    if len(corners) > number_markers / 2:
        # Refine detected markers
        corners, ids, rejected, recovered = aruco_detector.refineDetectedMarkers(gray, charuco_board, corners, ids, rejected_img_points)

        # Interpolate charuco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, charuco_board
        )        

        if charuco_corners is not None and charuco_ids is not None:
            # Draw corners and ids
            if plot:
                color_img = cv2.aruco.drawDetectedCornersCharuco(color_img, charuco_corners, charuco_ids)

            # Estimate pose
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, charuco_board, camera_matrix, dist_coeffs, np.empty(1), np.empty(1)
            )

            if retval:
                # Draw axis
                if plot:
                    cv2.drawFrameAxes(color_img, camera_matrix, dist_coeffs, rvec, tvec, 0.1, thickness=1)

                board_origin_3d = np.array([[0., 0., 0.]])
                board_origin_pixel, _ = cv2.projectPoints(board_origin_3d, rvec, tvec, camera_matrix, dist_coeffs)
                board_origin_pixel = board_origin_pixel.ravel()

                return rvec, tvec, board_origin_pixel, color_img
        
    return None, None, None, color_img

def get_marker_pose(color_img, aruco_detector, marker_id, marker_length, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected_img_points = aruco_detector.detectMarkers(gray)

    if ids is None or marker_id not in ids:
        print(f"Marker {marker_id} not found. Detected IDs: {ids}")
        return None, None, None

    marker_index = np.where(ids == marker_id)[0][0]
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners[marker_index], 
        marker_length, 
        camera_matrix, 
        dist_coeffs
    )

    rvec = rvecs[0]
    tvec = tvecs[0]

    # Calculate the center of the marker in image coordinates
    marker_center_2d = np.mean(corners[marker_index][0], axis=0)

    return rvec, tvec, marker_center_2d




def reprojection_error_eye_to_hand(x0, ee_poses, marker_poses):
    camera_pose_params = x0[:6]
    marker_offset_params = x0[6:]

    # Convert camera pose parameters to 4x4 transformation matrix
    camera_rotation = cv2.Rodrigues(camera_pose_params[:3])[0]
    camera_translation = camera_pose_params[3:]
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = camera_rotation
    camera_pose[:3, 3] = camera_translation

    # Convert marker offset parameters to 4x4 transformation matrix
    marker_offset_rotation = cv2.Rodrigues(marker_offset_params[:3])[0]
    marker_offset_translation = marker_offset_params[3:]
    marker_offset = np.eye(4)
    marker_offset[:3, :3] = marker_offset_rotation
    marker_offset[:3, 3] = marker_offset_translation

    residuals = []
    for ee_pose, marker_pose in zip(ee_poses, marker_poses):
        # Calculate expected marker position in base frame
        expected_marker_pose_base = ee_pose @ marker_offset

        # Transform observed marker from camera frame to base frame
        observed_marker_pose_base = camera_pose @ marker_pose

        # Calculate residual in base frame (only position, not orientation)
        residual = (expected_marker_pose_base[:3, 3] - observed_marker_pose_base[:3, 3]).flatten()
        residuals.extend(residual)

    return np.array(residuals)

def reprojection_error_eye_in_hand(params, ee_poses, marker_poses):
    cam2ee_pose_params = params[:6]
    marker2base_pose_params = params[6:]

    cam2ee_rotation = cv2.Rodrigues(cam2ee_pose_params[:3])[0]
    cam2ee_translation = cam2ee_pose_params[3:]
    cam2ee_pose = np.eye(4)
    cam2ee_pose[:3, :3] = cam2ee_rotation
    cam2ee_pose[:3, 3] = cam2ee_translation

    marker2base_rotation = cv2.Rodrigues(marker2base_pose_params[:3])[0]
    marker2base_translation = marker2base_pose_params[3:]
    marker2base_pose = np.eye(4)
    marker2base_pose[:3, :3] = marker2base_rotation
    marker2base_pose[:3, 3] = marker2base_translation

    residuals = []
    for ee_pose, marker_pose in zip(ee_poses, marker_poses):
        # Calculate expected marker position in base frame
        expected_marker_pose_base = marker2base_pose

        # Transform observed marker from camera frame to base frame
        observed_marker_pose_base = ee_pose @ cam2ee_pose @ marker_pose

        # Calculate residual in base frame
        residual = (expected_marker_pose_base[:3, 3] - observed_marker_pose_base[:3, 3]).flatten()
        residuals.extend(residual)

    return np.array(residuals)


def get_rigid_transform(A, B):
    assert A.shape == B.shape

    # Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center the points
    points_A_centered = A - centroid_A
    points_B_centered = B - centroid_B

    # Compute SVD of covariance matrix
    H = np.dot(points_A_centered.T, points_B_centered)
    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    # Ensure a proper rotation matrix (no reflections)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute translation
    t = centroid_B - np.dot(R, centroid_A)

    return R, t