import numpy as np
from utils import generate_6d_pose, calibrate_eye_hand, reprojection_error_eye_in_hand, get_board_pose


up_axis = [0, -1, 0]                # camera up axis when looking at the target #Duo: This does not mean the direction of the lens.
eye = [0, -0.5, 0.3]
target = [0, -0.5, 0]
offset_rpy = [0, np.pi, 0]          # adjust the rpy of the camera to match the robot's coordinate system
pose = generate_6d_pose(eye, target, up_axis, offset_rpy)
print(pose)