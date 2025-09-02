import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from ur_tools.camera.realsense_camera import Camera
from ur_tools.ur_control.ur5_robotiq_2f_85_control import UR_Robotiq_2f_85_Controller 
from ur_tools.ur_control.ur5_VGA10 import UR5_VGA10


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

def debug_second_arm():
    dt = 0.002
    robot = UR5_VGA10(ip="172.17.139.103")
    init_ee_pos = robot.get_ee_state()
    robot.moveJ(
        [1.5708118677139282, -2.2, 1.9, -1.383, -1.5700505415545862, 0], speed=0.3, acceleration=0.3
    )
    print("Initial end-effector position:\n", init_ee_pos)
    base2base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ur_tools/ur_control", "base2base.txt")
    base2base = np.loadtxt(base2base_path, delimiter=" ")
    print("Base to base transformation:\n", base2base)
    # get the rotation axis and theta from the transformation matrix using scipy
    rot = R.from_matrix(base2base[:3,:3])
    rotvec = rot.as_rotvec()
    theta = np.linalg.norm(rotvec)
    axis = rotvec / theta if theta > 0 else np.array([1, 0, 0])
    print("Rotation axis:", axis)
    print("Rotation angle (theta) in degrees:", np.degrees(theta))
    exit()
    xs = np.array([-0.1, 0, 0.1])
    ys = np.array([-0.4, -0.5, -0.6])
    xs, ys = np.meshgrid(xs, ys)
    z = 0
    # make xs, ys, z a (N, 3) array
    P = np.column_stack((xs.flatten(), ys.flatten(), z * np.ones_like(xs.flatten())))
    print("End-effector positions:\n", P)
    # transfer P to Q
    Q = (base2base[:3,:3] @ P.T).T + base2base[:3,3]
    print("Transformed end-effector positions:\n", Q)
    for point in Q:
        robot.moveL([point[0], point[1], 0.2, 0, np.pi, 0], 0.1, 0.1)
        robot.moveL([point[0], point[1], 0.002, 0, np.pi, 0], 0.1, 0.1)
        input("Press Enter to continue...")
        robot.moveL([point[0], point[1], 0.02, 0, np.pi, 0], 0.1, 0.1)

if __name__ == "__main__":
    debug_second_arm()