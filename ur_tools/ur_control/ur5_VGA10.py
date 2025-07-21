import socket
import time

import numpy as np
import cv2

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_io import RTDEIOInterface as RTDEIO
from ur_tools.ur_control.ur5 import UR5


class UR5_VGA10(UR5):
    def __init__(self):
        self.rtde_frequency = 100.0
        self.joint_acc = 0.4
        self.joint_vel = 0.2
        self.joint_acc_slow = 0.2
        self.joint_vel_slow = 0.1
        self.joint_acc_fast = 1.3
        self.joint_vel_fast = 0.9
        self.tool_vel_fast = 0.6
        self.tool_acc_fast = 0.8
        self.tool_vel = 0.3
        self.tool_acc = 0.4
        self.tool_vel_slow = 0.02
        self.tool_acc_slow = 0.2
        self.tool_acc_stop = 1.0
        self.pre_grasp_config = [74.44, -91.23, 87.61, -86.29, -89.51, -105.44]
        self.pre_grasp_config = [np.deg2rad(j) for j in self.pre_grasp_config]
        self.take_photo_config = [75.25, -100.69, 68.68, -57.90, -89.57, -101.69]
        self.take_photo_config = [np.deg2rad(j) for j in self.take_photo_config]
        self.drop_joint_config = [-45.53, -120.04, 106.72, -76.62, -89.46, -135.45]
        self.drop_joint_config = [np.deg2rad(j) for j in self.drop_joint_config]
        self.take_photo_yaw = np.deg2rad(0)
        self.force_threshold = 15

        self.pixel_size = 0.001

        # check the control script for parameters
        self.vg_cmd_in_reg = 19
        self.vg_cmd_out_reg = 19
        self.vg_threshold = 30
        self.home = [
            1.305213451385498, 
            -1.592827936212057, 
            1.6703251043902796, 
            -1.647313734094137, 
            -1.5624845663653772, 
            -0.17452842393984014
            ]   # robot will go here before starting the calibration

        # Use external UR cap, on the panel -> program, need to have
        # BeforeStart -> script: rtde_initialize_vg.script
        # Robot Program -> script: rtde_control_vg.script
        _ip = "172.17.139.100"
        self.rtde_c = RTDEControl(_ip, self.rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
        self.rtde_r = RTDEReceive(_ip, self.rtde_frequency, use_upper_range_registers=False)
        self.rtde_i = RTDEIO(_ip, self.rtde_frequency, use_upper_range_registers=False)

        self.rtde_i.setInputIntRegister(self.vg_cmd_in_reg, 0)
        # self.go_take_photo()
        self.vg_grip(0.5)
        self.vg_release(0.5)


    def camera_ee2base(self, camera_ee_pose):
        _ee_pose = self.rtde_r.getActualTCPPose()
        _ee_pose = np.array(_ee_pose)
        ee_pose = np.eye(4)
        ee_pose[:3, :3] = cv2.Rodrigues(_ee_pose[3:])[0]
        ee_pose[:3, 3] = _ee_pose[:3]
        return np.dot(ee_pose, camera_ee_pose)

    def vg_grip(self, timeout=3):
        print("vacuuming...")
        self.rtde_i.setInputIntRegister(self.vg_cmd_in_reg, 1)

        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(0.1)
            curr_vg = self.rtde_r.getOutputIntRegister(self.vg_cmd_out_reg)
            if curr_vg >= self.vg_threshold:
                time.sleep(0.5)
                print(f"vacuuming done, current vacuum value = {curr_vg}")
                self.rtde_i.setInputIntRegister(self.vg_cmd_in_reg, 0)
                return True

        print(f"vacuuming timeout ({timeout}), current vacuum value = {curr_vg}")
        self.rtde_i.setInputIntRegister(self.vg_cmd_in_reg, 0)
        time.sleep(0.1)

        return False

    def vg_release(self, timeout=1):
        print("releasing...")
        self.rtde_i.setInputIntRegister(self.vg_cmd_in_reg, 2)

        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(0.1)
            curr_vg = self.rtde_r.getOutputIntRegister(self.vg_cmd_out_reg)
            if curr_vg < 1:
                print("releasing done")
                self.rtde_i.setInputIntRegister(self.vg_cmd_in_reg, 0)
                time.sleep(0.1)
                return True

        print(f"releasing timeout ({timeout}), current vacuum value = {curr_vg}")
        self.rtde_i.setInputIntRegister(self.vg_cmd_in_reg, 0)
        time.sleep(0.1)

        return False
    
    def go_home(self):
        print("Moving to home position...")
        self.moveJ(self.home, self.joint_vel_slow, self.joint_acc_slow)
        time.sleep(1)
        print("Robot is at home position.")
    
if __name__ == "__main__":
    robot = UR5_VGA10()
    init_q = robot.get_joint_state()
    print("Initial joint state:", init_q)
    init_ee_position = robot.get_ee_state()
    print("Initial end-effector position:", init_ee_position)
    init_q[0]  += 0.1
    robot.moveJ(init_q, 0.5, 0.5)
    init_q[0]  -= 0.1
    robot.moveJ(init_q, 0.5, 0.5)

    init_ee_position[2] =  0.040
    robot.moveL(init_ee_position, 0.5, 0.5)
    ee_position = robot.get_ee_state()
    print("End-effector position after move:", ee_position)
    robot.vg_grip()
    init_ee_position[2] +=  0.2
    robot.moveL(init_ee_position, 0.5, 0.5)
    ee_position = robot.get_ee_state()
    print("End-effector position after move:", ee_position)
    robot.vg_release()

    # from rtde_control import Path
    # velocity = 0.5
    # acceleration = 0.5
    # blend_1 = 0.0
    # blend_2 = 0.02
    # blend_3 = 0.0
    # path_pose1 = [-0.143, -0.435, 0.20, -0.001, 0, 0.04, velocity, acceleration, blend_1]
    # path_pose2 = [-0.143, -0.51, 0.21, -0.001, 0, 0.04, velocity, acceleration, blend_2]
    # path_pose3 = [-0.32, -0.61, 0.31, -0.001, 0, 0.04, velocity, acceleration, blend_3]
    # path = [path_pose1, path_pose2, path_pose3]
    # new_path = Path()
    # new_path.appendMovelPath(path)
    # robot.rtde_c.movePath(new_path)
    # # robot.rtde_c.stopScript()
    # exit()

    # robot.vg_grip()

    # input('wait')

    # robot.vg_release()

    # from real_camera.camera import Camera

    # camera = Camera()

    # # robot.calibrate_marker(camera, False)

    # robot.handeye_calibration(camera)

    # robot.rtde_c.stopScript()
