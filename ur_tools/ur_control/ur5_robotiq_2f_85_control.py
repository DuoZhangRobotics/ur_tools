from ur_tools.ur_control.ur5 import UR5
from ur_tools.ur_control.utils import Colors
from scipy.spatial.transform import Rotation as R
from ur_tools.ur_control.robotiq_gripper_control import RobotiqGripper as RobotiqGripperControl
from ur_tools.ur_control.robotiq_gripper import RobotiqGripper
import pybullet as pb
import pybullet_data
from pybullet_utils import bullet_client
import time
import pandas as pd
import numpy as np
import rtde_receive
import rtde_control
import sys
import os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)


class UR_Robotiq_2f_85_Controller(UR5):
    def __init__(self, robot_ip="172.17.139.100", dt=1.0 / 500, init_gripper=False):
        self.ip = robot_ip
        self.dt = dt
        self.rtde_c, self.rtde_r = self.init_robot()
        self.gripper = self.init_gripper() if init_gripper else None
        self.traj = None
        self._pb = None
        self.pb_robot_arm = None
        self.pb_robot_ee_id = None
        self.pb_robot_joints = None
        self.pb_overlap_rate = None
        self.pb_base_offset = None
        self.actual_positions = []
        self.desired_positions = []
        self.actual_ee_positions = []
        self.desired_ee_positions = []
        self.actual_ee_rot = []
        self.desired_ee_rot = []
        self.tool_z_offset = 0.21

    def init_robot(self):
        rtde_c = rtde_control.RTDEControlInterface(self.ip)
        rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)
        actual_q = rtde_r.getActualQ()
        print("Init q: ", actual_q)
        actual_q = rtde_r.getActualQ()
        return rtde_c, rtde_r

    def init_gripper(self):
        gripper = RobotiqGripper(self.ip, 63352)
        gripper.connect()
        if not gripper.is_active():
            gripper.activate()
        # gripper = RobotiqGripperControl(self.rtde_c)
        # gripper.activate()
        # gripper.set_force(0)
        # gripper.set_speed(0)
        return gripper

    def close_gripper(self):
        if isinstance(self.gripper, RobotiqGripper):
            if self.gripper is None:
                print(Colors.RED, "Warning: Gripper is not initialized!", Colors.RESET)
                return
            gripper_closed_threshold = self.gripper.get_closed_position() - 10
            self.gripper.close_and_wait_for_pos(self.gripper._max_speed, 0)
            gripper_pos = self.gripper.get_current_position()
            grasp_success = gripper_pos < gripper_closed_threshold
            if not grasp_success:
                print(Colors.RED, "Warning: Grasp failed!", Colors.RESET)
            else:
                print(Colors.GREEN, "Grasp success!", Colors.RESET)
            return grasp_success
            # self.gripper.close(self.gripper._max_speed, 0)
        elif isinstance(self.gripper, RobotiqGripperControl):
            if self.gripper is None:
                print(Colors.RED, "Warning: Gripper is not initialized!", Colors.RESET)
                return
            self.gripper.close()

    def open_gripper(self):
        if isinstance(self.gripper, RobotiqGripper):
            if self.gripper is None:
                print(Colors.RED, "Warning: Gripper is not initialized!", Colors.RESET)
                return
            # self.gripper.open(self.gripper._max_speed, 0)
            self.gripper_move_to(10)
        elif isinstance(self.gripper, RobotiqGripperControl):
            if self.gripper is None:
                print(Colors.RED, "Warning: Gripper is not initialized!", Colors.RESET)
                return
            self.gripper.open()

    def gripper_move_to(self, position):
        if isinstance(self.gripper, RobotiqGripper):
            if self.gripper is None:
                print(Colors.RED, "Warning: Gripper is not initialized!", Colors.RESET)
                return
            self.gripper.move(position, speed=self.gripper._max_speed, force=0)
        elif isinstance(self.gripper, RobotiqGripperControl):
            if self.gripper is None:
                print(Colors.RED, "Warning: Gripper is not initialized!", Colors.RESET)
                return

    def reached_goal(self, goal_q, tol=0.001):
        actual_q = self.rtde_r.getActualQ()
        if np.linalg.norm(np.array(actual_q) - np.array(goal_q)) < tol:
            return True
        else:
            return False

    # def moveJ(self, pos, speed, acceleration):
    #     self.rtde_c.moveJ(pos, speed, acceleration)

    # def servoJ(self, pos, speed, acceleration, look_ahead, gain):
    #     self.rtde_c.servoJ(pos, speed, acceleration, self.dt, look_ahead, gain)

    # def get_joint_state(self):
    #     return self.rtde_r.getActualQ()

    # def get_ee_state(self):
    #     return self.rtde_r.getActualTCPPose()


def single_arm_test():
    dt = 0.002
    robot = UR_Robotiq_2f_85_Controller(
        dt=dt, robot_ip="172.17.139.100", init_gripper=True)
    init_q = robot.get_joint_state()
    print("Initial joint state:\n", init_q)
    init_ee_pos = robot.get_ee_state()
    print("Initial end-effector position:\n", init_ee_pos)
    # robot.gripper_move_to(50)
    # robot.moveJ([0, -np.pi/2, 0, 0, 0, 0], speed=0.1, acceleration=0.1)

    robot.close_gripper()
    robot.open_gripper()

    # for i in range(3):
    #     init_q[0] += 0.1
    #     robot.moveJ(init_q, 0.5, 0.5, sleep=0.1)
    #     init_q[0] -= 0.1
    #     robot.moveJ(init_q, 0.5, 0.5, sleep=0.1)
    # controller0.close_gripper()
    # controller0.open_gripper()


if __name__ == "__main__":
    single_arm_test()
