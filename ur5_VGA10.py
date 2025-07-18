import sys
import os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)
import rtde_control
import rtde_receive
import numpy as np
import pandas as pd
import time
from pybullet_utils import bullet_client
import pybullet_data
import pybullet as pb
from robotiq_gripper import RobotiqGripper
from robotiq_gripper_control import RobotiqGripper as RobotiqGripperControl
from scipy.spatial.transform import Rotation as R
from utils import Colors


class UR_VGA10_Controller:
    def __init__(self, robot_ip="172.17.139.100", dt=1.0 / 500, robot_position=None, init_gripper=False, init_pb=False):
        self.ip = robot_ip
        self.dt = dt
        self.rtde_c, self.rtde_r = self.init_robot()
        self.gripper = self.init_gripper() if init_gripper else None
        self.robot_position = robot_position
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
        if init_pb:
            self.init_pb_env()

    def init_pb_env(self):
        self._pb = bullet_client.BulletClient(connection_mode=pb.DIRECT)
        self._pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._pb.setTimeStep(self.dt)
        print(Colors.GREEN, "Pybullet initialized", Colors.RESET)
        self.reset_pb_env()
        print(Colors.GREEN, "Pybullet env reset", Colors.RESET)

    def reset_pb_env(self):
        ''' reset Pybullet env '''
        self._pb.resetSimulation()
        self._pb.setGravity(0, 0, -9.8)
        self.pb_base_offset = self.robot_position
        # Load two UR5e
        self.pb_robot_arm = self._pb.loadURDF(
            os.path.join(os.path.join(root_path, "motion_planning"), "assets/ur5e/ur5e.urdf"),
            basePosition=self.pb_base_offset, useFixedBase=True)
        self.pb_robot_joints = []
        for i in range(self._pb.getNumJoints(self.pb_robot_arm)):
            info = self._pb.getJointInfo(self.pb_robot_arm, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            # if joint_name == "ee_fixed_joint":
            if joint_name == "flange-tool0":
                self.pb_robot_ee_id = joint_id
            if joint_type == pb.JOINT_REVOLUTE:
                self.pb_robot_joints.append(joint_id)
        self._pb.enableJointForceTorqueSensor(self.pb_robot_arm, self.pb_robot_ee_id, 1)

    def pb_get_pos(self, joint_state):
        ''' joint state -> pose state '''
        for joint_id in range(len(self.pb_robot_joints)):
            self._pb.setJointMotorControl2(
                bodyUniqueId=self.pb_robot_arm,
                jointIndex=self.pb_robot_joints[joint_id],
                controlMode=pb.POSITION_CONTROL,
                targetPosition=joint_state[joint_id],
                positionGain=1,
                maxVelocity=3.14,
            )
        step = 0
        max_step = 500
        while True:
            error = np.max(np.abs(
                np.array([self._pb.getJointState(self.pb_robot_arm, i)[0] for i in self.pb_robot_joints]) - np.array(
                    joint_state)))
            self._pb.stepSimulation()
            step += 1
            if step > max_step or error < EPSILON:
                break
        ee_pos, ee_rot = pb.getLinkState(
            self.pb_robot_arm, self.pb_robot_ee_id
        )[:2]
        return ee_pos, ee_rot

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
            self.gripper.open(self.gripper._max_speed, 0)
        elif isinstance(self.gripper, RobotiqGripperControl):
            if self.gripper is None:
                print(Colors.RED, "Warning: Gripper is not initialized!", Colors.RESET)
                return
            self.gripper.open()

    def reached_goal(self, goal_q, tol=0.001):
        actual_q = self.rtde_r.getActualQ()
        if np.linalg.norm(np.array(actual_q) - np.array(goal_q)) < tol:
            return True
        else:
            return False

    def control_with_servoJ(self, steps, look_ahead=0.03, gain=200, speed=0.5, acceleration=0.5,
                            fixed_dt: bool = True, test_gripper_communication=False):  # gain=500
        self.actual_positions = []
        self.desired_positions = []
        self.actual_ee_positions = []
        self.desired_ee_positions = []
        self.actual_ee_rot = []
        self.desired_ee_rot = []

        start_time = time.time()
        for i in range(steps):
            waypoint = self.traj[i]
            pos = waypoint.position[:6].cpu().tolist()
            self.desired_positions.append(pos)
            pos[0] += np.pi / 2
            self.rtde_c.servoJ(pos, speed, acceleration, self.dt, look_ahead, gain)
            time.sleep(self.dt)
            actual_q = self.rtde_r.getActualQ()
            actual_ee = self.rtde_r.getActualTCPPose()
            actual_ee[2] += self.tool_z_offset
            self.actual_positions.append(actual_q)
            rot = R.from_rotvec(actual_ee[3:])
            actual_rot = rot.as_euler("ZYX", degrees=True)
            self.actual_ee_positions.append(actual_ee[:3])
            self.actual_ee_rot.append(actual_rot.tolist())
        duration = time.time() - start_time

        time.sleep(0.5)
        for i in range(steps - 1, -1, -1):
            waypoint = self.traj[i]
            pos = waypoint.position[:6].cpu().tolist()
            self.desired_positions.append(pos)
            pos[0] += np.pi / 2
            if test_gripper_communication:
                self.close_gripper()
            self.rtde_c.servoJ(pos, speed, acceleration, self.dt, look_ahead, gain)
            time.sleep(self.dt)
            if test_gripper_communication:
                self.open_gripper()
            actual_q = self.rtde_r.getActualQ()
            actual_ee = self.rtde_r.getActualTCPPose()
            actual_ee[2] += self.tool_z_offset
            self.actual_positions.append(actual_q)
            rot = R.from_rotvec(actual_ee[3:])
            actual_rot = rot.as_euler("ZYX", degrees=True)
            self.actual_ee_positions.append(actual_ee[:3])
            self.actual_ee_rot.append(actual_rot.tolist())
        self.open_gripper()

        if self.init_pb_env:
            if self._pb is None:
                self.init_pb()
            for i in range(len(self.desired_positions)):
                joint_state = self.desired_positions[i]
                ee_pos, ee_rot = self.pb_get_pos(joint_state)
                ee_pos -= self.pb_base_offset
                rot = R.from_quat(ee_rot)
                R1 = rot.as_matrix()
                # R2 = R.from_euler('Z', -np.pi/2)
                R2 = R.from_euler('Z', -np.pi)
                rot = R.from_matrix(R2.as_matrix() @ R1)
                self.desired_ee_positions.append(R2.apply(ee_pos))
                # self.desired_ee_positions.append(ee_pos)
                self.desired_ee_rot.append(rot.as_euler("ZYX", degrees=True))
        return duration

    def moveJ(self, pos, speed, acceleration):
        self.rtde_c.moveJ(pos, speed, acceleration)

    def servoJ(self, pos, speed, acceleration, look_ahead, gain):
        self.rtde_c.servoJ(pos, speed, acceleration, self.dt, look_ahead, gain)

    def get_joint_state(self):
        return self.rtde_r.getActualQ()

    def get_ee_state(self):
        return self.rtde_r.getActualTCPPose()

def single_arm_test():
    dt = 0.002
    controller0 = UR_VGA10_Controller(dt=dt, robot_ip="172.17.139.100", init_pb=False, init_gripper=False)
    init_q = controller0.get_joint_state()
    first_q = init_q
    first_q[0] += 0.1
    controller0.moveJ(first_q, speed=0.5, acceleration=0.5)
    first_q[0] -= 0.1
    controller0.moveJ(first_q, speed=0.5, acceleration=0.5)
    # controller0.close_gripper()
    # controller0.open_gripper()

if __name__ == "__main__":
    single_arm_test()
