import os
import time
import numpy as np
from typing import List
from ur_tools.ur_control.ur5 import UR5
from ur_tools.ur_control.ur5_robotiq_2f_85_control import UR_Robotiq_2f_85_Controller
from ur_tools.camera.calibrate_eye_to_hand import calibrate_eye_to_hand
from ur_tools.camera.realsense_camera import Camera

def calibrate_arm2():
    camera = Camera(calibrated=True, on_hand=False)
    cam2base1 = camera.cam2robot
    print(f"Camera pose relative to arm1 base:\n{cam2base1}")
    # We assume arm1's base is at the origin of the world frame
    # Now we have to calibrate the base of arm2 to the camera
    cam2base2 = calibrate_eye_to_hand(camera=camera, ip_address="172.17.139.103", write_pose_to_file=False, x_num=5, y_num=5) # this is the ip of arm2
    camera_rotation = cam2base2[:3, :3] 
    camera_translation = cam2base2[:3, 3]
    print(f"Camera rotation relative to arm2 base:\n{camera_rotation}")
    print(f"Camera translation relative to arm2 base:\n{camera_translation}")
    base2base = cam2base1 @ np.linalg.inv(cam2base2)
    print(f"Base2base transformation:\n{base2base}")
    print("Saving...")
    saving_path = os.path.join(os.path.dirname(__file__), "base2base.txt")
    np.savetxt(saving_path, base2base, delimiter=" ")

    
class DualArm:
    def __init__(self, arm1: UR5, arm2: UR5, speed=0.5, acceleration=0.5):
        self.arm1 = arm1
        self.arm2 = arm2
        self.speed = speed
        self.acceleration = acceleration
        assert self.arm1.dt == self.arm2.dt, "The time step of the two arms must be the same."
        self.dt = self.arm1.dt
        self.arm1.close_gripper()
        self.arm1.open_gripper()
        self.arm2.close_gripper()
        self.arm2.open_gripper()
        self.look_ahead = 0.1
        self.gain = 1000
        self.home = [
            [1.5708118677139282, -2.2, 1.9, -1.383, -1.5700505415545862, 0],
            [1.5708118677139282, -2.2, 1.9, -1.383, -1.5700505415545862, 0]
        ]

    def gripper_control(self, action1, action2):
        if action1 == "CLOSE":
            self.arm1.close_gripper()
        elif action1 == "OPEN":
            self.arm1.open_gripper()

        if action2 == "CLOSE":
            self.arm2.close_gripper()
        elif action2 == "OPEN":
            self.arm2.open_gripper()
        time.sleep(0.5)
    
    def go_home(self):
        init_q = self.get_joint_states()
        interpolate = lambda v1, v2, t: (1 - t) * v1 + t * v2
        for i in np.linspace(0, 1, 100):
            self.arm1.servoJ(interpolate(np.array(init_q[0]), np.array(self.home[0]), i), speed=self.speed, acceleration=self.acceleration, look_ahead=self.look_ahead, gain=self.gain)
            self.arm2.servoJ(interpolate(np.array(init_q[1]), np.array(self.home[1]), i), speed=self.speed, acceleration=self.acceleration, look_ahead=self.look_ahead, gain=self.gain)
            time.sleep(self.arm1.dt)
            if np.linalg.norm(np.array(self.arm1.get_joint_state()) - self.home[0]) < 1e-3 and np.linalg.norm(np.array(self.arm2.get_joint_state()) - self.home[1]) < 1e-3:
                break
        print('Robot moved to home position')

    def move_to_start(self, plan):
        interpolate = lambda v1, v2, t: (1 - t) * v1 + t * v2
        init_q = self.get_joint_states()
        start = [plan[0][0][0], plan[0][1][0]]
        for i in np.linspace(0, 1, 100):
            self.arm1.servoJ(interpolate(np.array(init_q[0]), np.array(start[0]), i), speed=self.speed, acceleration=self.acceleration, look_ahead=self.look_ahead, gain=self.gain)
            self.arm2.servoJ(interpolate(np.array(init_q[1]), np.array(start[1]), i), speed=self.speed, acceleration=self.acceleration, look_ahead=self.look_ahead, gain=self.gain)
            time.sleep(self.arm1.dt)
            if np.linalg.norm(np.array(self.arm1.get_joint_state()) - start) < 1e-3 and np.linalg.norm(np.array(self.arm2.get_joint_state()) - start) < 1e-3:
                break
        print('Robot moved to start position, ready to execute the plan')

    def execute(self, recorded_plan, interpolation_num=5):
        self.go_home()
        self.move_to_start(recorded_plan)
        input("Press Enter to start execution...")
        time.sleep(10)
        last_position = None
        for section in recorded_plan:
            if type(section[0]) == str:
                self.gripper_control(section[0], section[1])
            else:
                self.sanity_check(section, last_position)
                last_position = [section[0][-1], section[1][-1]]
                self.servoJ(section, interpolation_num=interpolation_num)
    
    def sanity_check(self, plan, last_position=None):
        # check if one section of plan has too large deviation
        if last_position is not None:
            if np.linalg.norm(np.array(plan[0][0]) - np.array(last_position[0])) > 0.1:
                print(f"Warning: Large deviation detected for arm{0} and last position with {np.linalg.norm(np.array(plan[0][0]) - np.array(last_position[0]))}")
                print(plan[0][0])
                print(last_position[0])
            if np.linalg.norm(np.array(plan[1][0]) - np.array(last_position[1])) > 0.1:
                print(f"Warning: Large deviation detected for arm{1} and last position with {np.linalg.norm(np.array(plan[1][0]) - np.array(last_position[1]))}")
                print(plan[1][0])
                print(last_position[1])
        for i in range(len(plan[0])-1):
            if np.linalg.norm(np.array(plan[0][i]) - np.array(plan[0][i+1])) > 0.1:
                print(f"Warning: Large deviation detected for arm{0} between section {i} and {i+1} with {np.linalg.norm(np.array(plan[0][i]) - np.array(plan[0][i+1]))}")
                print(plan[0][i])
                print(plan[0][i+1])
            if np.linalg.norm(np.array(plan[1][i]) - np.array(plan[1][i+1])) > 0.1:
                print(f"Warning: Large deviation detected for arm{1} between section {i} and {i+1} with {np.linalg.norm(np.array(plan[1][i]) - np.array(plan[1][i+1]))}")
                print(plan[1][i])
                print(plan[1][i+1])
        print("Sanity check complete")

    def moveJ(self, positions:List[List[float]]):
        print("moving arm1")
        self.arm1.moveJ(positions[0], speed=self.speed, acceleration=self.acceleration, sleep=0)
        print("moving arm2")
        self.arm2.moveJ(positions[1], speed=self.speed, acceleration=self.acceleration, sleep=0)
        time.sleep(self.dt)
        
    
    def servoJ(self, plan:List, interpolation_num:int=0):
        interpolate = lambda v1, v2, t: (1 - t) * v1 + t * v2
        for i in range(len(plan[0])-1):
            pos1 = plan[0][i]
            pos2 = plan[1][i]
            print(f"arm1: {pos1}")
            print(f"arm2: {pos2}")
            # self.arm1.moveJ(pos=pos1, speed=self.speed, acceleration=self.acceleration, sleep=0.0)
            # self.arm2.moveJ(pos=pos2, speed=self.speed, acceleration=self.acceleration, sleep=0.0)
            # self.moveJ([pos1, pos2])
            # time.sleep(0.5)
            if interpolation_num > 0:
                for j in range(interpolation_num):
                    self.arm1.servoJ(pos=interpolate(plan[0][i], plan[0][i+1], j/interpolation_num), look_ahead=self.look_ahead, gain=self.gain, speed=self.speed, acceleration=self.acceleration)
                    self.arm2.servoJ(pos=interpolate(plan[1][i], plan[1][i+1], j/interpolation_num), look_ahead=self.look_ahead, gain=self.gain, speed=self.speed, acceleration=self.acceleration)
                    time.sleep(self.dt)
            else:
                self.arm1.servoJ(pos=plan[0][i], look_ahead=self.look_ahead, gain=self.gain, speed=self.speed, acceleration=self.acceleration)
                self.arm2.servoJ(pos=plan[1][i], look_ahead=self.look_ahead, gain=self.gain, speed=self.speed, acceleration=self.acceleration)
                time.sleep(self.dt)

    def get_joint_states(self):
        arm1_joint_states = self.arm1.get_joint_state()
        arm2_joint_states = self.arm2.get_joint_state()
        return [arm1_joint_states, arm2_joint_states]

if __name__ == "__main__":
    calibrate_arm2()
    # arm1 = UR_Robotiq_2f_85_Controller(robot_ip="172.17.139.100", init_gripper=True, dt=0.02)
    # arm2 = UR_Robotiq_2f_85_Controller(robot_ip="172.17.139.103", init_gripper=True, dt=0.02)
    # dual_arm = DualArm(arm1, arm2)
    # init_q = dual_arm.get_joint_states()
    # print("Initial joint states:")
    # print(init_q)
    # for i in range(5):
    #     init_q[0][0] += 0.1
    #     init_q[1][0] += 0.1
    #     dual_arm.moveJ(init_q)
    #     init_q[0][0] -= 0.1
    #     init_q[1][0] -= 0.1
    #     dual_arm.moveJ(init_q)




