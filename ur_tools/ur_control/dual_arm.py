import os
import numpy as np
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
    cam2base2 = calibrate_eye_to_hand(camera=camera, ip_address="172.17.139.103", write_pose_to_file=False, x_num=4, y_num=4) # this is the ip of arm2
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
    def __init__(self, arm1: UR5, arm2: UR5):
        self.arm1 = arm1
        self.arm2 = arm2
    
if __name__ == "__main__":
    calibrate_arm2()
    # arm1 = UR_Robotiq_2f_85_Controller(robot_ip="172.17.139.100", init_gripper=True, dt=0.02)
    # arm1.close_gripper()
    # arm1.open_gripper()
    # arm2 = UR_Robotiq_2f_85_Controller(robot_ip="172.17.139.103", init_gripper=True, dt=0.02)
    # arm2.close_gripper()
    # arm2.open_gripper()
    # dual_arm = DualArm(arm1, arm2)
