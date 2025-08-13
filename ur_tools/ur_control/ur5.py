import numpy as np
import time
from ur_tools.camera.utils import generate_6d_pose

class UR5:
    def __init__(self, ip):
        self.ip = ip
        self.rtde_r, self.rtde_c = None, None
        self.home = None
        self.ee2base = None
    
    def moveJ(self, pos, speed, acceleration):
        self.rtde_c.moveJ(pos, speed, acceleration)
        time.sleep(0.5)
    
    def moveL(self, pos, speed, acceleration):
        self.rtde_c.moveL(pos, speed, acceleration)
        time.sleep(0.5)

    def servoJ(self, pos, speed, acceleration, look_ahead, gain):
        self.rtde_c.servoJ(pos, speed, acceleration, self.dt, look_ahead, gain)

    def get_joint_state(self):
        return self.rtde_r.getActualQ()

    def get_ee_state(self):
        return self.rtde_r.getActualTCPPose()

    def randomize_ee_pose(self, x_limits, y_limits, z_heights, x_num, y_num, randomness_up, target):
        x_values = np.linspace(x_limits[0], x_limits[1], x_num)
        y_values = np.linspace(y_limits[0], y_limits[1], y_num)
        up_axis = [0, -1, 0]
        offset_rpy = [0, np.pi, 0]  # Roll, Pitch, Yaw offsets
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        poses = []
        for x, y in zip(x_grid.flatten(), y_grid.flatten()):
            # select z randomly, and rpy randomly
            z_i = np.random.randint(0, len(z_heights))
            # add random noise to up_axis
            random_up_axis = np.array(up_axis) + np.random.uniform(-randomness_up, randomness_up, 3)
            random_up_axis = random_up_axis / np.linalg.norm(random_up_axis)
            pose = generate_6d_pose([x, y, z_heights[z_i]], target, random_up_axis, offset_rpy)
            poses.append(pose)
        return poses

    @staticmethod
    def rodrigues_rotation_matrix(rvec):
        theta = np.linalg.norm(rvec)
        if theta < 1e-8:
            return np.eye(3)

        r = rvec / theta
        K = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    @staticmethod
    def pose_to_matrix(pose):
        """
        Convert a 6D pose to a 4x4 transformation matrix.
        Args:
            pose: array-like (6,) – [x, y, z, rx, ry, rz]
        Returns:
            T: np.ndarray (4, 4) – transformation matrix
        """
        t = np.array(pose[:3])
        rvec = np.array(pose[3:])
        R = UR5.rodrigues_rotation_matrix(rvec)
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    def get_ee2base(self):
        """
        Get the transformation matrix from base to end-effector.
        Returns:
            T: np.ndarray (4, 4) – transformation matrix from base to end-effector
        """
        ee_pose = self.get_ee_state()
        self.ee2base = self.pose_to_matrix(ee_pose)
        return self.ee2base
    
    def cam2ee_to_ee2base(self, cam_ee_pose):
        """
        Convert camera end-effector pose to base end-effector pose.
        Args:
            cam_ee_pose: array-like (3,) – camera end-effector pose
        Returns:
            T: np.ndarray (4, 4) – transformation matrix from base to camera end-effector
        """
        homogeneous_cam_ee_pose = np.ones((4, 1))
        homogeneous_cam_ee_pose[:3, 0] = cam_ee_pose
        if self.ee2base is None:
            raise ValueError("End-effector to base transformation matrix is not set. Call get_ee2base() first.")
        
        world_pose = self.ee2base @ homogeneous_cam_ee_pose
        return world_pose[:3, 0]