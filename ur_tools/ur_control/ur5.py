import numpy as np

class UR5:
    def __init__(self, ip):
        self.ip = ip
        self.rtde_r, self.rtde_c = None, None
        self.home = None
    
    def moveJ(self, pos, speed, acceleration):
        self.rtde_c.moveJ(pos, speed, acceleration)
    
    def moveL(self, pos, speed, acceleration):
        self.rtde_c.moveL(pos, speed, acceleration)

    def servoJ(self, pos, speed, acceleration, look_ahead, gain):
        self.rtde_c.servoJ(pos, speed, acceleration, self.dt, look_ahead, gain)

    def get_joint_state(self):
        return self.rtde_r.getActualQ()

    def get_ee_state(self):
        return self.rtde_r.getActualTCPPose()
    
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
        return self.pose_to_matrix(ee_pose)
