class UR5:
    def __init__(self, ip):
        self.ip = ip
        self.rtde_r, self.rtde_c = None, None
    
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