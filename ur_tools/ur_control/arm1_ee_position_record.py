import os
import numpy as np
from ur_tools.ur_control.ur5_VGA10 import UR5_VGA10


arm = UR5_VGA10(ip="172.17.139.103")
ee_state = arm.get_ee_state()
print(ee_state)
# open a txt file name arm1 ee positions and append the x, y, z of the current end effector state
file_path = "ur_tools/ur_control/arm2_ee_positions.txt"
with open(file_path, "a") as f:
    f.write(f"{ee_state[0]} {ee_state[1]} {ee_state[2]}\n")
