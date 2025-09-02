import os
import numpy as np
from ur_tools.ur_control.ur5_VGA10 import UR5_VGA10

xs = np.array([-0.1, 0, 0.1])
ys = np.array([-0.4, -0.5, -0.6])
xs, ys = np.meshgrid(xs, ys)
arm = UR5_VGA10(ip="172.17.139.103")
# arm.moveL([0, -0.5, 0.4, 0, np.pi, 0], 0.1, 0.1)
cnt=0
skip_cnt = 8
for x, y in zip(xs.flatten(), ys.flatten()):
    x = -1 * x
    y = -1.095-y
    if cnt < skip_cnt:
        cnt += 1
        continue
    print(f"Moving to position: x={x}, y={y}")
    arm.moveL([x, y, 0.2, 0, np.pi, 0], 0.1, 0.1)
    arm.moveL([x, y, 0.01, 0, np.pi, 0], 0.1, 0.1)
    input("Press enter to move to the next position")
    arm.moveL([x, y, 0.2, 0, np.pi, 0], 0.1, 0.1)
    