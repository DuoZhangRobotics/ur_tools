import cv2
import numpy as np
from ur_tools.camera.realsense_camera import Camera


camera = Camera(on_hand=False)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
detector = cv2.aruco.ArucoDetector(aruco_dict)

camera.show_video_realtime()
colored_image, depth_image = camera.get_data()

# Convert to grayscale for ArUco detection
gray = cv2.cvtColor(colored_image, cv2.COLOR_RGB2GRAY)

corners, ids, rejected = detector.detectMarkers(gray)

center = np.mean(corners[0], axis=1)
print(f"Center of detected marker: {center}")
x, y = center[0][0], center[0][1]
z = depth_image[int(y), int(x)]
print(f"Depth of detected marker: {z}")
x_cam = (x - camera.intrinsics[0, 2]) * z / camera.intrinsics[0, 0]
y_cam = (y - camera.intrinsics[1, 2]) * z / camera.intrinsics[1, 1]
print(f"Camera coordinates: ({x_cam}, {y_cam}, {z})")
world_pos = camera.pos_in_camera_to_ee(np.array([[x_cam, y_cam, z]]))
print(f"World coordinates: {world_pos}")

world_pos = camera.pos_in_image_to_robot(np.array([[x, y]]), np.array([z]))
print(f"World coordinates: {world_pos}")
