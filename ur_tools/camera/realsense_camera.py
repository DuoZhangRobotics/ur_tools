# Author: Baichuan Huang

# First import the library
import pyrealsense2 as rs

# Import Numpy for easy array manipulation
import numpy as np

# Import OpenCV for easy image rendering
import cv2
import sys
import os
import atexit
from typing import Tuple
import time
from subprocess import Popen, PIPE
from scipy import optimize
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R

from ur_tools.camera.utils import *

class Camera:
    """Customized realsense camera"""

    def __init__(self, calibrated=True):
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_dict_arm = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        aruco_dict_board = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_detector_board = cv2.aruco.ArucoDetector(aruco_dict_board, aruco_params)
        self.aruco_detector_arm = cv2.aruco.ArucoDetector(aruco_dict_arm, aruco_params)

        # Create a pipeline
        self.pipeline = rs.pipeline()

        config = rs.config()
        # config.enable_device('f1380397')

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.device_name = str(device.get_info(rs.camera_info.name))
        print(self.device_name)

        if self.device_name == "Intel RealSense D455":
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        elif self.device_name == "Intel RealSense D435":
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        elif self.device_name == "Intel RealSense L515":
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        else:
            raise Exception("Undefined device!")

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Start streaming
        profile = self.pipeline.start(config)
        atexit.register(self.stop_streaming)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        # >>>>> In case of "RuntimeError: Frame didn't arrive within 5000"
        # device = profile.get_device()
        # depth_sensor = device.first_depth_sensor()
        # device.hardware_reset()
        # <<<<<
        self.depth_scale = depth_sensor.get_depth_scale()

        if self.device_name == "Intel RealSense D455":
            preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
            for i in range(int(preset_range.max)):
                visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset, i)
                if visulpreset == "Default":
                    print("Setting visual preset to Default")
                    # depth_sensor.set_option(rs.option.visual_preset, i)
                    break
            color_sensor = profile.get_device().first_color_sensor()
            color_sensor.set_option(rs.option.enable_auto_exposure, False)
            color_sensor.set_option(rs.option.exposure, 200)
            # color_sensor.set_option(rs.option.gain, 64)
            color_sensor.set_option(rs.option.power_line_frequency, 2)
        elif self.device_name == "Intel RealSense D435":
            preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
            for i in range(int(preset_range.max)):
                visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset, i)
                if visulpreset == "Default":
                    print("Setting visual preset to Default")
                    depth_sensor.set_option(rs.option.visual_preset, i)
                    break
            color_sensor = profile.get_device().first_color_sensor()
            color_sensor.set_option(rs.option.enable_auto_exposure, True)
            color_sensor.set_option(rs.option.power_line_frequency, 2)
        elif self.device_name == "Intel RealSense L515":
            # print("Setting visual preset to Short Range")
            # depth_sensor.set_option(rs.option.visual_preset, int(rs.l500_visual_preset.short_range))
            print("Setting visual preset to No Ambient Light")
            depth_sensor.set_option(rs.option.visual_preset, int(rs.l500_visual_preset.no_ambient_light))
            depth_sensor.set_option(
                rs.option.min_distance, 200
            )  # 0.2 meters.
            depth_sensor.set_option(
                rs.option.confidence_threshold, 3.0
            )  # default is 1.0
            color_sensor = profile.get_device().first_color_sensor()
            color_sensor.set_option(rs.option.enable_auto_exposure, True)
            # color_sensor.set_option(rs.option.exposure, 300.000)
            # color_sensor.set_option(rs.option.gain, 500.000)
            # color_sensor.set_option(rs.option.power_line_frequency, 2)
            # color_sensor.set_option(rs.option.contrast, 50.0)
            

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        self.clipping_distance_in_meters = [0.2, 1]

        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        self.intrinsics = np.array([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]])
        self.distortion_coeffs = np.array(intrinsics.coeffs)

        print(color_profile)
        print("Depth Scale is: ", self.depth_scale)
        print(f"Clipping distance: {self.clipping_distance_in_meters} meters")
        print("Intrinsics:\n", self.intrinsics)

        # Filters
        self.decimation = rs.decimation_filter(2)
        self.depth_to_disparity = rs.disparity_transform(True)
        self.temporal = rs.temporal_filter(0.4, 20, 3)
        # self.spatial = rs.spatial_filter(0.5, 20, 2, 0)
        self.disparity_to_depth = rs.disparity_transform(False)
        # preset_range = self.temporal.get_option_range(rs.option.filter_smooth_alpha)
        # print(preset_range, preset_range.min, preset_range.max, preset_range.step)

        # Give some time to be stable
        print("Give time for camera to warm up")
        start_time = time.time()
        while time.time() - start_time < 0.1:
            self.pipeline.wait_for_frames()

        if calibrated:
            self.cam2ee = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cam2ee_pose.txt'), delimiter=" ")
            self.cam2ee_inv = np.linalg.inv(self.cam2ee)
            print("Reading camera pose from file\n", self.cam2ee)
        self.configs = [{
            "image_size": (color_profile.height(), color_profile.width()),
            "intrinsics": self.intrinsics,
            # "position": self.camera_pose[:3, 3],
            # "rotation": R.from_matrix(self.camera_pose[:3, :3]).as_quat(),
            "zrange": (0.01, 1.0),
        }]


    def stop_streaming(self):
        """Release camera resource"""
        self.pipeline.stop()
        print("Stop camera")

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            np.ndarray: color image (BGR)
            np.ndarray: depth image
        """
        # Get frameset of color and depth, consider temporal filter, so that the depth image is more stable
        for _ in range(10):
            # Get frameset of color and depth
            frames = self.pipeline.wait_for_frames()

            # Filtering
            if self.device_name == "Intel RealSense D455":
                frames = self.decimation.process(frames).as_frameset()
                frames = self.depth_to_disparity.process(frames).as_frameset()
                frames = self.temporal.process(frames).as_frameset()
                frames = self.disparity_to_depth.process(frames).as_frameset()
            elif self.device_name == "Intel RealSense D435":
                frames = self.decimation.process(frames).as_frameset()
                frames = self.depth_to_disparity.process(frames).as_frameset()
                frames = self.temporal.process(frames).as_frameset()
                frames = self.disparity_to_depth.process(frames).as_frameset()
            elif self.device_name == "Intel RealSense L515":
                # frames = self.depth_to_disparity.process(frames).as_frameset()
                # frames = self.temporal.process(frames).as_frameset()
                # frames = self.disparity_to_depth.process(frames).as_frameset()
                pass

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            # Validate that both frames are valid
            assert color_frame
            assert aligned_depth_frame

            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            depth_image = np.asanyarray(aligned_depth_frame.get_data())

            depth_image = depth_image * self.depth_scale
            depth_image = depth_image.astype(np.float32)
            depth_image[depth_image > self.clipping_distance_in_meters[1]] = self.clipping_distance_in_meters[1]
            depth_image[depth_image < self.clipping_distance_in_meters[0]] = 0

        return color_image, depth_image
    

    def show_data(self, matplotlib=False):
        """Show color and depth image"""
        if matplotlib:
            import matplotlib.pyplot as plt
            color_image, depth_image = self.get_data()
            plt.subplot(121)
            plt.imshow(color_image)
            plt.title("Color Image")
            plt.axis("off")
            plt.subplot(122)
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = np.uint8(depth_normalized)
            depth_normalized = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            plt.imshow(depth_normalized)
            plt.title("Depth Image")
            plt.axis("off")
            plt.show()
        else:
            color_image, depth_image = self.get_data()
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = np.uint8(depth_normalized)
            depth_normalized = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            cv2.imshow("Color Image", color_image)
            cv2.imshow("Depth Image", depth_normalized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def read_tags_board(self, color_img):

        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = self.aruco_detector_board.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            print("No ArUco markers detected.")
            return None

        tag_loc_camera = {}
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners = corners[i][0]
            marker_center_2d = np.mean(marker_corners, axis=0)
            tag_loc_camera[marker_id] = marker_center_2d
            print("this is marker_id", marker_id)
            assert marker_id in self.tag_loc_robot
        assert len(tag_loc_camera) == len(self.tag_loc_robot)

        return tag_loc_camera

    def pose_in_robot_to_image(self, poses_in_robot):
        """Convert 3D points in robot coordinate to 2D points in image coordinate"""
        rvec = cv2.Rodrigues(self.cam2ee_inv[:3, :3])[0]
        tvec = self.cam2ee_inv[:3, 3]
        projected_points, _ = cv2.projectPoints(poses_in_robot, rvec, tvec, self.intrinsics, self.distortion_coeffs)
        return projected_points

    def pos_in_camera_to_ee(self, points_in_camera):
        """
        Args: xyz_in_camera: 3D point in camera coordinate
        """
        homogeneous_points = np.hstack((points_in_camera, np.ones((len(points_in_camera), 1))))
        homogeneous_points_world = np.dot(self.cam2ee, homogeneous_points.T).T

        return homogeneous_points_world[:, :3]

    def pos_in_image_to_robot(self, image_points, depths):
        """
        Args: image_points: 2D points in image coordinate
            depths: depth of the points
        """
        # Invert the intrinsics matrix
        intrinsics_inv = np.linalg.inv(self.intrinsics)

        # Convert image point to normalized point
        image_points_homogeneous = np.column_stack((image_points, np.ones(len(image_points))))
        normalized_points = (intrinsics_inv @ image_points_homogeneous.T).T

        # Scale the points by the depth
        scaled_points = normalized_points * depths[:, np.newaxis]

        # Convert to homogeneous coordinates
        scaled_points_homogeneous = np.column_stack((scaled_points, np.ones(len(scaled_points))))

        # Transform to robot coordinates
        world_points = (self.cam2ee @ scaled_points_homogeneous.T).T

        return world_points[:, :3]

    def pos_in_image_to_camera(self, pixel, depth):
        """Convert 2D pixels in image coordinate to 3D points in camera coordinate"""
        x = (pixel[0] - self.intrinsics[0][2]) * depth / self.intrinsics[0][0]
        y = (pixel[1] - self.intrinsics[1][2]) * depth / self.intrinsics[1][1]
        return np.array([x, y, depth])

    def _get_rigid_transform_error(self, z_scale):

        # Apply z offset and compute new observed points using camera intrinsics
        observed_z = self.observed_pts[:, 2:] * z_scale
        observed_x = np.multiply(self.observed_pix[:, [0]] - self.intrinsics[0][2], observed_z / self.intrinsics[0][0])
        observed_y = np.multiply(self.observed_pix[:, [1]] - self.intrinsics[1][2], observed_z / self.intrinsics[1][1])
        new_observed_pts = np.concatenate((observed_x, observed_y, observed_z), axis=1)

        # Estimate rigid transform between measured points and new observed points
        R, t = get_rigid_transform(np.asarray(self.measured_pts), np.asarray(new_observed_pts))
        t.shape = (3, 1)
        self.world2camera = np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)

        # Compute rigid transform error
        registered_pts = np.dot(R, np.transpose(self.measured_pts)) + np.tile(t, (1, self.measured_pts.shape[0]))
        error = np.transpose(registered_pts) - new_observed_pts
        error = np.sum(np.multiply(error, error))
        rmse = np.sqrt(error / self.measured_pts.shape[0])
        # print("RMSE: ", rmse)
        return rmse


if __name__ == "__main__":
    camera = Camera()