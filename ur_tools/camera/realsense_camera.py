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
        elif self.device_name == "Intel RealSense D435":
            preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
            for i in range(int(preset_range.max)):
                visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset, i)
                print(f"Visual preset {i}: {visulpreset}")
                if visulpreset == "Default":
                    print("Setting visual preset to Default")
                    depth_sensor.set_option(rs.option.visual_preset, i)
                    break
            color_sensor = profile.get_device().first_color_sensor()
            self.pipeline.stop() # You have to stop the pipeline before setting options
            while True:
                try:
                    color_sensor.set_option(rs.option.enable_auto_exposure, True)
                    color_sensor.set_option(rs.option.contrast, 100)
                    color_sensor.set_option(rs.option.exposure, 390.0)
                    # color_sensor.set_option(rs.option.power_line_frequency, 2)
                    color_sensor.set_option(rs.option.gain, 50)
                    color_sensor.set_option(rs.option.brightness, -30)
                    color_sensor.set_option(rs.option.gamma, 100)
                    break
                except Exception as e:
                    time.sleep(0.1)
                    continue

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

        profile = self.pipeline.start(config) # restart it after setting options
        # atexit.register(self.stop_streaming)
        # Give some time to be stable
        print("Give time for camera to warm up")
        start_time = time.time()
        # while time.time() - start_time < 0.1:
        for init_cnt in range(10):
            try:
                self.pipeline.wait_for_frames()
                break
            except Exception as e:
                time.sleep(0.1)
                if init_cnt == 9:
                    print("Camera initialization failed after 10 attempts.")
                    sys.exit(1)
                continue

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

    def show_video_realtime(self, show_depth=True, window_scale=0.8):
        """
        Show real-time video stream from the camera.
        
        Args:
            show_depth (bool): Whether to show depth image alongside color image
            window_scale (float): Scale factor for the display windows (0.1 to 1.0)
        
        Controls:
            - Press 'q' to quit
            - Press 's' to save current frame
            - Press 'r' to reset window positions
        """
        print("Starting real-time video stream...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'r' - Reset window positions")
        
        frame_count = 0
        
        try:
            while True:
                # Get frameset directly from pipeline for real-time performance
                frames = self.pipeline.wait_for_frames()
                
                # Apply filters based on device type
                if self.device_name in ["Intel RealSense D455", "Intel RealSense D435"]:
                    frames = self.decimation.process(frames).as_frameset()
                    frames = self.depth_to_disparity.process(frames).as_frameset()
                    frames = self.temporal.process(frames).as_frameset()
                    frames = self.disparity_to_depth.process(frames).as_frameset()
                
                # Align frames
                aligned_frames = self.align.process(frames)
                
                # Get color and depth frames
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Process depth image
                depth_image = depth_image * self.depth_scale
                depth_image = depth_image.astype(np.float32)
                depth_image[depth_image > self.clipping_distance_in_meters[1]] = self.clipping_distance_in_meters[1]
                depth_image[depth_image < self.clipping_distance_in_meters[0]] = 0
                
                # Normalize depth for visualization
                depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_normalized = np.uint8(depth_normalized)
                depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                
                # Resize images if needed
                if window_scale != 1.0:
                    height, width = color_image.shape[:2]
                    new_width = int(width * window_scale)
                    new_height = int(height * window_scale)
                    color_image = cv2.resize(color_image, (new_width, new_height))
                    depth_colormap = cv2.resize(depth_colormap, (new_width, new_height))
                
                # Display color image
                cv2.imshow('RealSense Color Stream', color_image)
                
                # Display depth image if requested
                if show_depth:
                    cv2.imshow('RealSense Depth Stream', depth_colormap)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting video stream...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    color_filename = f"color_frame_{timestamp}.png"
                    depth_filename = f"depth_frame_{timestamp}.png"
                    
                    cv2.imwrite(color_filename, color_image)
                    if show_depth:
                        cv2.imwrite(depth_filename, depth_colormap)
                    
                    print(f"Saved frame: {color_filename}")
                    if show_depth:
                        print(f"Saved depth: {depth_filename}")
                elif key == ord('r'):
                    # Reset window positions
                    cv2.destroyAllWindows()
                    print("Reset window positions")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during video streaming: {e}")
        finally:
            cv2.destroyAllWindows()
            print(f"Video stream ended. Total frames: {frame_count}")

    def show_video_with_aruco(self, aruco_dict_type=cv2.aruco.DICT_4X4_250, window_scale=0.8):
        """
        Show real-time video stream with ArUco marker detection overlay.
        
        Args:
            aruco_dict_type: ArUco dictionary type for detection
            window_scale (float): Scale factor for the display windows
        
        Controls:
            - Press 'q' to quit
            - Press 's' to save current frame with detections
            - Press 'd' to toggle detection overlay
        """
        print("Starting real-time video stream with ArUco detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame with detections")
        print("  'd' - Toggle detection overlay")
        
        # Setup ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        
        frame_count = 0
        show_detections = True
        
        try:
            while True:
                # Get current frame
                color_image, depth_image = self.get_data()
                
                # Convert RGB to BGR for OpenCV
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                
                if show_detections:
                    # Detect ArUco markers
                    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                    corners, ids, rejected = aruco_detector.detectMarkers(gray)
                    
                    # Draw detected markers
                    if ids is not None and len(ids) > 0:
                        cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                        
                        # Draw marker centers and IDs
                        for i, marker_id in enumerate(ids.flatten()):
                            marker_corners = corners[i][0]
                            marker_center = np.mean(marker_corners, axis=0).astype(int)
                            
                            # Draw center point
                            cv2.circle(color_image, tuple(marker_center), 5, (0, 0, 255), -1)
                            
                            # Draw ID text
                            cv2.putText(color_image, f'ID: {marker_id}', 
                                      (marker_center[0] + 10, marker_center[1] - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Display detection count
                        cv2.putText(color_image, f'Detected: {len(ids)} markers', 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(color_image, 'No markers detected', 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Resize if needed
                if window_scale != 1.0:
                    height, width = color_image.shape[:2]
                    new_width = int(width * window_scale)
                    new_height = int(height * window_scale)
                    color_image = cv2.resize(color_image, (new_width, new_height))
                
                # Display
                cv2.imshow('RealSense with ArUco Detection', color_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting video stream...")
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"aruco_detection_{timestamp}.png"
                    cv2.imwrite(filename, color_image)
                    print(f"Saved frame with detections: {filename}")
                elif key == ord('d'):
                    show_detections = not show_detections
                    status = "ON" if show_detections else "OFF"
                    print(f"Detection overlay: {status}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during video streaming: {e}")
        finally:
            cv2.destroyAllWindows()
            print(f"Video stream ended. Total frames: {frame_count}")


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
    camera.show_video_realtime()