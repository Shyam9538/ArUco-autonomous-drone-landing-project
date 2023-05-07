# This file contains the ArucoTracker class for tracking ArUco markers and sending the pose to Pixhawk via Mavlink.
# Imports necessary modules and libraries
import numpy as np
import cv2
import cv2.aruco as aruco
from cv2 import FONT_HERSHEY_PLAIN
import sys
import time
import math
import pyrealsense2 as rs
from pymavlink import mavutil

calib_data_path = "/home/console/Downloads/calib_data/MultiMatrix.npz"

# Define a class for Aruco marker tracking.
class ArucoTracker():
    # Initialize the tracker with the id of the marker to track, the size of the marker in meters, the camera matrix, the camera distortion coefficients, the size of the camera image, whether to show the video, the connection string to the pixhawk, and the pipeline for the realsense camera.
    def __init__(self, id_to_find, marker_size, camera_matrix, camera_distortion, camera_size=None, show_video=False, connection_string=None, pipeline=None):
        if camera_size is None:
            camera_size = [1280, 720]
        self.id_to_find = id_to_find
        self.marker_size = marker_size
        self._show_video = show_video
        self._camera_matrix = camera_matrix
        self._camera_distortion = camera_distortion
        self.pipeline = pipeline
        self.align = rs.align(rs.stream.color)
        self.vehicle = None
        self.is_detected = False
        self._kill = False

        self._R_flip = np.zeros((3, 3), dtype=np.float32)
        self._R_flip[0, 0] = 1.0
        self._R_flip[1, 1] = -1.0
        self._R_flip[2, 2] = -1.0

        self._aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
        self._parameters = aruco.DetectorParameters_create()

        self.font = cv2.FONT_HERSHEY_PLAIN

        self._t_read = time.time()
        self._t_detect = self._t_read
        self.fps_read = 0.0
        self.fps_detect = 0.0


    # Method to convert rotation matrix to Euler angles.
    def rotationMatrixToEulerAngles(self, R):

        # Define a nested function to check if the matrix is a rotation matrix.
        def isRotationMatrix(R):
            Rt = np.transpose(R)
            shouldBeIdentity = np.dot(Rt, R)
            I = np.identity(3, dtype=R.dtype)
            n = np.linalg.norm(I - shouldBeIdentity)
            return n < 1e-6
        assert (isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    # Method to update FPS reading
    def update_fps_read(self):
        t = time.time()
        self.fps_read = 1.0/(t - self._t_read)
        self._t_read = t

    # Method to update FPS detection
    def update_fps_detect(self):
        t = time.time()
        self.fps_detect = 1.0/(t - self._t_detect)
        self._t_detect = t

    # Method to read the RGB-D frames from the camera and return the color and depth images.
    def read_rgbd_frame(self):
        aligned_image = False
        while not aligned_image:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            else:
                aligned_image = True
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
        return True, color_image, depth_image

    # Method to get the depth of the marker from the depth image and the corners of the marker.
    def get_marker_depth(self, corners, depth_image):
        c = corners[0][0]
        center_x = int((c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4)
        center_y = int((c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4)
        depth = depth_image[center_y, center_x]
        return depth, center_x, center_y

    # Method to send the detected pose to Pixhawk via Mavlink.
    def send_pose_to_pixhawk(self, x, y, z, roll, pitch, yaw):
        if self.vehicle is not None:
            try:
                msg = self.vehicle.message_factory.vision_position_estimate_encode(
                    int(time.time() * 1e6),
                    x,
                    y,
                    z,
                    roll,
                    pitch,
                    yaw
                )
                self.vehicle.send_mavlink(msg)
            except Exception as e:
                print("Error sending pose data to Pixhawk:", e)

    # Method to track the ArUco marker.
    def track(self, loop=True, verbose=False, show_video=None, vehicle=None):
        self._kill = False
        if show_video is None:
            show_video = self._show_video
        marker_found = False
        x = y = z = 0
        font = cv2.FONT_HERSHEY_PLAIN

        while not self._kill:
            ret, frame, depth_image = self.read_rgbd_frame()
            self.update_fps_read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = aruco.detectMarkers(
                image=gray, dictionary=self._aruco_dict, parameters=self._parameters)

            if ids is not None and self.id_to_find in ids[0]:
                self.update_fps_detect()
                ret = aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self._camera_matrix, self._camera_distortion)
                rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]
                x = tvec[0]
                y = tvec[1]
                z = tvec[2]
                marker_found = True

                if marker_found:
                    depth, center_x, center_y = self.get_marker_depth(
                        corners, depth_image)
                    print("Marker detected")
                    str_depth = "Marker Depth = %d" % depth
                    cv2.putText(frame, str_depth, (0, 300),
                                self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

                aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, self._camera_matrix,
                                  self._camera_distortion, rvec, tvec, 10)
                R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
                R_tc = R_ct.T
                roll_marker, pitch_marker, yaw_marker = self.rotationMatrixToEulerAngles(
                    self._R_flip*R_tc)
                pos_camera = -R_tc*np.matrix(tvec).T

                if verbose == True:
                    print("Marker X = %.1f  Y = %.1f  Z = %.1f  - fps = %.0f" %
                          (tvec[0], tvec[1], tvec[2], self.fps_detect))

                if show_video:
                    str_position = "Marker Position x=%4.0f  y=%4.0f  z=%4.0f" % (
                        tvec[0], tvec[1], tvec[2])
                    cv2.putText(frame, str_position, (0, 100),
                                font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    str_attitude = "Marker Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (math.degrees(
                        roll_marker), math.degrees(pitch_marker), math.degrees(yaw_marker))
                    cv2.putText(frame, str_attitude, (0, 150),
                                font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    str_position = "Camera Position x=%4.0f  y=%4.0f  z=%4.0f" % (
                        pos_camera[0], pos_camera[1], pos_camera[2])
                    cv2.putText(frame, str_position, (0, 200),
                                font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    roll_camera, pitch_camera, yaw_camera = self.rotationMatrixToEulerAngles(
                        self._R_flip*R_tc)
                    str_attitude = "Camera Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (math.degrees(roll_camera), math.degrees(pitch_camera),
                                                                                  math.degrees(yaw_camera))
                    cv2.putText(frame, str_attitude, (0, 250),
                                font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            elif verbose:
                print("Nothing detected - fps = %.0f" % self.fps_read)

            if show_video:
                cv2.imshow('frame', frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    self._cap.release()
                    self.pipeline.stop()
                    cv2.destroyAllWindows()
                    break
                elif key == ord('h'):
                    self._kill = True
                    self.pipeline.stop()

            if not loop:
                return (marker_found, x, y, z)


if __name__ == "__main__":
    # Initialize the Intel RealSense camera pipeline and enable color and depth streams to be read.
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_profile = config.resolve(pipeline)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    found_rgb = any(
        s.get_info(rs.camera_info.name) == "RGB Camera" for s in device.sensors
    )
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == "L500":
        config.enable_stream(
            rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    # Set marker parameters
    id_to_find = 3
    marker_size = 14  # - [cm]

    # Load calibration data
    calib_data = np.load(calib_data_path)
    connection_string = "/dev/ttyACM0"
    camera_distortion = calib_data["distCoef"]
    camera_matrix = calib_data["camMatrix"]
    r_vectors = calib_data["rVector"]
    t_vectors = calib_data["tVector"]

    # Initialize ArucoTracker
    aruco_tracker = ArucoTracker(id_to_find=id_to_find, marker_size=marker_size, show_video=True,
                                 camera_matrix=camera_matrix, camera_distortion=camera_distortion, connection_string=connection_string, pipeline=pipeline)

    # Main loop for tracking the marker
    while True:
        marker_found, x, y, z = aruco_tracker.track(verbose=True, loop=False)
        if marker_found:
            print(
                f"Marker {id_to_find} found at position (x, y, z): {x:.2f}, {y:.2f}, {z:.2f}")
        time.sleep(0.1)