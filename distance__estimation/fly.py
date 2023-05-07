# This file contains the main logic for the drone to fly to a target altitude and then to a target marker.
# Imports necessary modules and libraries
import asyncio
import time
import math
from os import path
import mavsdk
import numpy as np
from mavsdk import System
from mavsdk.mission import (MissionItem, MissionPlan)
from mavsdk.telemetry import (LandedState, PositionNed, FlightMode)
from mavsdk.offboard import (Attitude, OffboardError, VelocityNedYaw)
from mavsdk import telemetry
from pymavlink import mavutil
import cv2
from aruco_pose import ArucoTracker
import pyrealsense2 as rs
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

calib_data_path = "/home/console/Downloads/calib_data/MultiMatrix.npz"

# Arming and taking off the drone, making it reach a target altitude
async def arm_and_takeoff(drone, target_altitude, connection_string1, ascent_rate=0.05):
    global pipeline, depth_scale
    print("Arming...")

    await drone.action.arm()

    print("-- Setting initial setpoint")
    await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.0))

    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed with error code: \
              {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return

    align_to = rs.stream.color
    align = rs.align(align_to)

    print("Taking off...")

    current_altitude = 0
    target_reached = False
    ascent_started = False
    altitude = 0
    while not target_reached:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_center_pixel = depth_image[depth_frame.height //
                                         2, depth_frame.width // 2]

        await send_pose_to_pixhawk(connection_string1, drone, 0, 0, -altitude, 0, 0, 0)

        if not ascent_started or depth_center_pixel != 0:
            ascent_started = True
            altitude_difference = target_altitude - current_altitude
            current_altitude += altitude_difference * ascent_rate

            current_altitude = min(current_altitude, target_altitude)

            await drone.offboard.set_position_ned(PositionNed(0, 0, -current_altitude))
            await asyncio.sleep(1)

        altitude = depth_center_pixel * depth_scale
        print(f"Altitude: {altitude} meters")

        if altitude >= target_altitude * 0.95:
            print("Reached target altitude")
            target_reached = True
        else:
            altitude_difference = target_altitude - altitude
            current_altitude += altitude_difference * 0.1
            await drone.offboard.set_position_ned(PositionNed(0, 0, -current_altitude))

        await asyncio.sleep(1)

# Sends North, East, Down velocity commands to the drone
async def send_ned_velocity(drone, velocity_x, velocity_y, velocity_z, duration):
    await drone.offboard.set_velocity_ned(
        VelocityNedYaw(velocity_x, velocity_y, velocity_z, 0)
    )
    await asyncio.sleep(duration)

# Land the drone
async def land(drone):
    print("Landing...")
    await drone.action.land()

# Connects to the drone
async def connect_drone(connection_string):
    drone = System()
    await drone.connect(system_address=connection_string)

    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"Drone connected: {state.is_connected}")
            break
    return drone

# Sends pose data to the Pixhawk
async def send_pose_to_pixhawk(connection_string, drone, x, y, z, roll, pitch, yaw):
    if drone is not None:
        master = mavutil.mavlink_connection(connection_string)
        try:
            master.mav.vision_position_estimate_send(
                int(time.time() * 1e6), x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)
        except Exception as e:
            print("Error sending pose data to Pixhawk:", e)

# Initialize the Intel RealSense camera
def initialize_realsense():
    global pipeline, depth_scale
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

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)

    if device_product_line == "L500":
        config.enable_stream(
            rs.stream.color, 960, 540, rs.format.bgr8, 60)
    else:
        config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 60)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    return pipeline, depth_scale

# Convert marker position to angles
def marker_position_to_angle(x, y, z):
    angle_x = math.atan2(x, z)
    angle_y = math.atan2(y, z)
    return (angle_x, angle_y)

# Convert camera coordinates to UAV coordinates
def camera_to_uav(x_cam, y_cam):
    x_uav = x_cam
    y_uav = y_cam
    return (x_uav, y_uav)

# Convert UAV coordinates to North and East coordinates
def uav_to_ne(x_uav, y_uav, yaw_rad):
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    north = x_uav * c - y_uav * s
    east = x_uav * s + y_uav * c
    return (north, east)

# Check if the drone should descend based on angle
def check_angle_descend(angle_x, angle_y, angle_desc):
    return (math.sqrt(angle_x ** 2 + angle_y ** 2) <= angle_desc)

# Get depth from the Intel RealSense camera


def get_depth_from_realsense(pipeline, depth_scale):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_center_pixel = depth_image[depth_frame.height //
                                     2, depth_frame.width // 2]
    return depth_center_pixel*depth_scale*100

# Run the main function
async def main():
    global pipeline, depth_scale
    connection_string = "serial:///dev/ttyACM0:921600"
    connection_string1 = "/dev/ttyACM0, 921600"
    print('Connecting...')
    drone = await connect_drone(connection_string)

    pipeline, depth_scale = initialize_realsense()

    takeoff_altitude = 0.1  # - meters
    await arm_and_takeoff(drone, takeoff_altitude, connection_string1, ascent_rate=0.05)

    rad_2_deg = 180.0 / math.pi
    deg_2_rad = 1.0 / rad_2_deg

    id_to_find = 2
    marker_size = 14  # - cm
    freq_send = 50  # - Hz

    land_alt_cm = 30.0
    angle_descend = 20 * deg_2_rad
    land_speed_cms = 30.0

    cwd = path.dirname(path.abspath(__file__))

    calib_data = np.load(calib_data_path)
    print(calib_data.files)

    camera_matrix = calib_data["camMatrix"]
    camera_distortion = calib_data["distCoef"]
    r_vectors = calib_data["rVector"]
    t_vectors = calib_data["tVector"]
    aruco_tracker = ArucoTracker(id_to_find=id_to_find, marker_size=marker_size, show_video=False,
                                 camera_matrix=camera_matrix, camera_distortion=camera_distortion, pipeline=pipeline)

    time_0 = time.time()

    print("Marker tracking started")
    while True:

        marker_found, x_cm, y_cm, z_cm = aruco_tracker.track(loop=False)

        if marker_found:
            x_cm, y_cm = camera_to_uav(x_cm, y_cm)
            angle_x, angle_y = marker_position_to_angle(x_cm, y_cm, z_cm)

            if time.time() >= time_0 + 1.0 / freq_send:
                time_0 = time.time()

                z_cm = get_depth_from_realsense(pipeline, depth_scale)

                print("")
                print("Altitude = %.0fcm" % z_cm)
                print("Marker found x = %5.0f cm  y = %5.0f cm -> angle_x = %5f  angle_y = %5f" %
                      (x_cm, y_cm, angle_x * rad_2_deg, angle_y * rad_2_deg))

                async for current_attitude in drone.telemetry.attitude_euler():
                    yaw_rad = math.radians(current_attitude.yaw_deg)
                    roll_rad = math.radians(current_attitude.roll_deg)
                    pitch_rad = math.radians(current_attitude.pitch_deg)

                    break

                north, east = uav_to_ne(x_cm, y_cm, yaw_rad)
                print("Marker N = %5.0f cm   E = %5.0f cm   Yaw = %.0f deg" %
                      (north, east, yaw_rad * rad_2_deg))

                await send_pose_to_pixhawk(connection_string1, drone, north, east, -z_cm, roll_rad, -pitch_rad, -yaw_rad)

                if check_angle_descend(angle_x, angle_y, angle_descend):
                    print("Low error: descending")
                    vz = -(land_speed_cms * 0.01 / freq_send)
                elif z_cm >= takeoff_altitude:
                    vz = -0.2
                    east = 0
                    north = 0
                else:
                    vz = 0

                await send_ned_velocity(drone, east * 0.01, north * 0.01, vz, 1)

            if z_cm <= land_alt_cm and drone.telemetry.flight_mode() == FlightMode.OFFBOARD:
                print("Commanding to land")
                await land(drone)
                pipeline.stop()
        else:
            async for position_ned in drone.telemetry.position_velocity_ned():
                target_x, target_y, target_z = position_ned.position.north_m, position_ned.position.east_m, position_ned.position.down_m
                await drone.offboard.set_position_ned(PositionNed(target_x, target_y, target_z))
                break


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())