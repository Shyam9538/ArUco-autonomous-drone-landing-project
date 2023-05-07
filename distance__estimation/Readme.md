Drone Altitude and Marker Tracking

This repository contains code to control a drone using the MAVSDK-Python library, an Intel RealSense depth camera, and an ArUco marker. The drone flies to a specified target altitude and then tracks a target ArUco marker in its field of view. The script also allows the drone to land when the marker is detected at a certain distance and angle.


Requirements
Python 3.x
MAVSDK-Python
NumPy
OpenCV
pyrealsense2 (only required for using an Intel RealSense depth camera)
pymavlink

You can install the required packages using pip:
pip install mavsdk numpy opencv-python-headless pyrealsense2 pymavlink


Usage
Connect the drone to your computer using the appropriate connection method (e.g., USB or telemetry radios). Make sure the drone is powered on and has GPS lock.
Place an ArUco marker with the specified ID (default is 2) within the drone's field of view. The marker size should be specified in the script (default is 14 cm).

Run the script:
python fly.py

The drone will arm, take off to the target altitude (default is 0.1 m), and begin tracking the ArUco marker. The drone will descend and land when the marker is detected at a specific distance and angle.


Customization
You can customize the behavior of the drone by modifying the following variables in the main.py script:
takeoff_altitude: The target altitude for the drone to reach after takeoff (in meters).
id_to_find: The ID of the ArUco marker to track.
marker_size: The size of the ArUco marker (in centimeters).
land_alt_cm: The altitude at which the drone should land when the marker is detected (in centimeters).
angle_descend: The angle at which the drone should begin descending (in degrees).

The ArucoTracker class in the provided script can be used for tracking ArUco markers and sending the pose to Pixhawk via Mavlink. It is initialized with the marker ID, marker size, camera matrix, and other necessary parameters. The class provides methods to read RGB-D frames from the camera, estimate the marker's depth, and send the pose information to Pixhawk. The track() method can be used to track the ArUco marker in real-time.