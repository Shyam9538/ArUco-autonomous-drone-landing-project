Camera Calibration using Chessboard Pattern
This repository contains code to calibrate a camera using a chessboard pattern. The code is split into two main parts:

Capturing images of a chessboard pattern with a RealSense camera.
Camera calibration using the captured images.
Requirements
Python 3.x
OpenCV
NumPy
pyrealsense2 (only required for capturing images with a RealSense camera)

You can install the required packages using pip:
pip install opencv-python-headless numpy pyrealsense2

Usage
Place a chessboard pattern in front of the camera. Ensure that the chessboard has the dimensions specified in the CHESS_BOARD_DIM variable (default is 9x6).

Run the first part of the code to capture images of the chessboard pattern. Press the 's' key to save an image with the detected chessboard pattern, and press the 'q' key to quit the program.

python capture_images.py

Once you have captured enough images, run the second part of the code to calibrate the camera using the images.

python calibrate_camera.py

The calibration data, including the camera matrix, distortion coefficients, rotation vectors, and translation vectors, will be saved in the calib_data folder as MultiMatrix.npz.