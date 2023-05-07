# Import necessary libraries
import cv2 as cv
import os
import numpy as np

# Define chessboard dimensions
CHESS_BOARD_DIM = (9, 6)

# Define the size of each square in the chessboard in millimeters
SQUARE_SIZE = 23  # millimeters

# Set termination criteria for corner detection
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define the directory path for saving calibration data
calib_data_path = "../calib_data"
CHECK_DIR = os.path.isdir(calib_data_path)

# Check if the directory exists, if not, create it
if not CHECK_DIR:
    os.makedirs(calib_data_path)
    print(f'"{calib_data_path}" Directory is created')

else:
    print(f'"{calib_data_path}" Directory already Exists.')

# Create a 3D array of object points representing the chessboard corners
obj_3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)

obj_3D[:, :2] = np.mgrid[0: CHESS_BOARD_DIM[0], 0: CHESS_BOARD_DIM[1]].T.reshape(
    -1, 2
)
obj_3D *= SQUARE_SIZE
print(obj_3D)

# Initialize lists for storing 3D object points and 2D image points
obj_points_3D = []
img_points_2D = []

# Define the directory path for the images containing the chessboard pattern
image_dir_path = "images"

# Loop through the images in the directory
files = os.listdir(image_dir_path)
for file in files:
    print(file)
    imagePath = os.path.join(image_dir_path, file)

    # Read the image and convert it to grayscale
    image = cv.imread(imagePath)
    grayScale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners in the image
    ret, corners = cv.findChessboardCorners(image, CHESS_BOARD_DIM, None)
    if ret == True:
        # Add the 3D object points and 2D image points to their respective lists
        obj_points_3D.append(obj_3D)
        corners2 = cv.cornerSubPix(
            grayScale, corners, (3, 3), (-1, -1), criteria)
        img_points_2D.append(corners2)

        # Draw the detected chessboard corners on the image
        img = cv.drawChessboardCorners(image, CHESS_BOARD_DIM, corners2, ret)

cv.destroyAllWindows()

# Calibrate the camera using the 3D object points and 2D image points
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    obj_points_3D, img_points_2D, grayScale.shape[::-1], None, None
)
print("calibrated")

# Save the camera matrix, distortion coefficients, rotation vectors, and translation vectors to a file
np.savez(
    f"{calib_data_path}/MultiMatrix",
    camMatrix=mtx,
    distCoef=dist,
    rVector=rvecs,
    tVector=tvecs,
)

# Load the calibration data from the file
data = np.load(f"{calib_data_path}/MultiMatrix.npz")

camMatrix = data["camMatrix"]
distCof = data["distCoef"]
rVector = data["rVector"]
tVector = data["tVector"]

print("loaded calibration data successfully")