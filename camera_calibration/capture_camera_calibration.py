# Import necessary libraries
import cv2 as cv
import os
import pyrealsense2 as rs
import numpy as np

# Define chessboard dimensions
CHESS_BOARD_DIM = (9, 6)
# Initialize image counter
n = 0
# Define the image directory path
image_dir_path = "images"

# Check if the directory exists, if not, create it
CHECK_DIR = os.path.isdir(image_dir_path)
if not CHECK_DIR:
    os.makedirs(image_dir_path)
    print(f'"{image_dir_path}" Directory is created')
else:
    print(f'"{image_dir_path}" Directory already Exists.')

# Set termination criteria for corner detection
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Function to detect chessboard corners in the image


def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv.findChessboardCorners(grayImage, boardDimension)
    if ret == True:
        corners1 = cv.cornerSubPix(
            grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)

    return image, ret


# Set up RealSense camera pipeline and enable color stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
profile = pipeline.start(config)

try:
    while True:
        # Get color frames from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            print(
                "Error: Frame is None. Check if the camera is properly connected and working.")
            continue

        # Convert the color frame to a NumPy array
        frame = np.asanyarray(color_frame.get_data())
        copyFrame = frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect chessboard corners in the image
        image, board_detected = detect_checker_board(
            frame, gray, criteria, CHESS_BOARD_DIM)

        # Display the image counter on the frame
        cv.putText(
            frame,
            f"saved_img : {n}", (30, 40), cv.FONT_HERSHEY_PLAIN, 1.4, (0,
                                                                       255, 0), 2, cv.LINE_AA,
        )

        # Show the original frame and the frame with the detected chessboard corners
        cv.imshow("frame", frame)
        cv.imshow("copyFrame", copyFrame)

        # Check for key presses
        key = cv.waitKey(1)

        if key == ord("q"):
            break
        if key == ord("s") and board_detected == True:
            # Save the image with the detected chessboard pattern
            cv.imwrite(f"{image_dir_path}/image{n}.png", copyFrame)
            n += 1
except Exception as e:
    print(e)
finally:
    # Stop the RealSense camera pipeline
    pipeline.stop()
# Close all windows
cv.destroyAllWindows()

print("Total saved images:", n)