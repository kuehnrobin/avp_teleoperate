import math
import numpy as np

np.set_printoptions(precision=2, suppress=True)
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations

import time
import cv2
from constants_vuer import *
from TeleVision import OpenTeleVision
from dynamixel.active_cam import DynamixelAgent
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

# Configuration for stereo camera
resolution = (720, 1280)  # This will be used for each individual camera
crop_size_w = 1
crop_size_h = 0
resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)

# Initialize the Dynamixel servo controller (XL430-W250 is compatible with the existing driver)
# Just make sure the port is correct for your hardware setup
agent = DynamixelAgent(port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8IT033-if00-port0")
agent._robot.set_torque_mode(True)

# OpenCV stereo camera setup - replace ZED implementation
# Assuming camera IDs are 0 and 1 for left and right cameras, adjust if needed
left_cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
right_cam = cv2.VideoCapture(1, cv2.CAP_V4L2)

# Set camera properties for consistent image capture
for cam in [left_cam, right_cam]:
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[1])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[0])
    cam.set(cv2.CAP_PROP_FPS, 60)  # Set to 60fps to match original code

# Check if cameras are opened successfully
if not left_cam.isOpened() or not right_cam.isOpened():
    print("Error opening one or both cameras. Please check camera connections.")
    exit()


img_shape = (resolution_cropped[0], 2 * resolution_cropped[1], 3)
img_height, img_width = resolution_cropped[:2]
shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)
image_queue = Queue()
toggle_streaming = Event()
tv = OpenTeleVision(resolution_cropped, shm.name, image_queue, toggle_streaming)

print("Starting camera streaming and head tracking. Press Ctrl+C to exit.")

try:
    while True:
        start = time.time()

        # Get head orientation from the VR headset
        head_mat = grd_yup2grd_zup[:3, :3] @ tv.head_matrix[:3, :3] @ grd_yup2grd_zup[:3, :3].T
        if np.sum(head_mat) == 0:
            head_mat = np.eye(3)
        head_rot = rotations.quaternion_from_matrix(head_mat[0:3, 0:3])
        
        try:
            # Convert quaternion to yaw-pitch-roll (same as in original code)
            ypr = rotations.euler_from_quaternion(head_rot, 2, 1, 0, False)
            # Command the Dynamixel servos
            agent._robot.command_joint_state(ypr[:2])
        except:
            pass

        # Capture frames from both cameras
        ret_left, frame_left = left_cam.read()
        ret_right, frame_right = right_cam.read()
        
        if ret_left and ret_right:
            # Crop frames if needed (similar to the original cropping for ZED camera)
            frame_left_cropped = frame_left[crop_size_h:, crop_size_w:-crop_size_w]
            frame_right_cropped = frame_right[crop_size_h:, crop_size_w:-crop_size_w]
            
            # Concatenate frames horizontally
            bgr = np.hstack((frame_left_cropped, frame_right_cropped))
            
            # Convert to RGB (same format as expected by the rest of the pipeline)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            # Copy to shared memory for transmission to VR
            np.copyto(img_array, rgb)
        else:
            print("Error reading frames from cameras")
        
        end = time.time()
        # Optionally log frame rate
        # print(f"Frame rate: {1/(end-start):.2f} fps")

except KeyboardInterrupt:
    print("Terminating program...")
finally:
    # Clean up resources
    left_cam.release()
    right_cam.release()
    agent._robot.set_torque_mode(False)  # Safely turn off torque
    shm.close()
    shm.unlink()
    print("Resources cleaned up, exiting.")