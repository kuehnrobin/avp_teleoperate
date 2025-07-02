import numpy as np
import logging
import os
import sys
import time
import cv2
import argparse
from multiprocessing import shared_memory
from threading import Thread
from scipy.spatial.transform import Rotation as R

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.open_television.tv_wrapper import TeleVisionWrapper
from teleop.robot_control.dynamixel.active_cam import DynamixelAgent
from teleop.image_server.image_client import ImageClient

# Configure logging
def setup_logging(verbose=False):
    """Set up logging for the application"""
    log_dir = os.path.join(current_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure basic logging
    log_level = logging.INFO if not verbose else logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "teleop_active_cam.log")),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    # Only show warnings and errors on console by default
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING if not verbose else logging.DEBUG)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Replace the console handler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger('teleop_active_cam')

def main():
    parser = argparse.ArgumentParser(description="Active Camera Control using VR Head Tracking")
    parser.add_argument('--port', type=str, default="/dev/ttyUSB0", help="Serial port for the Dynamixel servo controller")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("Starting Active Camera Head Tracking System")

    # Initialize Dynamixel servo controller for the active camera platform
    try:
        agent = DynamixelAgent(port=args.port)
        agent._robot.set_torque_mode(True)
        logger.info(f"Dynamixel controller initialized on port {args.port}")
    except Exception as e:
        logger.error(f"Failed to initialize Dynamixel controller: {e}")
        return

    # Image configuration - using the same settings as in teleop_hand_and_arm.py
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 1280],  # [1080, 3840], #[480, 1280]
        'head_camera_id_numbers': [6],
        'wrist_camera_type': 'opencv',
        'wrist_camera_image_shape': [480, 640],
        'wrist_camera_id_numbers': [10, 12],
    }

    # Determine if using binocular setup
    ASPECT_RATIO_THRESHOLD = 2.0
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False

    # Setup image dimensions
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    # Create shared memory for the image
    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)
    
    # Initialize the TeleVision wrapper to get head movements
    tv_wrapper = TeleVisionWrapper(BINOCULAR, tv_img_shape, tv_img_shm.name, ngrok=True)
    logger.info("TeleVision wrapper initialized for head tracking")
    
    # Reuse the same image client as in teleop_hand_and_arm.py
    # The active camera will show what the camera sees (we're just moving the camera platform)
    img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name)
    image_receive_thread = Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.start()
    logger.info("Image receive thread started")

    # Main control loop
    try:
        logger.info("Starting active camera control. Press Ctrl+C to exit.")
        
        while True:
            start_time = time.time()
            
            # Get data from the VR headset
            head_rmat, _, _, _, _ = tv_wrapper.get_data()
            
            # Convert rotation matrix to Euler angles
            rot = R.from_matrix(head_rmat)
            euler_angles = rot.as_euler('xyz', degrees=True)
            
            # Extract yaw and pitch for camera movement (ignoring roll)
            # Note: You may need to adjust these values based on your specific setup
            yaw, pitch = euler_angles[1], euler_angles[0]
            
            # Apply scaling and limits to control camera movement sensitivity
            yaw_scaled = np.clip(yaw * 0.5, -45, 45)   # Scale and limit yaw movement
            pitch_scaled = np.clip(pitch * 0.5, -30, 30)  # Scale and limit pitch movement
            
            logger.debug(f"Head orientation - Yaw: {yaw_scaled:.2f}°, Pitch: {pitch_scaled:.2f}°")
            
            try:
                # Send commands to Dynamixel servos to move the camera
                agent._robot.command_joint_state([yaw_scaled, pitch_scaled])
            except Exception as e:
                logger.warning(f"Failed to command servos: {e}")
            
            # Display video feed (optional)
            try:
                if np.any(tv_img_array):
                    resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                    cv2.imshow("Active Camera View", resized_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
            except Exception as e:
                logger.warning(f"Error displaying video: {e}")
            
            # Maintain target frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, 1.0/30.0 - elapsed)  # Target 30fps
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Periodically log frame rate
            if logger.isEnabledFor(logging.DEBUG) and (int(time.time()) % 5 == 0):
                fps = 1.0 / (time.time() - start_time)
                logger.debug(f"Frame rate: {fps:.2f} fps")
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Clean up resources
        agent._robot.set_torque_mode(False)
        cv2.destroyAllWindows()
        tv_img_shm.close()
        tv_img_shm.unlink()
        logger.info("Resources cleaned up, exiting.")

if __name__ == "__main__":
    main()