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
    parser.add_argument('--safe-mode', action='store_true', default=True, help='Enable safe mode with limited movement')
    parser.add_argument('--max-movement', type=float, default=10.0, help='Maximum movement in degrees from start position')
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("Starting Active Camera Head Tracking System")

    # Safe starting positions (in degrees) - these are the calibrated safe positions
    # ID 1 (vertical/pitch): 18.54°, ID 2 (horizontal/yaw): 91.41°
    START_PITCH_DEG = 18.54  # Servo ID 1 - vertical movement (pitch)
    START_YAW_DEG = 91.41    # Servo ID 2 - horizontal movement (yaw)
    
    # Convert to radians for the servo commands
    start_pitch_rad = np.deg2rad(START_PITCH_DEG)
    start_yaw_rad = np.deg2rad(START_YAW_DEG)
    start_joints = np.array([start_pitch_rad, start_yaw_rad])
    
    # Safety limits (in degrees from start position)
    max_movement_deg = args.max_movement
    if args.safe_mode:
        max_movement_deg = min(max_movement_deg, 5.0)  # Extra conservative in safe mode
        logger.info(f"Safe mode enabled - limiting movement to ±{max_movement_deg}° from start position")
    
    logger.info(f"Starting positions - Pitch: {START_PITCH_DEG}°, Yaw: {START_YAW_DEG}°")
    logger.info(f"Movement limits: ±{max_movement_deg}° from start position")

    # Initialize Dynamixel servo controller for the active camera platform
    try:
        agent = DynamixelAgent(port=args.port, start_joints=start_joints)
        agent._robot.set_torque_mode(True)
        
        # Move to starting position slowly and safely
        logger.info("Moving to safe starting position...")
        agent._robot.command_joint_state(start_joints)
        time.sleep(2.0)  # Wait for servos to reach position
        
        logger.info(f"Dynamixel controller initialized on port {args.port}")
        logger.info("Servos positioned at safe starting position")
    except Exception as e:
        logger.error(f"Failed to initialize Dynamixel controller: {e}")
        return

    # Image configuration - using the same settings as in teleop_hand_and_arm.py
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 1280],  # [1080, 3840], #[480, 1280]
        'head_camera_id_numbers': [2],
        # 'wrist_camera_type': 'opencv',
        # 'wrist_camera_image_shape': [480, 640],
        # 'wrist_camera_id_numbers': [10, 12],
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
    img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name, server_address = "127.0.0.1")
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
            
            # Extract pitch and yaw for camera movement (ignoring roll)
            # euler_angles[0] = pitch (up/down), euler_angles[1] = yaw (left/right)
            head_pitch, head_yaw = euler_angles[0], euler_angles[1]
            
            # Apply scaling to reduce sensitivity and convert to movement relative to start position
            pitch_movement_deg = head_pitch * 0.3  # Scale down head movement
            yaw_movement_deg = head_yaw * 0.3      # Scale down head movement
            
            # Apply safety limits (movement from start position)
            pitch_movement_deg = np.clip(pitch_movement_deg, -max_movement_deg, max_movement_deg)
            yaw_movement_deg = np.clip(yaw_movement_deg, -max_movement_deg, max_movement_deg)
            
            # Calculate target positions (start position + movement)
            target_pitch_deg = START_PITCH_DEG + pitch_movement_deg
            target_yaw_deg = START_YAW_DEG + yaw_movement_deg
            
            # Convert to radians for servo commands
            target_pitch_rad = np.deg2rad(target_pitch_deg)
            target_yaw_rad = np.deg2rad(target_yaw_deg)
            
            # Command servos: [pitch (ID 1), yaw (ID 2)]
            target_joints = np.array([target_pitch_rad, target_yaw_rad])
            
            logger.debug(f"Head: pitch={head_pitch:.1f}°, yaw={head_yaw:.1f}° | "
                        f"Target: pitch={target_pitch_deg:.1f}°, yaw={target_yaw_deg:.1f}°")
            
            try:
                # Send commands to Dynamixel servos to move the camera
                agent._robot.command_joint_state(target_joints)
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