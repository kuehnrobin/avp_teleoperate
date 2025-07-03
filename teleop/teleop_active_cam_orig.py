#!/usr/bin/env python3
"""
Active Camera Control using VR Head Tracking with DynamixelAgent
Reverted from dynamixel-api back to original DynamixelAgent approach to test servo path optimization
"""

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
from teleop.image_server.image_client import ImageClient
from teleop.robot_control.dynamixel.active_cam import DynamixelAgent

class ActiveCameraController:
    """Controller for the active camera servo system using DynamixelAgent."""
    
    def __init__(self, port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3R4A5A-if00-port0", pitch_id=1, yaw_id=2):
        """Initialize the camera controller with two servos for pitch and yaw.
        
        Args:
            port: Serial port for the U2D2 interface
            pitch_id: Dynamixel ID for pitch servo (vertical movement)  
            yaw_id: Dynamixel ID for yaw servo (horizontal movement)
        """
        self.port = port
        self.pitch_id = pitch_id
        self.yaw_id = yaw_id
        
        # Starting positions in radians (safe positions: pitch=-175.08°, yaw=86.04°)
        # self.start_positions = np.array([-1.58 * np.pi / 180, 91.05 * np.pi / 180])
        self.start_positions = np.array([-175.08 * np.pi / 180, 86.04 * np.pi / 180])
        
        # Initialize DynamixelAgent
        self.agent = None
        self.connected = False
    
    def connect(self):
        """Connect to servos using DynamixelAgent."""
        try:
            self.agent = DynamixelAgent(port=self.port, start_joints=self.start_positions)
            self.agent._robot.set_torque_mode(True)
            self.connected = True
            logging.info(f"Connected to active camera servos on {self.port}")
            
            # Perform safety check on initial positions
            self.check_initial_positions(tolerance_deg=20.0)
            
            return True
        except Exception as e:
            logging.error(f"Failed to connect to servos: {e}")
            self.disconnect()
            raise
    
    def disconnect(self):
        """Disconnect from servos."""
        try:
            if self.agent and hasattr(self.agent, '_robot'):
                self.agent._robot.set_torque_mode(False)
                logging.info("Disabled servo torque and disconnected")
        except Exception as e:
            logging.warning(f"Error during disconnect: {e}")
        finally:
            self.connected = False
    
    def get_positions(self):
        """Get current positions of both servos in radians.
        
        Returns:
            tuple: (pitch_position, yaw_position) in radians
        """
        if not self.connected:
            raise RuntimeError("Not connected to servos")
        
        try:
            # Get joint state returns [pitch, yaw] in radians
            positions = self.agent.act({})
            return tuple(positions)
        except Exception as e:
            logging.error(f"Error reading servo positions: {e}")
            raise # Re-raise the exception to be handled by the caller
    
    def set_positions(self, pitch_rad, yaw_rad):
        """Set target positions for both servos.
        
        Args:
            pitch_rad: Target pitch position in radians
            yaw_rad: Target yaw position in radians
        """
        if not self.connected:
            raise RuntimeError("Not connected to servos")
        
        try:
            # Command joint state with [pitch, yaw] in radians
            target_positions = [pitch_rad, yaw_rad]
            self.agent._robot.command_joint_state(target_positions)
        except Exception as e:
            logging.error(f"Error setting servo positions: {e}")
    
    def check_initial_positions(self, tolerance_deg=20.0):
        """
        Check if the initial positions of the servos are within a safe tolerance.
        If not, raises a RuntimeError to stop the script.
        """
        if not self.connected:
            raise RuntimeError("Cannot check initial positions, not connected to servos.")

        logging.info("Checking initial servo positions for safety...")
        
        try:
            current_positions_rad = self.get_positions()
            current_positions_deg = np.rad2deg(current_positions_rad)
            start_positions_deg = np.rad2deg(self.start_positions)

            pitch_diff = abs(current_positions_deg[0] - start_positions_deg[0])
            yaw_diff = abs(current_positions_deg[1] - start_positions_deg[1])

            if pitch_diff > tolerance_deg or yaw_diff > tolerance_deg:
                error_msg = (
                    f"\n!!! SAFETY ALERT: SERVO POSITION OUT OF TOLERANCE !!!\n"
                    f"Initial servo position deviates by more than {tolerance_deg}° from the expected start.\n"
                    f"------------------------------------------------------------------------------------\n"
                    f"Pitch Servo (ID {self.pitch_id}):\n"
                    f"  - Current Position: {current_positions_deg[0]:.2f}°\n"
                    f"  - Expected Start:   {start_positions_deg[0]:.2f}°\n"
                    f"  - Deviation:        {pitch_diff:.2f}°\n"
                    f"Yaw Servo (ID {self.yaw_id}):\n"
                    f"  - Current Position: {current_positions_deg[1]:.2f}°\n"
                    f"  - Expected Start:   {start_positions_deg[1]:.2f}°\n"
                    f"  - Deviation:        {yaw_diff:.2f}°\n"
                    f"------------------------------------------------------------------------------------\n"
                    f"This may indicate a physical obstruction or a desynchronization. "
                    f"Please check the hardware before restarting.\n"
                )
                logging.error(error_msg)
                raise RuntimeError("Initial servo position check failed. Aborting for safety.")
            
            logging.info("Initial servo positions are within safe limits. Continuing.")

        except Exception as e:
            logging.error(f"Failed to perform initial position check: {e}")
            raise

    def is_moving(self):
        """Check if either servo is currently moving.
        
        Returns:
            bool: True if any servo is moving (placeholder for now)
        """
        if not self.connected:
            return False
        
        # For now, assume movement check is not available in DynamixelAgent
        # Could be implemented by checking position changes over time
        return False
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

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
            logging.FileHandler(os.path.join(log_dir, "teleop_active_cam_orig.log")),
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
    
    return logging.getLogger('teleop_active_cam_orig')

def main():
    parser = argparse.ArgumentParser(description="Active Camera Control using VR Head Tracking - DynamixelAgent Version")
    parser.add_argument('--port', type=str, default="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3R4A5A-if00-port0", 
                       help="Serial port for the Dynamixel servo controller")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--safe-mode', action='store_true', default=True, help='Enable safe mode with limited movement')
    parser.add_argument('--max-movement', type=float, default=30.0, help='Maximum movement in degrees from start position (default: 30.0° for safety)')
    parser.add_argument('--use-opencv', action='store_true', default=True, help='Use OpenCV camera instead of ZED')
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info(f"Starting active camera control with DynamixelAgent on port: {args.port}")

    # Image configuration for OpenCV camera
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 640],  # Single camera resolution
        'head_camera_id_numbers': [0],  # Use camera 0
    }
    
    # Calculate image shape and create shared memory
    tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)
    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)

    # Initialize image client
    img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name)
    image_receive_thread = Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.start()
    logger.info("Image receive thread started")

    # Initialize TeleVision wrapper (for VR head tracking)
    tv_wrapper = TeleVisionWrapper(False, tv_img_shape, tv_img_shm.name, ngrok=True)  # False = monocular
    logger.info("TeleVision wrapper initialized")

    # Initialize camera controller
    camera_controller = None
    
    try:
        # Connect to camera servos
        logger.info("Connecting to camera servos...")
        camera_controller = ActiveCameraController(port=args.port)
        camera_controller.connect()

        # Get initial positions from the controller
        start_pitch_rad, start_yaw_rad = camera_controller.start_positions
        logger.info(f"Initial servo positions (degrees): Pitch={np.rad2deg(start_pitch_rad):.2f}, Yaw={np.rad2deg(start_yaw_rad):.2f}")

        # Wait for the first head tracking data
        logger.info("Waiting for head tracking data...")
        while tv_wrapper.get_head_orientation() is None:
            time.sleep(0.1)
        
        # Get the initial head rotation to use as a reference
        initial_head_rotation = R.from_quat(tv_wrapper.get_head_orientation())
        logger.info("Initial head rotation captured. Press 'r' to start teleoperation.")

        # Wait for the user to press 'r' to start
        while True:
            # Display the camera feed with a prompt
            frame = tv_img_array.copy()
            prompt_text = "Press 'r' to start teleoperation"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(prompt_text, font, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, prompt_text, (text_x, text_y), font, 1, (0, 255, 0), 2)
            cv2.imshow('Active Camera Teleop', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                logger.info("'r' pressed, starting teleoperation.")
                break
            elif key == ord('q'):
                logger.info("'q' pressed, shutting down before start.")
                raise KeyboardInterrupt("Shutdown requested by user")

        while True:
            # Get current head rotation
            current_head_rotation_quat = tv_wrapper.get_head_orientation()
            if current_head_rotation_quat is None:
                time.sleep(0.01)
                continue
            current_head_rotation = R.from_quat(current_head_rotation_quat)
            
            # Calculate relative rotation from the initial orientation
            relative_rotation = current_head_rotation * initial_head_rotation.inv()
            euler_angles = relative_rotation.as_euler('xyz', degrees=True)
            
            # Extract pitch and yaw changes
            pitch_delta_deg = euler_angles[0]  # Rotation around X-axis
            yaw_delta_deg = euler_angles[1]    # Rotation around Y-axis
            
            # Apply a scaling factor to reduce sensitivity
            scaling_factor = 0.1 if args.safe_mode else 0.2
            
            # Calculate the desired total movement from the start position
            pitch_movement_deg = pitch_delta_deg * scaling_factor
            yaw_movement_deg = yaw_delta_deg * scaling_factor

            # Calculate the absolute target positions
            target_pitch_rad = start_pitch_rad + np.deg2rad(pitch_movement_deg)
            target_yaw_rad = start_yaw_rad - np.deg2rad(yaw_movement_deg)  # Yaw is inverted

            # Enforce absolute limits to create a "virtual cage"
            max_mov_rad = np.deg2rad(args.max_movement)
            min_pitch = start_pitch_rad - max_mov_rad
            max_pitch = start_pitch_rad + max_mov_rad
            min_yaw = start_yaw_rad - max_mov_rad
            max_yaw = start_yaw_rad + max_mov_rad
            
            # Clip the target positions to the absolute limits
            target_pitch_rad = np.clip(target_pitch_rad, min_pitch, max_pitch)
            target_yaw_rad = np.clip(target_yaw_rad, min_yaw, max_yaw)

            # Set the new target positions for the servos
            camera_controller.set_positions(target_pitch_rad, target_yaw_rad)
            
            # Optional: Log the target positions for debugging
            if args.verbose:
                logger.debug(f"Target (deg): Pitch={np.rad2deg(target_pitch_rad):.2f}, Yaw={np.rad2deg(target_yaw_rad):.2f}")

            # Display the camera feed
            cv2.imshow('Active Camera Teleop', tv_img_array)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("'q' pressed, shutting down.")
                break
            
            time.sleep(0.02) # Loop at ~50Hz

    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, shutting down.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    
    finally:
        # Cleanup
        logger.info("Shutting down...")
        
        if camera_controller:
            camera_controller.disconnect()
        
        cv2.destroyAllWindows()
        
        try:
            tv_img_shm.unlink()
            tv_img_shm.close()
        except:
            pass
        
        logger.info("Cleanup complete")

if __name__ == "__main__":
    main()
