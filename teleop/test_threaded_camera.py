#!/usr/bin/env python3
"""
Test script for the threaded ActiveCameraController
This demonstrates the new API usage pattern.
"""

import numpy as np
import logging
import os
import sys
import time
from multiprocessing import shared_memory

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.open_television.tv_wrapper import TeleVisionWrapper
from teleop.image_server.image_client import ImageClient
from teleop.robot_control.active_head_cam import ActiveCameraController

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('test_threaded_camera')
    
    # Image configuration (similar to teleop_hand_and_arm.py)
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 1280],
        'head_camera_id_numbers': [6],
    }
    
    tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)
    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    
    # Initialize TeleVision wrapper
    tv_wrapper = TeleVisionWrapper(False, tv_img_shape, tv_img_shm.name, ngrok=True)
    logger.info("TeleVision wrapper initialized")
    
    # Initialize camera controller with threading support
    camera_controller = None
    
    try:
        logger.info("Initializing threaded camera controller...")
        camera_controller = ActiveCameraController(
            port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3R4A5A-if00-port0",
            safe_mode=True,  # Enable safe mode for testing
            max_movement_deg=30.0,  # Conservative limit for testing
            logger=logger
        )
        
        # Connect to servos
        logger.info("Connecting to servos...")
        if not camera_controller.connect():
            logger.error("Failed to connect to servos")
            return
        
        # Enable head tracking (this starts the thread)
        logger.info("Enabling head tracking...")
        if not camera_controller.enable_head_tracking(tv_wrapper):
            logger.error("Failed to enable head tracking")
            return
        
        logger.info("Head tracking enabled! Move your head to test the servos.")
        logger.info("The camera should now follow your head movements automatically.")
        logger.info("Press Ctrl+C to stop...")
        
        # Main loop - just monitor the servo states
        frame_count = 0
        while True:
            # Get servo states for monitoring/recording
            servo_states = camera_controller.get_servo_states()
            
            # Log servo positions every 2 seconds (60 frames at 30Hz)
            if frame_count % 60 == 0:
                logger.info(f"Servo States:")
                logger.info(f"  Current: Pitch={np.rad2deg(servo_states['current_pitch']):.2f}°, "
                           f"Yaw={np.rad2deg(servo_states['current_yaw']):.2f}°")
                logger.info(f"  Target:  Pitch={np.rad2deg(servo_states['target_pitch']):.2f}°, "
                           f"Yaw={np.rad2deg(servo_states['target_yaw']):.2f}°")
            
            frame_count += 1
            time.sleep(1/30.0)  # 30Hz monitoring loop
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Cleanup
        if camera_controller:
            camera_controller.disconnect()  # This also stops the thread
        
        try:
            tv_img_shm.unlink()
            tv_img_shm.close()
        except:
            pass
        
        logger.info("Cleanup complete")

if __name__ == "__main__":
    main()
