import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Array, Lock
import threading
import logging
import os

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.open_television.tv_wrapper import TeleVisionWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller, Gripper_Controller
from teleop.robot_control.robot_hand_inspire import Inspire_Controller
from teleop.robot_control.dynamixel.active_cam import DynamixelAgent
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.pose_logger import PoseLogger
from scipy.spatial.transform import Rotation as R

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
            logging.FileHandler(os.path.join(log_dir, "teleop.log")),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    # Set specific loggers to different levels
    # Keep the TV wrapper quiet unless in verbose mode
    logging.getLogger('tv_wrapper').setLevel(logging.WARNING if not verbose else logging.DEBUG)
    
    # Only show warnings and errors on console by default
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Replace the console handler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger('teleop')

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
        
        # Corrected starting positions in radians (Pitch: -175.08°, Yaw: 86.04°)
        self.start_positions = np.array([-175.08 * np.pi / 180, 86.04 * np.pi / 180])
        
        # Initialize DynamixelAgent
        self.agent = None
        self.connected = False
        self.logger = logging.getLogger('ActiveCameraController')
    
    def connect(self):
        """Connect to servos using DynamixelAgent and perform safety checks."""
        try:
            self.agent = DynamixelAgent(port=self.port, start_joints=self.start_positions)
            self.agent._robot.set_torque_mode(True)
            self.connected = True
            self.logger.info(f"Connected to active camera servos on {self.port}")

            # Perform safety check on initial positions
            self.check_initial_positions()

            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to servos: {e}")
            self.disconnect()
            raise # Re-raise to notify the main script

    def disconnect(self):
        """Disconnect from servos."""
        try:
            if self.agent and hasattr(self.agent, '_robot'):
                self.agent._robot.set_torque_mode(False)
                self.logger.info("Disabled servo torque and disconnected")
        except Exception as e:
            self.logger.warning(f"Error during disconnect: {e}")
        finally:
            self.connected = False
    
    def get_positions(self):
        """Get current positions of both servos in radians.
        
        Returns:
            tuple: (pitch_position, yaw_position) in radians
        """
        if not self.connected:
            return (0.0, 0.0)
        
        try:
            # Get joint state returns [pitch, yaw] in radians
            positions = self.agent.act({})
            return tuple(positions)
        except Exception as e:
            self.logger.error(f"Error reading servo positions: {e}")
            raise # Re-raise the exception to be handled by the caller

    def set_positions(self, pitch_rad, yaw_rad):
        """Set target positions for both servos.
        
        Args:
            pitch_rad: Target pitch position in radians
            yaw_rad: Target yaw position in radians
        """
        if not self.connected:
            return
        
        try:
            # Command joint state with [pitch, yaw] in radians
            target_positions = [pitch_rad, yaw_rad]
            self.agent._robot.command_joint_state(target_positions)
        except Exception as e:
            self.logger.error(f"Error setting servo positions: {e}")

    def check_initial_positions(self, tolerance_deg=20.0):
        """
        Check if the initial positions of the servos are within a safe tolerance.
        If not, raises a RuntimeError to stop the script.
        """
        if not self.connected:
            raise RuntimeError("Cannot check initial positions, not connected to servos.")

        self.logger.info("Checking initial servo positions for safety...")
        
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
                self.logger.error(error_msg)
                raise RuntimeError("Initial servo position check failed. Aborting for safety.")
            
            self.logger.info("Initial servo positions are within safe limits. Continuing.")

        except Exception as e:
            self.logger.error(f"Failed to perform initial position check: {e}")
            raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type = str, default = './utils/data', help = 'path to save data')
    parser.add_argument('--frequency', type = int, default = 30.0, help = 'save data\'s frequency')

    parser.add_argument('--record', action = 'store_true', help = 'Save data or not')
    parser.add_argument('--no-record', dest = 'record', action = 'store_false', help = 'Do not save data')
    parser.set_defaults(record = False)

    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--hand', type=str, choices=['dex3', 'gripper', 'inspire1'], help='Select hand controller')
    parser.add_argument('--retargeting-method', type=str, choices=['vector', 'dexpilot'], default='vector', 
                      help='Select hand retargeting method: vector (default) or dexpilot')

    parser.add_argument('--cyclonedds_uri', type=str, default='enxa0cec8616f27', help='Network interface for CycloneDX (default: enxa0cec8616f27)')
    # Speed Limit
    parser.add_argument('--arm-speed', type=float, default=None, 
                      help='Set the arm velocity limit (default is controller-specific)')
    parser.add_argument('--no-gradual-speed', action='store_true',
                      help='Disable gradual speed increase')
    
    # Active Camera options
    parser.add_argument('--active-camera', action='store_true', help='Enable active camera head tracking')
    parser.add_argument('--camera-port', type=str, default="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3R4A5A-if00-port0", 
                       help='Serial port for the active camera servo controller')
    parser.add_argument('--camera-safe-mode', action='store_true', default=True, help='Enable safe mode with limited camera movement')
    parser.add_argument('--camera-max-movement', type=float, default=1.0, help='Maximum camera movement in degrees from start position (default: 1.0° for safety)')
    
    # Logging options
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--no-pose-logging', dest='pose_logging', action='store_false', help='Disable background pose logging')
    parser.set_defaults(pose_logging=True)
    parser.add_argument('--force', action='store_true', help='If set, record real qvel/torque for arms and hands (default: False)')
    parser.set_defaults(force=False)

    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info(f"Starting teleop_hand_and_arm.py with args: {args}")

    # Initialize pose logger if enabled
    pose_logger = None
    if args.pose_logging:
        pose_logger = PoseLogger(
            log_dir=os.path.join(current_dir, "logs", "pose_data"),
            log_interval=1.0,  # Save to disk every 1 second
            max_buffer_size=100  # Or after 100 frames, whichever comes first
        ).start()
        logger.info(f"Background pose logging enabled, saving to {pose_logger.log_file}")

    # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 1280],# [1080, 3840], #[480, 1280],  # Head camera resolution
        'head_camera_id_numbers': [6],
        'wrist_camera_type': 'opencv',
        'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
        'wrist_camera_id_numbers': [8, 10],
    }
    ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    if 'wrist_camera_type' in img_config:
        WRIST = True
    else:
        WRIST = False
    
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)

    if WRIST:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name)
    else:
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)

    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()
    logger.info("Image receive thread started")

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    tv_wrapper = TeleVisionWrapper(BINOCULAR, tv_img_shape, tv_img_shm.name, ngrok=True) # True for quest3
    logger.info("TeleVision wrapper initialized")

    # arm
    if args.arm == 'G1_29':
        arm_ctrl = G1_29_ArmController(networkInterface=args.cyclonedds_uri)
        arm_ik = G1_29_ArmIK()
        if args.arm_speed is not None:
            arm_ctrl.arm_velocity_limit = args.arm_speed
            logger.info(f"Setting custom arm velocity limit: {args.arm_speed}")
    elif args.arm == 'G1_23':
        arm_ctrl = G1_23_ArmController(networkInterface=args.cyclonedds_uri)
        arm_ik = G1_23_ArmIK()
        if args.arm_speed is not None:
            arm_ctrl.arm_velocity_limit = args.arm_speed
    elif args.arm == 'H1_2':
        arm_ctrl = H1_2_ArmController(networkInterface=args.cyclonedds_uri)
        arm_ik = H1_2_ArmIK()
        if args.arm_speed is not None:
            arm_ctrl.arm_velocity_limit = args.arm_speed
    elif args.arm == 'H1':
        arm_ctrl = H1_ArmController()
        arm_ik = H1_ArmIK()
        if args.arm_speed is not None:
            arm_ctrl.arm_velocity_limit = args.arm_speed

    # active camera
    camera_controller = None
    if args.active_camera:
        try:
            logger.info(f"Initializing active camera controller on port: {args.camera_port}")
            camera_controller = ActiveCameraController(port=args.camera_port)
            if camera_controller.connect():
                logger.info("Active camera controller initialized successfully")
                # Get initial positions for reference
                initial_pitch, initial_yaw = camera_controller.get_positions()
                logger.info(f"Initial camera positions - Pitch: {np.degrees(initial_pitch):.1f}°, Yaw: {np.degrees(initial_yaw):.1f}°")
            else:
                logger.warning("Failed to connect to active camera controller")
                camera_controller = None
        except Exception as e:
            logger.error(f"Failed to initialize active camera controller: {e}")
            camera_controller = None

    # handbased on the 
    if args.hand == "dex3":
        # Dynamically set shared array size based on --force
        if args.force:
            hand_state_size = 66  # 33 for left, 33 for right
        else:
            hand_state_size = 38  # 19 for left, 19 for right
        left_hand_array = Array('d', 75, lock = True)         # [input]
        right_hand_array = Array('d', 75, lock = True)        # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', hand_state_size, lock = False)  # [output] current left, right hand state
        dual_hand_action_array = Array('d', 14, lock = False) # [output] current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, networkInterface=args.cyclonedds_uri, force=args.force, retargeting_method=args.retargeting_method)
    elif args.hand == "gripper":
        left_hand_array = Array('d', 75, lock=True)
        right_hand_array = Array('d', 75, lock=True)
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        gripper_ctrl = Gripper_Controller(left_hand_array, right_hand_array, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array, networkInterface=args.cyclonedds_uri)
    elif args.hand == "inspire1":
        left_hand_array = Array('d', 75, lock = True)          # [input]
        right_hand_array = Array('d', 75, lock = True)         # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
        dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
        hand_ctrl = Inspire_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, retargeting_method=args.retargeting_method)
    else:
        pass
    
    if args.record:
        recorder = EpisodeWriter(task_dir = args.task_dir, frequency = args.frequency, rerun_log = True)
        recording = False
        logger.info(f"Episode recorder initialized with task_dir={args.task_dir}") 
    try:
        user_input = input("Please enter the start signal (enter 'r' to start the subsequent program):\n")
        if user_input.lower() == 'r':
            if not args.no_gradual_speed:
                arm_ctrl.speed_gradual_max()
                logger.info("Gradual speed increase enabled")
            running = True
            frame_counter = 0
            
            logger.info("Starting main control loop")
            while running:
                start_time = time.time()
                
                # Get pose data
                head_rmat, left_wrist, right_wrist, left_hand, right_hand = tv_wrapper.get_data()
                frame_counter += 1
                
                # Log pose data if enabled
                if pose_logger:
                    pose_logger.log_pose(head_rmat, left_wrist, right_wrist, left_hand, right_hand)
                
                # Active camera control using head tracking
                if camera_controller and camera_controller.connected:
                    try:
                        # Convert head rotation matrix to Euler angles
                        rotation = R.from_matrix(head_rmat)
                        euler_angles = rotation.as_euler('xyz', degrees=False)
                        
                        # Extract pitch and yaw for camera movement (ignoring roll)
                        head_pitch = -euler_angles[0]  # Negative for intuitive control
                        head_yaw = euler_angles[1]
                        
                        # Apply movement scaling and safety limits
                        max_movement_rad = np.radians(args.camera_max_movement)
                        initial_pitch, initial_yaw = camera_controller.start_positions
                        
                        if args.camera_safe_mode:
                            # Apply scaling and limits for safe operation
                            target_pitch = np.clip(initial_pitch + head_pitch * 0.3, 
                                                 initial_pitch - max_movement_rad, 
                                                 initial_pitch + max_movement_rad)
                            target_yaw = np.clip(initial_yaw + head_yaw * 0.3, 
                                               initial_yaw - max_movement_rad, 
                                               initial_yaw + max_movement_rad)
                        else:
                            target_pitch = initial_pitch + head_pitch * 0.5
                            target_yaw = initial_yaw + head_yaw * 0.5
                        
                        # Send commands to camera servos
                        camera_controller.set_positions(target_pitch, target_yaw)
                        
                    except Exception as e:
                        logger.warning(f"Active camera control error: {e}")

                # send hand skeleton data to hand_ctrl.control_process
                if args.hand:
                    left_hand_array[:] = left_hand.flatten()
                    right_hand_array[:] = right_hand.flatten()

                # get current state data.
                current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
                current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

                # solve ik using motor data and wrist pose, then use ik results to control arms.
                time_ik_start = time.time()
                sol_q, sol_tauff  = arm_ik.solve_ik(left_wrist, right_wrist, current_lr_arm_q, current_lr_arm_dq)
                time_ik_end = time.time()
                # print(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

                # Log periodic IK timing information
                if frame_counter % 300 == 0:  # Every ~10 seconds at 30fps
                    logger.info(f"IK solve time: {(time_ik_end - time_ik_start)*1000:.2f}ms")

                tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                
                # Add recording status overlay
                if args.record:
                    status_text = ""
                    help_text = ""
                    if recording:
                        status_text = "RECORDING"
                        help_text = "Controls: [r]=abort | [q]=save optimal | [w]=save suboptimal | [e]=save recovery"
                    else:
                        status_text = "READY TO RECORD"
                        help_text = "Controls: [s]=start recording | [q]=quit (no recording) | [ESC]=quit"
                    
                    # Add status text overlay
                    cv2.putText(tv_resized_image, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if recording else (0, 255, 0), 2)
                    
                    # Add help text overlay
                    cv2.putText(tv_resized_image, help_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("record image", tv_resized_image)
                key = cv2.waitKey(1) & 0xFF
                
                # Handle key presses for recording control
                if key == 27 and not args.record:
                    running = False
                    logger.info("User requested exit with 'ESC' key")
                elif args.record:
                    if key == ord('s') and not recording:
                        # Start recording
                        if recorder.create_episode():
                            recording = True
                            logger.info("Started recording episode")
                        else:
                            logger.warning("Failed to create recording episode")
                    elif key == ord('r') and recording:
                        # Abort recording
                        recorder.abort_episode()
                        recording = False
                        logger.info("Aborted recording episode")
                    elif recording and key in [ord('q'), ord('w'), ord('e')]:
                        # Save with quality labels
                        quality_map = {ord('q'): 'optimal', ord('w'): 'suboptimal', ord('e'): 'recovery'}
                        quality = quality_map[key]
                        recorder.save_episode(quality=quality)
                        recording = False
                        logger.info(f"Saved recording episode with quality: {quality}")

                # record data
                if args.record:
                    # dex hand or gripper
                    if args.hand == "dex3":
                        with dual_hand_data_lock:
                            if args.force:
                                # [q0...q6, dq0...dq6, tau0...tau6, p0...p11] (33 each)
                                left_hand_state = dual_hand_state_array[0:7]
                                left_hand_vel = dual_hand_state_array[7:14]
                                left_hand_torque = dual_hand_state_array[14:21]
                                left_hand_pressures = dual_hand_state_array[21:33]
                                right_hand_state = dual_hand_state_array[33:40]
                                right_hand_vel = dual_hand_state_array[40:47]
                                right_hand_torque = dual_hand_state_array[47:54]
                                right_hand_pressures = dual_hand_state_array[54:66]
                            else:
                                # [q0...q6, p0...p11] (19 each)
                                left_hand_state = dual_hand_state_array[0:7]
                                left_hand_vel = []  # Not available without --force
                                left_hand_torque = []  # Not available without --force
                                left_hand_pressures = dual_hand_state_array[7:19]
                                right_hand_state = dual_hand_state_array[19:26]
                                right_hand_vel = []  # Not available without --force
                                right_hand_torque = []  # Not available without --force
                                right_hand_pressures = dual_hand_state_array[26:38]
                            left_hand_action = dual_hand_action_array[:7]
                            right_hand_action = dual_hand_action_array[-7:]
                    elif args.hand == "gripper":
                        with dual_gripper_data_lock:
                            left_hand_state = [dual_gripper_state_array[1]]
                            right_hand_state = [dual_gripper_state_array[0]]
                            left_hand_action = [dual_gripper_action_array[1]]
                            right_hand_action = [dual_gripper_action_array[0]]
                            # No pressure sensors for gripper
                            left_hand_pressures = []
                            right_hand_pressures = []
                            # Add velocities and torques for gripper
                            left_hand_vel = []
                            right_hand_vel = []
                            left_hand_torque = []
                            right_hand_torque = []
                    elif args.hand == "inspire1":
                        with dual_hand_data_lock:
                            left_hand_state = dual_hand_state_array[:6]
                            right_hand_state = dual_hand_state_array[-6:]
                            left_hand_action = dual_hand_action_array[:6]
                            right_hand_action = dual_hand_action_array[-6:]
                            # No pressure sensors for inspire hand
                            left_hand_pressures = []
                            right_hand_pressures = []
                            # Add velocities and torques for inspire hand
                            left_hand_vel = []
                            right_hand_vel = []
                            left_hand_torque = []
                            right_hand_torque = []
                    else:
                        print("No dexterous hand set.")
                        pass
                    # head image
                    current_tv_image = tv_img_array.copy()
                    # wrist image
                    if WRIST:
                        current_wrist_image = wrist_img_array.copy()
                    # arm state and action
                    left_arm_state  = current_lr_arm_q[:7]
                    right_arm_state = current_lr_arm_q[-7:]
                    left_arm_action = sol_q[:7]
                    right_arm_action = sol_q[-7:]
                    # Get velocities and torques for arms
                    left_arm_vel  = current_lr_arm_dq[:7] if (args.force and len(current_lr_arm_dq) >= 14) else []
                    right_arm_vel = current_lr_arm_dq[-7:] if (args.force and len(current_lr_arm_dq) >= 14) else []
                    # Try to get torques if available
                    try:
                        current_lr_arm_torque = arm_ctrl.get_current_dual_arm_torque()
                        left_arm_torque = current_lr_arm_torque[:7] if current_lr_arm_torque is not None else []
                        right_arm_torque = current_lr_arm_torque[-7:] if current_lr_arm_torque is not None else []
                    except Exception:
                        left_arm_torque = []
                        right_arm_torque = []
                    
                    # Note: Hand torque and velocity values are already collected above in the hand-specific sections

                    if recording:
                        colors = {}
                        depths = {}
                        if BINOCULAR:
                            colors[f"color_{0}"] = current_tv_image[:, :tv_img_shape[1]//2]
                            colors[f"color_{1}"] = current_tv_image[:, tv_img_shape[1]//2:]
                            if WRIST:
                                colors[f"color_{2}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                                colors[f"color_{3}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                        else:
                            colors[f"color_{0}"] = current_tv_image
                            if WRIST:
                                colors[f"color_{1}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                                colors[f"color_{2}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                        states = {
                            "left_arm": {                                                                    
                                "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                                "qvel":   left_arm_vel if isinstance(left_arm_vel, list) else left_arm_vel.tolist(),
                                "torque": left_arm_torque if isinstance(left_arm_torque, list) else left_arm_torque.tolist(),                        
                            }, 
                            "right_arm": {                                                                    
                                "qpos":   right_arm_state.tolist(),       
                                "qvel":   right_arm_vel if isinstance(right_arm_vel, list) else right_arm_vel.tolist(),
                                "torque": right_arm_torque if isinstance(right_arm_torque, list) else right_arm_torque.tolist(),                         
                            },                        
                            "left_hand": {                                                                    
                                "qpos":   left_hand_state,           
                                "qvel":   left_hand_vel if isinstance(left_hand_vel, list) else left_hand_vel,                          
                                "torque": left_hand_torque if isinstance(left_hand_torque, list) else left_hand_torque,                          
                                "pressures": left_hand_pressures,     # Add pressure data
                            }, 
                            "right_hand": {                                                                    
                                "qpos":   right_hand_state,       
                                "qvel":   right_hand_vel if isinstance(right_hand_vel, list) else right_hand_vel,                          
                                "torque": right_hand_torque if isinstance(right_hand_torque, list) else right_hand_torque, 
                                "pressures": right_hand_pressures,   # Add pressure data
                            }, 
                            "body": None, # TODO Hier könnte man um den Körper Erweitern
                        }
                        
                        # Add camera servo states if active camera is enabled
                        if camera_controller and camera_controller.connected:
                            try:
                                camera_pitch, camera_yaw = camera_controller.get_positions()
                                states["camera"] = {
                                    "qpos": [camera_pitch, camera_yaw],  # [pitch, yaw] in radians
                                    "qvel": [],  # Velocity not available
                                    "torque": []  # Torque not available
                                }
                            except Exception as e:
                                logger.warning(f"Failed to record camera servo positions: {e}")
                                states["camera"] = {
                                    "qpos": [0.0, 0.0],
                                    "qvel": [],
                                    "torque": []
                                }
                        
                        actions = { # Todo torque und pressure auch als action? Laut Paper bi-ACT schon.
                            "left_arm": {                                   
                                "qpos":   left_arm_action.tolist(),       
                                "qvel":   [],       
                                "torque": [],      
                            }, 
                            "right_arm": {                                   
                                "qpos":   right_arm_action.tolist(),       
                                "qvel":   [],       
                                "torque": [],       
                            },                         
                            "left_hand": {                                   
                                "qpos":   left_hand_action,       
                                "qvel":   [],       
                                "torque": [],       
                            }, 
                            "right_hand": {                                   
                                "qpos":   right_hand_action,       
                                "qvel":   [],       
                                "torque": [], 
                            }, 
                            "body": None, 
                        }
                        
                        # Add camera servo actions if active camera is enabled
                        if camera_controller and camera_controller.connected:
                            try:
                                # Use the same target positions as the current positions since we don't track actions separately
                                camera_pitch, camera_yaw = camera_controller.get_positions()
                                actions["camera"] = {
                                    "qpos": [camera_pitch, camera_yaw],  # [pitch, yaw] in radians
                                    "qvel": [],
                                    "torque": []
                                }
                            except Exception as e:
                                logger.warning(f"Failed to record camera servo actions: {e}")
                                actions["camera"] = {
                                    "qpos": [0.0, 0.0],
                                    "qvel": [],
                                    "torque": []
                                }
                        
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / float(args.frequency)) - time_elapsed)
                time.sleep(sleep_time)

                # Only log performance details occasionally to avoid flooding the log
                if frame_counter % 300 == 0:  # Every ~10 seconds at 30fps
                    actual_frequency = 1.0 / (time_elapsed + sleep_time) if (time_elapsed + sleep_time) > 0 else args.frequency
                    logger.info(f"Performance: frame_time={time_elapsed*1000:.1f}ms, sleep={sleep_time*1000:.1f}ms, actual_freq={actual_frequency:.1f}Hz")

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt, exiting program...")
    except Exception as e:
        logger.exception(f"Error in main loop: {e}")
    finally:
        # Clean up
        if pose_logger:
            pose_logger.stop()
            logger.info("Stopped pose logger and saved data")
            
        # Cleanup camera controller
        if camera_controller:
            camera_controller.disconnect()
            logger.info("Active camera controller disconnected")
            
        arm_ctrl.ctrl_dual_arm_go_home()
        logger.info("Arms returned to home position")
        
        tv_img_shm.unlink()
        tv_img_shm.close()
        if WRIST:
            wrist_img_shm.unlink()
            wrist_img_shm.close()
        if args.record:
            recorder.close()
        logger.info("Resources cleaned up, exiting program")
        exit(0)