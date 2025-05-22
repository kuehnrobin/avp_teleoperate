import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Array, Lock
import threading
import logging
from datetime import datetime
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.open_television.tv_wrapper import TeleVisionWrapper
from teleop.robot_control.robot_arm import (
    G1_29_ArmController,
    G1_23_ArmController,
    H1_2_ArmController,
    H1_ArmController,
)
from teleop.robot_control.robot_arm_ik import (
    G1_29_ArmIK,
    G1_23_ArmIK,
    H1_2_ArmIK,
    H1_ArmIK,
)
from teleop.robot_control.robot_hand_unitree import (
    Dex3_1_Controller,
    Gripper_Controller,
)
from teleop.robot_control.robot_hand_inspire import Inspire_Controller
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter


# Set up logging to file and console
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(current_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pose_analysis_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),  # Also output to console
        ],
    )
    logging.info(f"Logging to {log_file}")
    return log_file


# Helper function to analyze pose and return a string representation
def analyze_pose(name, pose_matrix):
    """Analyze pose matrix and return string representation"""
    result = []
    result.append(f"{name} Analysis:")

    # Extract position
    position = pose_matrix[:3, 3]
    result.append(
        f"  Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]"
    )

    # Extract rotation as matrix
    rot_mat = pose_matrix[:3, :3]
    result.append(f"  Rotation Matrix:\n{rot_mat}")

    # For wrist poses, highlight height
    if "wrist" in name.lower():
        result.append(f"  HEIGHT CHECK: {position[2]:.3f}")
        if position[2] > 0.2:  # Example threshold
            result.append(f"  WARNING: {name} is above threshold!")

    return "\n".join(result)


# Function to log full matrices
def log_matrices(
    head_rmat, left_wrist, right_wrist, left_hand, right_hand, frame_num=None
):
    prefix = f"FRAME {frame_num}: " if frame_num else ""
    logging.info(f"{prefix}Head Rotation Matrix:\n{head_rmat}")
    logging.info(f"{prefix}Left Wrist Matrix:\n{left_wrist}")
    logging.info(f"{prefix}Right Wrist Matrix:\n{right_wrist}")
    logging.info(f"{prefix}Left Hand Matrix Shape: {left_hand.shape}")
    logging.info(f"{prefix}Right Hand Matrix Shape: {right_hand.shape}")

    # Log more detailed analysis for wrists
    logging.info(analyze_pose("Left Wrist", left_wrist))
    logging.info(analyze_pose("Right Wrist", right_wrist))


if __name__ == "__main__":
    # Setup logging first thing
    log_file = setup_logging()
    logging.info("Starting teleop_hand_and_arm.py in DEBUG mode")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_dir", type=str, default="./utils/data", help="path to save data"
    )
    parser.add_argument(
        "--frequency", type=int, default=30.0, help="save data's frequency"
    )

    parser.add_argument("--record", action="store_true", help="Save data or not")
    parser.add_argument(
        "--no-record", dest="record", action="store_false", help="Do not save data"
    )
    parser.set_defaults(record=False)

    parser.add_argument(
        "--arm",
        type=str,
        choices=["G1_29", "G1_23", "H1_2", "H1"],
        default="G1_29",
        help="Select arm controller",
    )
    parser.add_argument(
        "--hand",
        type=str,
        choices=["dex3", "gripper", "inspire1"],
        help="Select hand controller",
    )

    parser.add_argument(
        "--cyclonedds_uri",
        type=str,
        default="enxa0cec8616f27",
        help="Network interface for CycloneDDS (default: enxa0cec8616f27)",
    )
    # Speed Limit
    parser.add_argument(
        "--arm-speed",
        type=float,
        default=None,
        help="Set the arm velocity limit (default is controller-specific)",
    )
    parser.add_argument(
        "--no-gradual-speed", action="store_true", help="Disable gradual speed increase"
    )
    # Logging frequency
    parser.add_argument(
        "--log-freq",
        type=int,
        default=30,
        help="How often to log pose data (in frames)",
    )

    args = parser.parse_args()
    logging.info(f"Command line args: {args}")

    # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    img_config = {
        "fps": 30,
        "head_camera_type": "opencv",
        "head_camera_image_shape": [
            480,
            1280,
        ],  # [1080, 3840], #[480, 1280],  # Head camera resolution
        "head_camera_id_numbers": [6],
        "wrist_camera_type": "opencv",
        "wrist_camera_image_shape": [480, 640],  # Wrist camera resolution
        "wrist_camera_id_numbers": [8, 10],
    }
    logging.info(f"Image configuration: {img_config}")

    ASPECT_RATIO_THRESHOLD = (
        2.0  # If the aspect ratio exceeds this value, it is considered binocular
    )
    if len(img_config["head_camera_id_numbers"]) > 1 or (
        img_config["head_camera_image_shape"][1]
        / img_config["head_camera_image_shape"][0]
        > ASPECT_RATIO_THRESHOLD
    ):
        BINOCULAR = True
    else:
        BINOCULAR = False
    if "wrist_camera_type" in img_config:
        WRIST = True
    else:
        WRIST = False

    if BINOCULAR and not (
        img_config["head_camera_image_shape"][1]
        / img_config["head_camera_image_shape"][0]
        > ASPECT_RATIO_THRESHOLD
    ):
        tv_img_shape = (
            img_config["head_camera_image_shape"][0],
            img_config["head_camera_image_shape"][1] * 2,
            3,
        )
    else:
        tv_img_shape = (
            img_config["head_camera_image_shape"][0],
            img_config["head_camera_image_shape"][1],
            3,
        )

    tv_img_shm = shared_memory.SharedMemory(
        create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize
    )
    tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)

    if WRIST:
        wrist_img_shape = (
            img_config["wrist_camera_image_shape"][0],
            img_config["wrist_camera_image_shape"][1] * 2,
            3,
        )
        wrist_img_shm = shared_memory.SharedMemory(
            create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize
        )
        wrist_img_array = np.ndarray(
            wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf
        )
        img_client = ImageClient(
            tv_img_shape=tv_img_shape,
            tv_img_shm_name=tv_img_shm.name,
            wrist_img_shape=wrist_img_shape,
            wrist_img_shm_name=wrist_img_shm.name,
        )
    else:
        img_client = ImageClient(
            tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name
        )

    image_receive_thread = threading.Thread(
        target=img_client.receive_process, daemon=True
    )
    image_receive_thread.daemon = True
    image_receive_thread.start()
    logging.info("Image client thread started")

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    tv_wrapper = TeleVisionWrapper(
        BINOCULAR, tv_img_shape, tv_img_shm.name, ngrok=True
    )  # True for quest3
    logging.info("TeleVisionWrapper initialized")

    # COMMENTED OUT: arm controller initialization
    """
    # arm
    if args.arm == 'G1_29':
        arm_ctrl = G1_29_ArmController(networkInterface=args.cyclonedds_uri)
        arm_ik = G1_29_ArmIK()
        if args.arm_speed is not None:
            arm_ctrl.arm_velocity_limit = args.arm_speed
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
    """
    logging.info("ARM CONTROLLERS DISABLED - Running in debug/analysis mode")

    # COMMENTED OUT: hand controller initialization (but keep data structures for debugging)
    if args.hand == "dex3":
        left_hand_array = Array("d", 75, lock=True)  # [input]
        right_hand_array = Array("d", 75, lock=True)  # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array(
            "d", 14, lock=False
        )  # [output] current left, right hand state(14) data.
        dual_hand_action_array = Array(
            "d", 14, lock=False
        )  # [output] current left, right hand action(14) data.
        # hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, networkInterface=args.cyclonedds_uri)
    elif args.hand == "gripper":
        left_hand_array = Array("d", 75, lock=True)
        right_hand_array = Array("d", 75, lock=True)
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array(
            "d", 2, lock=False
        )  # current left, right gripper state(2) data.
        dual_gripper_action_array = Array(
            "d", 2, lock=False
        )  # current left, right gripper action(2) data.
        # gripper_ctrl = Gripper_Controller(left_hand_array, right_hand_array, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array, networkInterface=args.cyclonedds_uri)
    elif args.hand == "inspire1":
        left_hand_array = Array("d", 75, lock=True)  # [input]
        right_hand_array = Array("d", 75, lock=True)  # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array(
            "d", 12, lock=False
        )  # [output] current left, right hand state(12) data.
        dual_hand_action_array = Array(
            "d", 12, lock=False
        )  # [output] current left, right hand action(12) data.
        # hand_ctrl = Inspire_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
    else:
        logging.info("No hand controller specified")

    if args.record:
        recorder = EpisodeWriter(
            task_dir=args.task_dir, frequency=args.frequency, rerun_log=True
        )
        recording = False

    try:
        logging.info("Press 'r' to start pose analysis mode, 'q' to quit")
        user_input = input(
            "Please enter the start signal (enter 'r' to start the subsequent program):\n"
        )
        if user_input.lower() == "r":
            if not args.no_gradual_speed:
                # arm_ctrl.speed_gradual_max()
                pass
            running = True

            # Initialize frame counter for logging
            frame_counter = 0

            logging.info("Starting main loop for pose analysis")
            while running:
                start_time = time.time()
                head_rmat, left_wrist, right_wrist, left_hand, right_hand = (
                    tv_wrapper.get_data()
                )

                # Update frame counter
                frame_counter += 1

                # Log data at specified frequency
                if frame_counter % args.log_freq == 0:
                    logging.info(f"=" * 50)
                    logging.info(f"FRAME {frame_counter} POSE ANALYSIS:")
                    logging.info(f"-" * 50)
                    log_matrices(
                        head_rmat,
                        left_wrist,
                        right_wrist,
                        left_hand,
                        right_hand,
                        frame_counter,
                    )
                    logging.info("=" * 50)

                # send hand skeleton data to hand_ctrl.control_process
                if args.hand:
                    left_hand_array[:] = left_hand.flatten()
                    right_hand_array[:] = right_hand.flatten()

                # COMMENTED OUT: arm control
                """
                # get current state data.
                current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
                current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

                # solve ik using motor data and wrist pose, then use ik results to control arms.
                time_ik_start = time.time()
                sol_q, sol_tauff  = arm_ik.solve_ik(left_wrist, right_wrist, current_lr_arm_q, current_lr_arm_dq)
                time_ik_end = time.time()
                # print(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
                """

                tv_resized_image = cv2.resize(
                    tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2)
                )
                cv2.imshow("record image", tv_resized_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    running = False
                    logging.info("User pressed 'q', exiting loop")
                elif key == ord("s") and args.record:
                    recording = not recording  # state flipping
                    if recording:
                        if not recorder.create_episode():
                            recording = False
                            logging.info("Failed to create recording episode")
                        else:
                            logging.info("Started recording episode")
                    else:
                        recorder.save_episode()

                # record data
                if args.record:
                    # dex hand or gripper
                    if args.hand == "dex3":
                        with dual_hand_data_lock:
                            left_hand_state = dual_hand_state_array[:7]
                            right_hand_state = dual_hand_state_array[-7:]
                            left_hand_action = dual_hand_action_array[:7]
                            right_hand_action = dual_hand_action_array[-7:]
                    elif args.hand == "gripper":
                        with dual_gripper_data_lock:
                            left_hand_state = [dual_gripper_state_array[1]]
                            right_hand_state = [dual_gripper_state_array[0]]
                            left_hand_action = [dual_gripper_action_array[1]]
                            right_hand_action = [dual_gripper_action_array[0]]
                    elif args.hand == "inspire1":
                        with dual_hand_data_lock:
                            left_hand_state = dual_hand_state_array[:6]
                            right_hand_state = dual_hand_state_array[-6:]
                            left_hand_action = dual_hand_action_array[:6]
                            right_hand_action = dual_hand_action_array[-6:]
                    else:
                        print("No dexterous hand set.")
                        pass
                    # head image
                    current_tv_image = tv_img_array.copy()
                    # wrist image
                    if WRIST:
                        current_wrist_image = wrist_img_array.copy()
                    # arm state and action
                    # left_arm_state  = current_lr_arm_q[:7]
                    # right_arm_state = current_lr_arm_q[-7:]
                    # left_arm_action = sol_q[:7]
                    # right_arm_action = sol_q[-7:]
                    left_arm_state = np.zeros(7)
                    right_arm_state = np.zeros(7)
                    left_arm_action = np.zeros(7)
                    right_arm_action = np.zeros(7)

                    if recording:
                        colors = {}
                        depths = {}
                        if BINOCULAR:
                            colors[f"color_{0}"] = current_tv_image[
                                :, : tv_img_shape[1] // 2
                            ]
                            colors[f"color_{1}"] = current_tv_image[
                                :, tv_img_shape[1] // 2 :
                            ]
                            if WRIST:
                                colors[f"color_{2}"] = current_wrist_image[
                                    :, : wrist_img_shape[1] // 2
                                ]
                                colors[f"color_{3}"] = current_wrist_image[
                                    :, wrist_img_shape[1] // 2 :
                                ]
                        else:
                            colors[f"color_{0}"] = current_tv_image
                            if WRIST:
                                colors[f"color_{1}"] = current_wrist_image[
                                    :, : wrist_img_shape[1] // 2
                                ]
                                colors[f"color_{2}"] = current_wrist_image[
                                    :, wrist_img_shape[1] // 2 :
                                ]
                        states = {
                            "left_arm": {
                                "qpos": left_arm_state.tolist(),  # numpy.array -> list
                                "qvel": [],
                                "torque": [],
                            },
                            "right_arm": {
                                "qpos": right_arm_state.tolist(),
                                "qvel": [],
                                "torque": [],
                            },
                            "left_hand": {
                                "qpos": left_hand_state,
                                "qvel": [],
                                "torque": [],
                            },
                            "right_hand": {
                                "qpos": right_hand_state,
                                "qvel": [],
                                "torque": [],
                            },
                            "body": None,
                        }
                        actions = {
                            "left_arm": {
                                "qpos": left_arm_action.tolist(),
                                "qvel": [],
                                "torque": [],
                            },
                            "right_arm": {
                                "qpos": right_arm_action.tolist(),
                                "qvel": [],
                                "torque": [],
                            },
                            "left_hand": {
                                "qpos": left_hand_action,
                                "qvel": [],
                                "torque": [],
                            },
                            "right_hand": {
                                "qpos": right_hand_action,
                                "qvel": [],
                                "torque": [],
                            },
                            "body": None,
                        }
                        recorder.add_item(
                            colors=colors, depths=depths, states=states, actions=actions
                        )

                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / float(args.frequency)) - time_elapsed)
                time.sleep(sleep_time)
                # print(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt, exiting program...")
    except Exception as e:
        logging.exception(f"Error in main loop: {e}")
    finally:
        # arm_ctrl.ctrl_dual_arm_go_home()
        logging.info("Cleaning up resources...")
        tv_img_shm.unlink()
        tv_img_shm.close()
        if WRIST:
            wrist_img_shm.unlink()
            wrist_img_shm.close()
        if args.record:
            recorder.close()
        logging.info(f"Analysis complete. Log file: {log_file}")
        logging.info("Exiting program.")
        exit(0)
