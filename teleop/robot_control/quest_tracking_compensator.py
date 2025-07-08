#!/usr/bin/env python3
"""
Quest Tracking Compensator

This module compensates for Quest 3 tracking errors that occur when the operator moves their head.
The tracking errors manifest as unwanted arm movements correlated with head orientation changes.

Author: Robin
Date: 2025
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import logging


class QuestTrackingCompensator:
    """
    Compensates for Quest 3 VR headset tracking errors by applying linear corrections
    to hand/wrist positions based on head orientation changes.
    
    The Quest 3 introduces tracking errors when the operator moves their head:
    - Looking down (pitch increase) -> robot arms move up
    - Looking up (pitch decrease) -> robot arms move down  
    - Looking left (yaw increase) -> robot arms move right
    - Looking right (yaw decrease) -> robot arms move left
    
    This compensator applies linear corrections to counteract these effects.
    """
    
    def __init__(self, 
                 pitch_compensation_factor=0.1,
                 yaw_compensation_factor=0.1,
                 enable_pitch_compensation=True,
                 enable_yaw_compensation=True,
                 enable_roll_compensation=False,
                 max_pitch_correction=0.05,
                 max_yaw_correction=0.05,
                 logger=None):
        """
        Initialize the Quest tracking compensator.
        
        Args:
            pitch_compensation_factor (float): Linear factor for pitch-based position correction
            yaw_compensation_factor (float): Linear factor for yaw-based position correction
            enable_pitch_compensation (bool): Enable pitch compensation
            enable_yaw_compensation (bool): Enable yaw compensation
            enable_roll_compensation (bool): Enable roll compensation (disabled by default)
            max_pitch_correction (float): Maximum correction distance in meters for pitch
            max_yaw_correction (float): Maximum correction distance in meters for yaw
            logger: Logger instance for debugging
        """
        self.pitch_compensation_factor = pitch_compensation_factor
        self.yaw_compensation_factor = yaw_compensation_factor
        self.enable_pitch_compensation = enable_pitch_compensation
        self.enable_yaw_compensation = enable_yaw_compensation
        self.enable_roll_compensation = enable_roll_compensation
        self.max_pitch_correction = max_pitch_correction
        self.max_yaw_correction = max_yaw_correction
        
        self.logger = logger or logging.getLogger('quest_tracking_compensator')
        
        # Store reference head orientation (set during calibration or first frame)
        self.reference_head_orientation = None
        self.is_calibrated = False
        
        # Statistics for monitoring
        self.frame_count = 0
        self.total_pitch_correction = 0.0
        self.total_yaw_correction = 0.0
        
        self.logger.info(f"Quest tracking compensator initialized - "
                        f"pitch_factor: {pitch_compensation_factor}, "
                        f"yaw_factor: {yaw_compensation_factor}, "
                        f"max_corrections: pitch={max_pitch_correction}m, yaw={max_yaw_correction}m")
    
    def calibrate(self, head_rmat):
        """
        Calibrate the compensator using the current head orientation as reference.
        Call this when the operator is in a neutral position.
        
        Args:
            head_rmat (np.ndarray): Head rotation matrix (3x3)
        """
        self.reference_head_orientation = head_rmat.copy()
        self.is_calibrated = True
        
        # Extract reference angles for logging
        ref_rotation = R.from_matrix(head_rmat)
        ref_euler = ref_rotation.as_euler('xyz', degrees=True)
        
        self.logger.info(f"Quest compensator calibrated with reference head orientation: "
                        f"roll={ref_euler[0]:.1f}°, pitch={ref_euler[1]:.1f}°, yaw={ref_euler[2]:.1f}°")
    
    def extract_head_angles(self, head_rmat):
        """
        Extract pitch, yaw, roll angles from head rotation matrix.
        
        Args:
            head_rmat (np.ndarray): Head rotation matrix (3x3)
            
        Returns:
            tuple: (pitch, yaw, roll) angles in radians
        """
        rotation = R.from_matrix(head_rmat)
        # Use 'xyz' convention: first rotate about x (roll), then y (pitch), then z (yaw)
        angles = rotation.as_euler('xyz')
        roll, pitch, yaw = angles[0], angles[1], angles[2]
        return pitch, yaw, roll
    
    def compute_angle_deltas(self, head_rmat):
        """
        Compute the angular differences from the reference head orientation.
        
        Args:
            head_rmat (np.ndarray): Current head rotation matrix (3x3)
            
        Returns:
            tuple: (delta_pitch, delta_yaw, delta_roll) in radians
        """
        if not self.is_calibrated:
            self.calibrate(head_rmat)
            return 0.0, 0.0, 0.0
        
        # Get current and reference angles
        current_pitch, current_yaw, current_roll = self.extract_head_angles(head_rmat)
        ref_pitch, ref_yaw, ref_roll = self.extract_head_angles(self.reference_head_orientation)
        
        # Compute deltas (handle angle wrapping)
        delta_pitch = self._normalize_angle(current_pitch - ref_pitch)
        delta_yaw = self._normalize_angle(current_yaw - ref_yaw)
        delta_roll = self._normalize_angle(current_roll - ref_roll)
        
        return delta_pitch, delta_yaw, delta_roll
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def _clamp_correction(self, correction, max_correction):
        """Clamp correction to maximum allowed value."""
        return np.clip(correction, -max_correction, max_correction)
    
    def compensate_wrist_positions(self, head_rmat, left_wrist_mat, right_wrist_mat):
        """
        Apply tracking compensation to wrist positions based on head orientation.
        
        Args:
            head_rmat (np.ndarray): Head rotation matrix (3x3)
            left_wrist_mat (np.ndarray): Left wrist transformation matrix (4x4)
            right_wrist_mat (np.ndarray): Right wrist transformation matrix (4x4)
            
        Returns:
            tuple: (compensated_left_wrist_mat, compensated_right_wrist_mat)
        """
        self.frame_count += 1
        
        # Make copies to avoid modifying the original matrices
        compensated_left = left_wrist_mat.copy()
        compensated_right = right_wrist_mat.copy()
        
        # Compute angle deltas from reference
        delta_pitch, delta_yaw, delta_roll = self.compute_angle_deltas(head_rmat)
        
        # Initialize correction vectors
        position_correction = np.zeros(3)
        
        # Apply pitch compensation (looking up/down affects arm height)
        if self.enable_pitch_compensation:
            # Positive pitch (looking down) should move arms down to compensate for upward drift
            # Negative pitch (looking up) should move arms up to compensate for downward drift
            pitch_correction = -delta_pitch * self.pitch_compensation_factor
            pitch_correction = self._clamp_correction(pitch_correction, self.max_pitch_correction)
            position_correction[2] += pitch_correction  # Z-axis is vertical in robot convention
            self.total_pitch_correction += abs(pitch_correction)
        
        # Apply yaw compensation (looking left/right affects arm lateral position)
        if self.enable_yaw_compensation:
            # Positive yaw (looking left) should move arms left to compensate for rightward drift
            # Negative yaw (looking right) should move arms right to compensate for leftward drift
            yaw_correction = -delta_yaw * self.yaw_compensation_factor
            yaw_correction = self._clamp_correction(yaw_correction, self.max_yaw_correction)
            position_correction[1] += yaw_correction  # Y-axis is lateral in robot convention
            self.total_yaw_correction += abs(yaw_correction)
        
        # Apply roll compensation if enabled (usually not needed)
        if self.enable_roll_compensation:
            # Roll compensation could affect both X and Y depending on implementation
            pass
        
        # Apply position corrections to both wrists
        compensated_left[0:3, 3] += position_correction
        compensated_right[0:3, 3] += position_correction
        
        # Log compensation details periodically
        if self.frame_count % 300 == 0:  # Every ~10 seconds at 30fps
            avg_pitch_correction = self.total_pitch_correction / 300
            avg_yaw_correction = self.total_yaw_correction / 300
            
            self.logger.info(f"Quest compensation stats (last 300 frames): "
                           f"avg_pitch_correction={avg_pitch_correction*1000:.2f}mm, "
                           f"avg_yaw_correction={avg_yaw_correction*1000:.2f}mm, "
                           f"current_deltas: pitch={np.degrees(delta_pitch):.1f}°, "
                           f"yaw={np.degrees(delta_yaw):.1f}°")
            
            # Reset counters
            self.total_pitch_correction = 0.0
            self.total_yaw_correction = 0.0
        
        # Log significant corrections
        if abs(delta_pitch) > np.radians(5) or abs(delta_yaw) > np.radians(5):
            self.logger.debug(f"Large head movement detected - "
                            f"pitch: {np.degrees(delta_pitch):.1f}°, "
                            f"yaw: {np.degrees(delta_yaw):.1f}°, "
                            f"correction: {position_correction*1000}")
        
        return compensated_left, compensated_right
    
    def reset_calibration(self):
        """Reset the calibration reference."""
        self.reference_head_orientation = None
        self.is_calibrated = False
        self.logger.info("Quest compensator calibration reset")
    
    def set_compensation_factors(self, pitch_factor=None, yaw_factor=None):
        """
        Update compensation factors during runtime.
        
        Args:
            pitch_factor (float, optional): New pitch compensation factor
            yaw_factor (float, optional): New yaw compensation factor
        """
        if pitch_factor is not None:
            self.pitch_compensation_factor = pitch_factor
            self.logger.info(f"Updated pitch compensation factor to {pitch_factor}")
        
        if yaw_factor is not None:
            self.yaw_compensation_factor = yaw_factor
            self.logger.info(f"Updated yaw compensation factor to {yaw_factor}")
    
    def get_status(self):
        """
        Get current status of the compensator.
        
        Returns:
            dict: Status information
        """
        return {
            'is_calibrated': self.is_calibrated,
            'frame_count': self.frame_count,
            'pitch_compensation_factor': self.pitch_compensation_factor,
            'yaw_compensation_factor': self.yaw_compensation_factor,
            'enable_pitch_compensation': self.enable_pitch_compensation,
            'enable_yaw_compensation': self.enable_yaw_compensation,
            'max_pitch_correction': self.max_pitch_correction,
            'max_yaw_correction': self.max_yaw_correction
        }


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('test')
    
    # Create compensator
    compensator = QuestTrackingCompensator(
        pitch_compensation_factor=0.05,  # 5cm correction per radian
        yaw_compensation_factor=0.03,    # 3cm correction per radian
        logger=logger
    )
    
    # Simulate head orientations
    # Reference orientation (looking straight)
    head_ref = np.eye(3)
    
    # Looking down (positive pitch)
    head_down = R.from_euler('xyz', [0, np.radians(10), 0]).as_matrix()
    
    # Looking left (positive yaw)  
    head_left = R.from_euler('xyz', [0, 0, np.radians(15)]).as_matrix()
    
    # Example wrist matrices
    left_wrist = np.eye(4)
    left_wrist[0:3, 3] = [-0.3, 0.2, 0.1]  # Example position
    
    right_wrist = np.eye(4)
    right_wrist[0:3, 3] = [0.3, 0.2, 0.1]  # Example position
    
    # Test compensation
    print("Testing Quest tracking compensator...")
    
    # Calibrate with reference
    compensator.calibrate(head_ref)
    
    # Test looking down
    print("\nTesting looking down (10° pitch):")
    left_comp, right_comp = compensator.compensate_wrist_positions(head_down, left_wrist, right_wrist)
    print(f"Original left wrist position: {left_wrist[0:3, 3]}")
    print(f"Compensated left wrist position: {left_comp[0:3, 3]}")
    print(f"Position change: {left_comp[0:3, 3] - left_wrist[0:3, 3]}")
    
    # Test looking left
    print("\nTesting looking left (15° yaw):")
    left_comp, right_comp = compensator.compensate_wrist_positions(head_left, left_wrist, right_wrist)
    print(f"Original left wrist position: {left_wrist[0:3, 3]}")
    print(f"Compensated left wrist position: {left_comp[0:3, 3]}")
    print(f"Position change: {left_comp[0:3, 3] - left_wrist[0:3, 3]}")
    
    print(f"\nCompensator status: {compensator.get_status()}")
