#!/usr/bin/env python3
"""
Thumb Pinch Corrector for Unitree Dex3 Hand

This module provides a real-time correction system for thumb positioning during pinching gestures.
Based on the analysis that shows thumb consistently dropping 10-17 degrees during pinching,
this corrector detects pinching conditions and applies targeted corrections.
"""

import numpy as np
import math

class ThumbPinchCorrector:
    """
    Real-time thumb position corrector for pinching gestures.
    
    The core issue identified:
    - Thumb drops from 70° to 52.6° during pinch (-17.4° change)
    - Need +10 to +17 degrees correction during pinching
    - Correction should be smooth and context-aware
    """
    
    def __init__(self):
        # Pinching detection parameters
        self.pinch_distance_threshold = 0.06  # 6cm - below this is considered pinching
        self.pinch_approach_threshold = 0.08  # 8cm - start applying correction gradually
        
        # Correction parameters
        self.max_thumb_correction = math.radians(15)  # 15 degrees max correction
        self.min_thumb_correction = math.radians(5)   # 5 degrees min correction
        
        # Joint indices for DexPilot joint ordering
        # Based on target_joint_names in unitree_dex3_left_dexpilot.yml:
        # [0] left_hand_thumb_0_joint, [1] left_hand_thumb_1_joint, [2] left_hand_thumb_2_joint,
        # [3] left_hand_middle_0_joint, [4] left_hand_middle_1_joint,
        # [5] left_hand_index_0_joint, [6] left_hand_index_1_joint
        self.thumb_joints = {
            'thumb_0': 0,  # thumb base rotation
            'thumb_1': 1,  # thumb bend (main correction target)
            'thumb_2': 2   # thumb tip
        }
        
        # Smoothing parameters
        self.correction_history = []
        self.history_length = 5
        
    def detect_pinching(self, thumb_tip_pos, index_tip_pos, middle_tip_pos):
        """
        Detect if the hand is in a pinching gesture.
        
        Args:
            thumb_tip_pos: 3D position of thumb tip
            index_tip_pos: 3D position of index tip  
            middle_tip_pos: 3D position of middle tip
            
        Returns:
            tuple: (is_pinching, pinch_strength, closest_finger_distance)
        """
        # Calculate distances from thumb to other fingertips
        thumb_index_dist = np.linalg.norm(thumb_tip_pos - index_tip_pos)
        thumb_middle_dist = np.linalg.norm(thumb_tip_pos - middle_tip_pos)
        
        # Use the closest finger for pinch detection
        closest_dist = min(thumb_index_dist, thumb_middle_dist)
        
        # Determine pinching state
        is_pinching = closest_dist < self.pinch_distance_threshold
        is_approaching = closest_dist < self.pinch_approach_threshold
        
        # Calculate pinch strength (0.0 = far apart, 1.0 = fully pinched)
        if closest_dist >= self.pinch_approach_threshold:
            pinch_strength = 0.0
        else:
            # Linear interpolation from approach threshold to pinch threshold
            pinch_strength = 1.0 - (closest_dist - self.pinch_distance_threshold) / (
                self.pinch_approach_threshold - self.pinch_distance_threshold
            )
            pinch_strength = max(0.0, min(1.0, pinch_strength))  # Clamp to [0,1]
        
        return is_pinching, pinch_strength, closest_dist
    
    def calculate_thumb_correction(self, pinch_strength, thumb_joint_angles):
        """
        Calculate the correction to apply to thumb joints based on pinch strength.
        
        Args:
            pinch_strength: Float 0.0-1.0 indicating how strong the pinch is
            thumb_joint_angles: Current thumb joint angles [thumb_0, thumb_1, thumb_2]
            
        Returns:
            numpy.array: Correction values to add to thumb joint angles
        """
        if pinch_strength <= 0.0:
            return np.zeros(3)
        
        # Primary correction targets thumb_1 joint (the main bending joint)
        thumb_1_correction = self.min_thumb_correction + (
            self.max_thumb_correction - self.min_thumb_correction
        ) * pinch_strength
        
        # Secondary corrections for other thumb joints
        # thumb_0: slight outward rotation to improve opposition
        thumb_0_correction = math.radians(3) * pinch_strength
        
        # thumb_2: minimal correction as it's often at joint limits
        thumb_2_correction = math.radians(2) * pinch_strength
        
        corrections = np.array([thumb_0_correction, thumb_1_correction, thumb_2_correction])
        
        # Apply smoothing
        self.correction_history.append(corrections)
        if len(self.correction_history) > self.history_length:
            self.correction_history.pop(0)
        
        # Return smoothed correction
        smoothed_correction = np.mean(self.correction_history, axis=0)
        return smoothed_correction
    
    def apply_correction(self, hand_joint_angles, thumb_tip_pos, index_tip_pos, middle_tip_pos):
        """
        Apply thumb correction to the full hand joint angle array.
        
        Args:
            hand_joint_angles: Full 7-element joint angle array
            thumb_tip_pos: 3D position of thumb tip
            index_tip_pos: 3D position of index tip
            middle_tip_pos: 3D position of middle tip
            
        Returns:
            numpy.array: Corrected joint angles
        """
        # Detect pinching
        is_pinching, pinch_strength, closest_dist = self.detect_pinching(
            thumb_tip_pos, index_tip_pos, middle_tip_pos
        )
        
        # Extract current thumb joint angles (DexPilot order: thumb joints are indices 0, 1, 2)
        thumb_angles = hand_joint_angles[0:3]  # First 3 joints are thumb joints
        
        # Calculate corrections
        corrections = self.calculate_thumb_correction(pinch_strength, thumb_angles)
        
        # Apply corrections to the joint angles (DexPilot order)
        corrected_angles = hand_joint_angles.copy()
        corrected_angles[0:3] += corrections  # Apply to thumb joints (indices 0, 1, 2)
        
        # Debug info (can be removed for production)
        if pinch_strength > 0.1:  # Only log when there's significant pinching
            print(f"Thumb Correction: pinch_strength={pinch_strength:.2f}, "
                  f"closest_dist={closest_dist:.3f}m, "
                  f"corrections=[{corrections[0]*180/math.pi:.1f}°, "
                  f"{corrections[1]*180/math.pi:.1f}°, "
                  f"{corrections[2]*180/math.pi:.1f}°]")
        
        return corrected_angles
    
    def reset_history(self):
        """Reset the correction history for clean startup."""
        self.correction_history = []


# Test functions for validation
def test_thumb_corrector():
    """Test the thumb corrector with various pinching scenarios."""
    corrector = ThumbPinchCorrector()
    
    print("=== Thumb Pinch Corrector Test ===\n")
    
    # Test case 1: Far apart (no correction)
    thumb_pos = np.array([0.08, 0.02, 0])
    index_pos = np.array([0.05, 0.08, 0])
    middle_pos = np.array([0, 0.09, 0])
    
    joint_angles = np.array([0.1, 0.5, 0.3, -0.5, -0.3, -0.7, -0.2])
    corrected = corrector.apply_correction(joint_angles, thumb_pos, index_pos, middle_pos)
    
    print("Test 1 - Far Apart:")
    print(f"  Original thumb angles: [{joint_angles[0]:.3f}, {joint_angles[1]:.3f}, {joint_angles[2]:.3f}]")
    print(f"  Corrected thumb angles: [{corrected[0]:.3f}, {corrected[1]:.3f}, {corrected[2]:.3f}]")
    print(f"  Correction applied: [{(corrected[0]-joint_angles[0])*180/math.pi:.1f}°, "
          f"{(corrected[1]-joint_angles[1])*180/math.pi:.1f}°, "
          f"{(corrected[2]-joint_angles[2])*180/math.pi:.1f}°]\n")
    
    # Test case 2: Close pinch (full correction)
    thumb_pos = np.array([0.03, 0.04, 0])
    index_pos = np.array([0.03, 0.05, 0])
    
    corrected = corrector.apply_correction(joint_angles, thumb_pos, index_pos, middle_pos)
    
    print("Test 2 - Close Pinch:")
    print(f"  Original thumb angles: [{joint_angles[0]:.3f}, {joint_angles[1]:.3f}, {joint_angles[2]:.3f}]")
    print(f"  Corrected thumb angles: [{corrected[0]:.3f}, {corrected[1]:.3f}, {corrected[2]:.3f}]")
    print(f"  Correction applied: [{(corrected[0]-joint_angles[0])*180/math.pi:.1f}°, "
          f"{(corrected[1]-joint_angles[1])*180/math.pi:.1f}°, "
          f"{(corrected[2]-joint_angles[2])*180/math.pi:.1f}°]\n")
    
    # Test case 3: Gradual approach
    print("Test 3 - Gradual Approach:")
    distances = [0.10, 0.08, 0.06, 0.05, 0.04]
    for dist in distances:
        thumb_pos = np.array([0.03, 0.04, 0])
        index_pos = np.array([0.03 + dist, 0.04, 0])
        
        corrected = corrector.apply_correction(joint_angles, thumb_pos, index_pos, middle_pos)
        correction = (corrected[1] - joint_angles[1]) * 180 / math.pi
        
        print(f"  Distance: {dist:.2f}m -> Thumb_1 correction: {correction:.1f}°")


if __name__ == "__main__":
    test_thumb_corrector()
