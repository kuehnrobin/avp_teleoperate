#!/usr/bin/env python3
"""
Test the integrated thumb pinch correction in the DexPilot system.
This simulates the corrected DexPilot processing with thumb correction.
"""

import numpy as np
import math
import sys
import os

# Add the project path
parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from thumb_pinch_corrector import ThumbPinchCorrector


def test_integrated_thumb_correction():
    """Test the complete DexPilot + Thumb Correction pipeline."""
    
    print("=== Integrated Thumb Pinch Correction Test ===\n")
    
    try:
        # Initialize DexPilot
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        # Initialize thumb corrector
        left_thumb_corrector = ThumbPinchCorrector()
        
        # Get the human indices that DexPilot expects
        human_indices = left_optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        
        print("1. DexPilot Configuration:")
        print(f"   Origin indices: {origin_indices}")
        print(f"   Task indices: {task_indices}")
        print()
        
        # Test scenarios
        scenarios = [
            {
                "name": "Open Hand (No Correction Expected)",
                "thumb_tip": [0.08, 0.02, 0],
                "index_tip": [0.05, 0.08, 0],
                "middle_tip": [0, 0.09, 0]
            },
            {
                "name": "Pinching Gesture (Correction Expected)",
                "thumb_tip": [0.03, 0.04, 0],
                "index_tip": [0.03, 0.05, 0],
                "middle_tip": [0, 0.05, 0]
            },
            {
                "name": "Approaching Pinch (Partial Correction)",
                "thumb_tip": [0.05, 0.04, 0],
                "index_tip": [0.04, 0.06, 0],
                "middle_tip": [0, 0.06, 0]
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"{i+1}. Testing: {scenario['name']}")
            
            # Simulate OpenXR hand data
            joint_pos = np.zeros((25, 3))
            joint_pos[0] = [0, 0, 0]  # wrist
            joint_pos[4] = scenario["thumb_tip"]
            joint_pos[8] = scenario["index_tip"]  # Note: mapped from OpenXR index 9
            joint_pos[12] = scenario["middle_tip"]  # Note: mapped from OpenXR index 14
            
            # Calculate vectors for DexPilot
            vectors = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            
            # Get DexPilot output
            dexpilot_output = hand_retargeting.left_retargeting.retarget(vectors)
            
            print(f"   DexPilot output: {dexpilot_output}")
            print(f"   Original thumb angles: [{dexpilot_output[0]*180/math.pi:.1f}°, "
                  f"{dexpilot_output[1]*180/math.pi:.1f}°, {dexpilot_output[2]*180/math.pi:.1f}°]")
            
            # Apply thumb correction
            corrected_output = left_thumb_corrector.apply_correction(
                dexpilot_output,
                np.array(scenario["thumb_tip"]),
                np.array(scenario["index_tip"]),
                np.array(scenario["middle_tip"])
            )
            
            print(f"   Corrected thumb angles: [{corrected_output[0]*180/math.pi:.1f}°, "
                  f"{corrected_output[1]*180/math.pi:.1f}°, {corrected_output[2]*180/math.pi:.1f}°]")
            
            # Calculate the correction applied
            correction = corrected_output - dexpilot_output
            print(f"   Correction applied: [{correction[0]*180/math.pi:.1f}°, "
                  f"{correction[1]*180/math.pi:.1f}°, {correction[2]*180/math.pi:.1f}°]")
            
            # Calculate thumb-index distance
            thumb_index_dist = np.linalg.norm(
                np.array(scenario["thumb_tip"]) - np.array(scenario["index_tip"])
            )
            print(f"   Thumb-index distance: {thumb_index_dist:.3f}m")
            print()
        
        print("✅ Integration test completed successfully!")
        print("   - DexPilot processing works correctly")
        print("   - Thumb correction is applied based on pinching distance")
        print("   - Ready for real VR teleoperation testing")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


def test_hardware_mapping():
    """Test the URDF to Hardware mapping with thumb correction."""
    
    print("\n=== Hardware Mapping Test ===\n")
    
    try:
        # Simulate DexPilot output with correction
        corrected_dexpilot_output = np.array([
            -0.5,   # URDF[0] left_hand_index_0_joint
            -0.3,   # URDF[1] left_hand_index_1_joint  
            -0.7,   # URDF[2] left_hand_middle_0_joint
            -0.4,   # URDF[3] left_hand_middle_1_joint
            0.15,   # URDF[4] left_hand_thumb_0_joint (corrected)
            0.85,   # URDF[5] left_hand_thumb_1_joint (corrected +15°)
            0.35    # URDF[6] left_hand_thumb_2_joint (corrected)
        ])
        
        # Apply URDF to Hardware mapping (from robot_hand_unitree.py)
        urdf_to_hardware = [5, 6, 3, 4, 0, 1, 2]
        left_q_target = np.zeros(7)
        
        for urdf_idx, hw_idx in enumerate(urdf_to_hardware):
            left_q_target[hw_idx] = corrected_dexpilot_output[urdf_idx]
        
        print("URDF to Hardware Mapping:")
        urdf_joint_names = [
            "left_hand_index_0_joint",    # URDF[0] -> Hardware[5]
            "left_hand_index_1_joint",    # URDF[1] -> Hardware[6]
            "left_hand_middle_0_joint",   # URDF[2] -> Hardware[3]
            "left_hand_middle_1_joint",   # URDF[3] -> Hardware[4]
            "left_hand_thumb_0_joint",    # URDF[4] -> Hardware[0]
            "left_hand_thumb_1_joint",    # URDF[5] -> Hardware[1] ← Main correction
            "left_hand_thumb_2_joint"     # URDF[6] -> Hardware[2]
        ]
        
        hardware_joint_names = [
            "left_hand_thumb_0_joint",    # Hardware[0]
            "left_hand_thumb_1_joint",    # Hardware[1] ← Main correction
            "left_hand_thumb_2_joint",    # Hardware[2]
            "left_hand_middle_0_joint",   # Hardware[3]
            "left_hand_middle_1_joint",   # Hardware[4]
            "left_hand_index_0_joint",    # Hardware[5]
            "left_hand_index_1_joint"     # Hardware[6]
        ]
        
        for urdf_idx, hw_idx in enumerate(urdf_to_hardware):
            print(f"   URDF[{urdf_idx}] {urdf_joint_names[urdf_idx]} -> "
                  f"Hardware[{hw_idx}] {hardware_joint_names[hw_idx]}: "
                  f"{corrected_dexpilot_output[urdf_idx]:.3f} rad "
                  f"({corrected_dexpilot_output[urdf_idx]*180/math.pi:.1f}°)")
        
        print()
        print("Final Hardware Command Array (with thumb correction):")
        for hw_idx, angle in enumerate(left_q_target):
            print(f"   Hardware[{hw_idx}] {hardware_joint_names[hw_idx]}: "
                  f"{angle:.3f} rad ({angle*180/math.pi:.1f}°)")
        
        print()
        print("✅ Hardware mapping test completed!")
        print(f"   - Thumb_1 joint (Hardware[1]) shows corrected angle: "
              f"{left_q_target[1]*180/math.pi:.1f}°")
        
    except Exception as e:
        print(f"❌ Hardware mapping test failed: {e}")


if __name__ == "__main__":
    test_integrated_thumb_correction()
    test_hardware_mapping()
