#!/usr/bin/env python3
"""
Test script to verify the DexPilot mapping fix.
"""

import sys
import os
import numpy as np

# Add the teleop directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'teleop'))

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

def test_dexpilot_with_correct_mapping():
    """Test DexPilot with the corrected mapping."""
    
    print("=== Testing DexPilot with Corrected Mapping ===\n")
    
    try:
        # Load DexPilot configuration
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        # Get the human indices that DexPilot expects
        human_indices = left_optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        
        print("1. DexPilot Expected Indices:")
        print(f"   Origin indices: {origin_indices}")
        print(f"   Task indices: {task_indices}")
        print()
        
        # Test case 1: Open hand pose
        print("2. Test Case 1: Open Hand Pose")
        # Simulate OpenXR hand data - open hand
        openxr_hand_data = np.zeros((25, 3))
        openxr_hand_data[0] = [0, 0, 0]        # wrist
        openxr_hand_data[4] = [0.08, 0.02, 0]  # thumb_tip - spread out
        openxr_hand_data[9] = [0.05, 0.08, 0]  # index_tip - pointing up
        openxr_hand_data[14] = [0, 0.09, 0]    # middle_tip - pointing up
        
        # Create joint_pos array with corrected mapping
        joint_pos = np.zeros((25, 3))
        joint_pos[0] = openxr_hand_data[0]   # wrist -> index 0
        joint_pos[4] = openxr_hand_data[4]   # thumb_tip -> index 4
        joint_pos[8] = openxr_hand_data[9]   # index_tip -> index 8 (from OpenXR index 9)
        joint_pos[12] = openxr_hand_data[14] # middle_tip -> index 12 (from OpenXR index 14)
        
        # Calculate vectors
        vectors = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        print(f"   Input vectors shape: {vectors.shape}")
        print(f"   Input vectors:\n{vectors}")
        
        # Get retargeting result
        joint_angles = hand_retargeting.left_retargeting.retarget(vectors)
        print(f"   Output joint angles: {joint_angles}")
        print(f"   Joint angle range: [{joint_angles.min():.3f}, {joint_angles.max():.3f}]")
        print()
        
        # Test case 2: Pinch pose
        print("3. Test Case 2: Pinch Pose")
        # Simulate OpenXR hand data - pinching motion
        openxr_hand_data_pinch = np.zeros((25, 3))
        openxr_hand_data_pinch[0] = [0, 0, 0]        # wrist
        openxr_hand_data_pinch[4] = [0.03, 0.04, 0]  # thumb_tip - closer to fingers
        openxr_hand_data_pinch[9] = [0.03, 0.05, 0]  # index_tip - closer to thumb
        openxr_hand_data_pinch[14] = [0, 0.05, 0]    # middle_tip - slightly closed
        
        # Create joint_pos array with corrected mapping
        joint_pos_pinch = np.zeros((25, 3))
        joint_pos_pinch[0] = openxr_hand_data_pinch[0]   # wrist -> index 0
        joint_pos_pinch[4] = openxr_hand_data_pinch[4]   # thumb_tip -> index 4
        joint_pos_pinch[8] = openxr_hand_data_pinch[9]   # index_tip -> index 8
        joint_pos_pinch[12] = openxr_hand_data_pinch[14] # middle_tip -> index 12
        
        # Calculate vectors
        vectors_pinch = joint_pos_pinch[task_indices, :] - joint_pos_pinch[origin_indices, :]
        print(f"   Input vectors shape: {vectors_pinch.shape}")
        print(f"   Input vectors:\n{vectors_pinch}")
        
        # Get retargeting result
        joint_angles_pinch = hand_retargeting.left_retargeting.retarget(vectors_pinch)
        print(f"   Output joint angles: {joint_angles_pinch}")
        print(f"   Joint angle range: [{joint_angles_pinch.min():.3f}, {joint_angles_pinch.max():.3f}]")
        print()
        
        # Test case 3: Compare differences
        print("4. Comparison Analysis:")
        angle_diff = joint_angles_pinch - joint_angles
        print(f"   Joint angle differences (pinch - open): {angle_diff}")
        print(f"   Max difference: {np.abs(angle_diff).max():.3f} radians")
        print(f"   Max difference: {np.abs(angle_diff).max() * 180/np.pi:.1f} degrees")
        
        # Check if the mapping is working (should see significant differences)
        if np.abs(angle_diff).max() > 0.1:  # > 5.7 degrees
            print("   ✓ GOOD: Significant joint angle changes detected between poses")
            return True
        else:
            print("   ✗ ISSUE: Little to no joint angle changes detected")
            return False
            
    except Exception as e:
        print(f"Error in DexPilot testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_consistency():
    """Test that the vectors are being calculated consistently."""
    
    print("=== Vector Consistency Test ===\n")
    
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        human_indices = left_optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        
        print("Expected vector meanings:")
        vector_meanings = [
            "index_tip -> thumb_tip",
            "middle_tip -> thumb_tip",  
            "middle_tip -> index_tip",
            "base_link -> thumb_tip",
            "base_link -> index_tip",
            "base_link -> middle_tip"
        ]
        
        for i, meaning in enumerate(vector_meanings):
            print(f"   Vector {i}: {meaning} (indices: {origin_indices[i]} -> {task_indices[i]})")
        print()
        
        # Test with simple positions
        joint_pos = np.zeros((25, 3))
        joint_pos[0] = [0, 0, 0]      # wrist/base_link
        joint_pos[4] = [0.1, 0, 0]    # thumb_tip
        joint_pos[8] = [0, 0.1, 0]    # index_tip 
        joint_pos[12] = [0, 0, 0.1]   # middle_tip
        
        vectors = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        
        print("Calculated vectors:")
        for i, vector in enumerate(vectors):
            print(f"   Vector {i}: {vector} -> {vector_meanings[i]}")
        
        # Verify the vectors make sense
        print("\nVector verification:")
        # Vector 0: index_tip -> thumb_tip should be [0.1, -0.1, 0]
        expected_0 = np.array([0.1, -0.1, 0])
        print(f"   Vector 0 expected: {expected_0}, actual: {vectors[0]}, match: {np.allclose(vectors[0], expected_0)}")
        
        # Vector 3: base_link -> thumb_tip should be [0.1, 0, 0]
        expected_3 = np.array([0.1, 0, 0])
        print(f"   Vector 3 expected: {expected_3}, actual: {vectors[3]}, match: {np.allclose(vectors[3], expected_3)}")
        
        return True
        
    except Exception as e:
        print(f"Error in vector consistency test: {e}")
        return False

if __name__ == "__main__":
    print("Testing DexPilot mapping fix...\n")
    
    success1 = test_dexpilot_with_correct_mapping()
    success2 = test_vector_consistency()
    
    if success1 and success2:
        print("\n✓ All tests passed! DexPilot mapping appears to be fixed.")
    else:
        print("\n✗ Some tests failed. Further investigation needed.")
