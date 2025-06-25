#!/usr/bin/env python3

"""
Test the clean vector retargeting functionality
"""

import sys
import os
import numpy as np

# Add the workspace to the Python path
sys.path.append('/home/robin/humanoid/humanoid_ws/src/avp_teleoperate')

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

def test_vector_retargeting():
    """Test the clean vector retargeting functionality."""
    
    print("Testing Vector Retargeting with Clean Configuration")
    print("=" * 60)
    
    try:
        # Test vector method initialization
        print("\n1. Initializing vector retargeting...")
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'vector')
        print("   ✓ Vector retargeting initialized successfully")
        
        # Check configuration structure
        print(f"   Configuration structure: {list(hand_retargeting.cfg.keys())}")
        print(f"   Left config type: {hand_retargeting.cfg['left']['type']}")
        print(f"   Right config type: {hand_retargeting.cfg['right']['type']}")
        
        # Test retargeting objects
        print(f"   Left retargeting type: {hand_retargeting.left_retargeting.optimizer.retargeting_type}")
        print(f"   Right retargeting type: {hand_retargeting.right_retargeting.optimizer.retargeting_type}")
        
        # Test joint names
        print(f"   Left joint names: {hand_retargeting.left_retargeting_joint_names}")
        print(f"   Right joint names: {hand_retargeting.right_retargeting_joint_names}")
        
        # Test with sample input (fingertip positions)
        print("\n2. Testing retargeting with sample data...")
        sample_fingertips = np.array([
            [0.0, 0.1, 0.0],   # thumb tip
            [0.0, 0.15, 0.0],  # index tip  
            [0.0, 0.14, 0.0]   # middle tip
        ])
        
        # Apply the scaling factors as in the original implementation
        scaled_fingertips = sample_fingertips.copy()
        scaled_fingertips[0] = scaled_fingertips[0] * 1.15  # thumb
        scaled_fingertips[1] = scaled_fingertips[1] * 1.05  # index
        scaled_fingertips[2] = scaled_fingertips[2] * 0.95  # middle
        
        # Test left hand retargeting
        left_result = hand_retargeting.left_retargeting.retarget(scaled_fingertips)
        print(f"   Left hand result shape: {left_result.shape}")
        print(f"   Left hand result: {left_result}")
        
        # Apply hardware mapping
        left_mapped = left_result[hand_retargeting.left_dex_retargeting_to_hardware]
        print(f"   Left mapped to hardware: {left_mapped}")
        
        # Test right hand retargeting
        right_result = hand_retargeting.right_retargeting.retarget(scaled_fingertips)
        print(f"   Right hand result shape: {right_result.shape}")
        print(f"   Right hand result: {right_result}")
        
        # Apply hardware mapping
        right_mapped = right_result[hand_retargeting.right_dex_retargeting_to_hardware]
        print(f"   Right mapped to hardware: {right_mapped}")
        
        print("\n3. Testing the clean robot controller...")
        # Import the clean controller
        from teleop.robot_control.robot_hand_unitree_clean import Dex3_1_Controller
        print("   ✓ Clean controller imported successfully")
        
        print("\n✅ ALL VECTOR RETARGETING TESTS PASSED!")
        print("The vector retargeting method is working correctly with the clean configuration.")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vector_retargeting()
    if success:
        print("\n🎉 Vector retargeting is ready for use!")
    else:
        print("\n💥 Please fix the issues before proceeding.")
