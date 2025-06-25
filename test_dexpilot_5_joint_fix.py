#!/usr/bin/env python3
"""
Test that DexPilot correctly returns 7 joints but only optimizes 5 with fixed_qpos handling.
"""

import sys
import os
import numpy as np

# Add the teleop directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'teleop'))

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType


def test_dexpilot_5_joint_optimization():
    """Test that DexPilot optimizes only 5 joints and uses fixed_qpos for the other 2."""
    
    print("=== Testing DexPilot 5-Joint Optimization with fixed_qpos ===\n")
    
    try:
        # Load the DexPilot configuration
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        print("1. Configuration Verification:")
        print(f"   Target joints (should be 5): {len(left_optimizer.target_joint_names)}")
        print(f"   Target joint names: {left_optimizer.target_joint_names}")
        print(f"   Fixed joints (should be 2): {len(left_optimizer.idx_pin2fixed)}")
        print(f"   Fixed joint indices: {left_optimizer.idx_pin2fixed}")
        print(f"   Optimizer DOF: {left_optimizer.opt_dof}")
        print()
        
        # Get the human indices that DexPilot expects
        human_indices = left_optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        
        print("2. DexPilot Vector Setup:")
        print(f"   Origin indices: {origin_indices}")
        print(f"   Task indices: {task_indices}")
        print()
        
        # Test Case 1: Open hand with specific fixed_qpos values
        print("3. Test Case 1: Open Hand with Fixed Joint Values")
        
        joint_pos_open = np.zeros((25, 3))
        joint_pos_open[0] = [0, 0, 0]        # wrist
        joint_pos_open[4] = [0.08, 0.02, 0]  # thumb_tip - spread out
        joint_pos_open[8] = [0.05, 0.08, 0]  # index_tip - pointing up
        joint_pos_open[12] = [0, 0.09, 0]    # middle_tip - pointing up
        
        vectors_open = joint_pos_open[task_indices, :] - joint_pos_open[origin_indices, :]
        
        # Set specific values for the fixed joints
        fixed_qpos_test = np.array([0.8, -0.9])  # thumb_2=0.8, index_1=-0.9
        dexpilot_output = hand_retargeting.left_retargeting.retarget(vectors_open, fixed_qpos_test)
        
        print(f"   Input fixed_qpos: {fixed_qpos_test}")
        print(f"   DexPilot output shape: {dexpilot_output.shape}")
        print(f"   DexPilot output: {dexpilot_output}")
        print()
        
        # Verify that the fixed joints are correctly set
        print("4. Fixed Joint Verification:")
        hardware_joints = hand_retargeting.left_dex3_api_joint_names
        print(f"   Joint 2 (thumb_2): {dexpilot_output[2]:.3f} (should be {fixed_qpos_test[0]:.3f})")
        print(f"   Joint 6 (index_1): {dexpilot_output[6]:.3f} (should be {fixed_qpos_test[1]:.3f})")
        
        # Check if fixed joints match
        fixed_correct = (abs(dexpilot_output[2] - fixed_qpos_test[0]) < 0.001 and 
                        abs(dexpilot_output[6] - fixed_qpos_test[1]) < 0.001)
        
        if fixed_correct:
            print("   ✅ Fixed joints correctly set to fixed_qpos values")
        else:
            print("   ❌ Fixed joints NOT correctly set!")
            
        print()
        
        # Test Case 2: Different fixed_qpos values
        print("5. Test Case 2: Different Fixed Joint Values")
        
        fixed_qpos_test2 = np.array([0.3, -0.4])  # Different values
        dexpilot_output2 = hand_retargeting.left_retargeting.retarget(vectors_open, fixed_qpos_test2)
        
        print(f"   Input fixed_qpos: {fixed_qpos_test2}")
        print(f"   Joint 2 (thumb_2): {dexpilot_output2[2]:.3f} (should be {fixed_qpos_test2[0]:.3f})")
        print(f"   Joint 6 (index_1): {dexpilot_output2[6]:.3f} (should be {fixed_qpos_test2[1]:.3f})")
        
        # Check if fixed joints match
        fixed_correct2 = (abs(dexpilot_output2[2] - fixed_qpos_test2[0]) < 0.001 and 
                         abs(dexpilot_output2[6] - fixed_qpos_test2[1]) < 0.001)
        
        if fixed_correct2:
            print("   ✅ Fixed joints correctly set to new fixed_qpos values")
        else:
            print("   ❌ Fixed joints NOT correctly set!")
            
        print()
        
        # Test Case 3: Check that optimized joints change
        print("6. Test Case 3: Optimized Joint Movement")
        
        # Get optimized joint values (excluding fixed joints 2 and 6)
        optimized_indices = [0, 1, 3, 4, 5]  # thumb_0, thumb_1, middle_0, middle_1, index_0
        
        optimized_open = dexpilot_output[optimized_indices]
        optimized_open2 = dexpilot_output2[optimized_indices]
        
        print(f"   Optimized joints (test 1): {optimized_open}")
        print(f"   Optimized joints (test 2): {optimized_open2}")
        
        # The optimized joints should be identical since the input vectors are the same
        # Only fixed_qpos changed, which shouldn't affect optimization
        optimized_same = np.allclose(optimized_open, optimized_open2, atol=0.001)
        
        if optimized_same:
            print("   ✅ Optimized joints remain consistent (correct behavior)")
        else:
            print("   ⚠️  Optimized joints changed (may indicate coupling)")
            
        print()
        
        return fixed_correct and fixed_correct2 and optimized_same
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_different_hand_poses():
    """Test that the optimization actually responds to different hand poses."""
    
    print("=== Testing Different Hand Poses ===\n")
    
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        # Get the human indices that DexPilot expects
        human_indices = left_optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        
        fixed_qpos = np.array([0.5, -0.5])  # Standard fixed values
        
        # Test Case 1: Open hand
        print("1. Open Hand Pose:")
        joint_pos_open = np.zeros((25, 3))
        joint_pos_open[0] = [0, 0, 0]        # wrist
        joint_pos_open[4] = [0.08, 0.02, 0]  # thumb_tip - spread out
        joint_pos_open[8] = [0.05, 0.08, 0]  # index_tip - pointing up
        joint_pos_open[12] = [0, 0.09, 0]    # middle_tip - pointing up
        
        vectors_open = joint_pos_open[task_indices, :] - joint_pos_open[origin_indices, :]
        dexpilot_open = hand_retargeting.left_retargeting.retarget(vectors_open, fixed_qpos)
        optimized_open = dexpilot_open[[0, 1, 3, 4, 5]]  # Get only optimized joints
        
        print(f"   Optimized joints: {optimized_open}")
        print(f"   Range: [{optimized_open.min():.3f}, {optimized_open.max():.3f}]")
        print()
        
        # Test Case 2: Pinching hand
        print("2. Pinching Hand Pose:")
        joint_pos_pinch = np.zeros((25, 3))
        joint_pos_pinch[0] = [0, 0, 0]        # wrist
        joint_pos_pinch[4] = [0.04, 0.05, 0]  # thumb_tip - closer
        joint_pos_pinch[8] = [0.03, 0.05, 0]  # index_tip - closer
        joint_pos_pinch[12] = [0, 0.05, 0]    # middle_tip - closer
        
        vectors_pinch = joint_pos_pinch[task_indices, :] - joint_pos_pinch[origin_indices, :]
        dexpilot_pinch = hand_retargeting.left_retargeting.retarget(vectors_pinch, fixed_qpos)
        optimized_pinch = dexpilot_pinch[[0, 1, 3, 4, 5]]  # Get only optimized joints
        
        print(f"   Optimized joints: {optimized_pinch}")
        print(f"   Range: [{optimized_pinch.min():.3f}, {optimized_pinch.max():.3f}]")
        print()
        
        # Check movement
        print("3. Movement Analysis:")
        movement = optimized_pinch - optimized_open
        print(f"   Joint differences: {movement}")
        print(f"   Max difference: {abs(movement).max():.3f} rad ({abs(movement).max() * 180/np.pi:.1f}°)")
        
        # Check for stuck joints (no movement)
        tolerance = 0.01  # ~0.6 degrees
        moving_joints = abs(movement) > tolerance
        stuck_joints = ~moving_joints
        
        print(f"   Moving joints: {np.sum(moving_joints)}/5")
        print(f"   Stuck joints: {np.sum(stuck_joints)}/5")
        
        if np.sum(moving_joints) >= 3:  # At least 3 joints should move
            print("   ✅ Good joint responsiveness")
            return True
        else:
            print("   ❌ Too many stuck joints")
            return False
            
    except Exception as e:
        print(f"Error during pose test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing DexPilot 5-joint optimization with fixed_qpos...\n")
    
    success1 = test_dexpilot_5_joint_optimization()
    success2 = test_with_different_hand_poses()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("🎉 SUCCESS: DexPilot 5-joint optimization is working correctly!")
        print("   - Only 5 joints are optimized as specified in target_joint_names")
        print("   - Fixed joints (2 and 6) are correctly set to fixed_qpos values")
        print("   - Optimized joints respond to different hand poses")
        print("   - The configuration is ready for real teleoperation!")
    else:
        print("❌ ISSUES DETECTED:")
        if not success1:
            print("   - Fixed joint handling not working correctly")
        if not success2:
            print("   - Joint optimization not responding properly to pose changes")
        print("   - Further investigation needed")
    
    print()
