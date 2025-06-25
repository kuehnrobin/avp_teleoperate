#!/usr/bin/env python3
"""
Test the corrected fixed_qpos mapping in DexPilot retargeting.
"""

import sys
import os
import numpy as np

# Add the teleop directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'teleop'))

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType


def test_fixed_qpos_mapping():
    """Test that fixed_qpos values are correctly mapped to the right joints."""
    
    print("=== Testing Corrected fixed_qpos Mapping ===\n")
    
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        # Get the human indices that DexPilot expects
        human_indices = left_optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        
        # Set up test hand pose
        joint_pos = np.zeros((25, 3))
        joint_pos[0] = [0, 0, 0]        # wrist
        joint_pos[4] = [0.08, 0.02, 0]  # thumb_tip
        joint_pos[8] = [0.05, 0.08, 0]  # index_tip
        joint_pos[12] = [0, 0.09, 0]    # middle_tip
        
        vectors = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        
        print("1. Joint Index Mapping Verification:")
        print("   URDF joint order (robot.dof_joint_names):")
        for i, name in enumerate(left_optimizer.robot.dof_joint_names):
            print(f"     {i}: {name}")
        print()
        
        print("   Fixed joint indices (idx_pin2fixed):", left_optimizer.idx_pin2fixed)
        fixed_joint_names = [left_optimizer.robot.dof_joint_names[i] for i in left_optimizer.idx_pin2fixed]
        print("   Fixed joint names:", fixed_joint_names)
        print()
        
        # Test different fixed_qpos values
        test_cases = [
            ("Case 1: Standard values", np.array([-0.5, 0.5])),
            ("Case 2: Extreme values", np.array([-0.9, 0.8])),
            ("Case 3: Zero values", np.array([0.0, 0.0])),
        ]
        
        for case_name, fixed_qpos in test_cases:
            print(f"2. {case_name}:")
            print(f"   Input fixed_qpos: {fixed_qpos}")
            print(f"   fixed_qpos[0] = {fixed_qpos[0]:.3f} should go to URDF[1] = {fixed_joint_names[0]}")
            print(f"   fixed_qpos[1] = {fixed_qpos[1]:.3f} should go to URDF[6] = {fixed_joint_names[1]}")
            
            dexpilot_output = hand_retargeting.left_retargeting.retarget(vectors, fixed_qpos)
            
            print(f"   DexPilot output (URDF order): {dexpilot_output}")
            print(f"   URDF[1] (index_1): {dexpilot_output[1]:.3f} (should be {fixed_qpos[0]:.3f})")
            print(f"   URDF[6] (thumb_2): {dexpilot_output[6]:.3f} (should be {fixed_qpos[1]:.3f})")
            
            # Check if values match
            index_1_correct = abs(dexpilot_output[1] - fixed_qpos[0]) < 0.001
            thumb_2_correct = abs(dexpilot_output[6] - fixed_qpos[1]) < 0.001
            
            if index_1_correct and thumb_2_correct:
                print("   ✅ Fixed joints correctly set!")
            else:
                print("   ❌ Fixed joints NOT correctly set!")
                if not index_1_correct:
                    print(f"      - index_1 mismatch: expected {fixed_qpos[0]:.3f}, got {dexpilot_output[1]:.3f}")
                if not thumb_2_correct:
                    print(f"      - thumb_2 mismatch: expected {fixed_qpos[1]:.3f}, got {dexpilot_output[6]:.3f}")
            print()
        
        # Test hardware mapping
        print("3. Hardware API Mapping Test:")
        fixed_qpos = np.array([-0.7, 0.6])  # Test values
        dexpilot_output = hand_retargeting.left_retargeting.retarget(vectors, fixed_qpos)
        
        # Apply the URDF to Hardware mapping
        hardware_output = np.zeros(7)
        urdf_to_hardware = [5, 6, 3, 4, 0, 1, 2]  # From our corrected code
        for urdf_idx, hw_idx in enumerate(urdf_to_hardware):
            hardware_output[hw_idx] = dexpilot_output[urdf_idx]
        
        print(f"   URDF output: {dexpilot_output}")
        print(f"   Hardware output: {hardware_output}")
        print()
        
        print("   Hardware joint verification:")
        hardware_joints = [
            'left_hand_thumb_0_joint',    # 0
            'left_hand_thumb_1_joint',    # 1  
            'left_hand_thumb_2_joint',    # 2 <- should be fixed_qpos[1] = 0.6
            'left_hand_middle_0_joint',   # 3
            'left_hand_middle_1_joint',   # 4
            'left_hand_index_0_joint',    # 5
            'left_hand_index_1_joint'     # 6 <- should be fixed_qpos[0] = -0.7
        ]
        
        for i, hw_joint in enumerate(hardware_joints):
            print(f"     Hardware[{i}] {hw_joint}: {hardware_output[i]:.3f}")
        
        # Check hardware fixed joints
        hw_thumb_2_correct = abs(hardware_output[2] - fixed_qpos[1]) < 0.001
        hw_index_1_correct = abs(hardware_output[6] - fixed_qpos[0]) < 0.001
        
        print()
        print("   Hardware fixed joint verification:")
        print(f"     thumb_2 (Hardware[2]): {hardware_output[2]:.3f} (should be {fixed_qpos[1]:.3f}) {'✅' if hw_thumb_2_correct else '❌'}")
        print(f"     index_1 (Hardware[6]): {hardware_output[6]:.3f} (should be {fixed_qpos[0]:.3f}) {'✅' if hw_index_1_correct else '❌'}")
        
        return hw_thumb_2_correct and hw_index_1_correct
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_joint_movement_response():
    """Test that the optimized joints still respond properly to hand pose changes."""
    
    print("=== Testing Joint Movement Response ===\n")
    
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        human_indices = hand_retargeting.left_retargeting.optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        
        fixed_qpos = np.array([-0.5, 0.5])  # Standard fixed values
        
        # Test 1: Open hand
        joint_pos_open = np.zeros((25, 3))
        joint_pos_open[0] = [0, 0, 0]        # wrist
        joint_pos_open[4] = [0.08, 0.02, 0]  # thumb_tip - spread out
        joint_pos_open[8] = [0.05, 0.08, 0]  # index_tip - pointing up
        joint_pos_open[12] = [0, 0.09, 0]    # middle_tip - pointing up
        
        vectors_open = joint_pos_open[task_indices, :] - joint_pos_open[origin_indices, :]
        dexpilot_open = hand_retargeting.left_retargeting.retarget(vectors_open, fixed_qpos)
        
        # Test 2: Pinching hand
        joint_pos_pinch = np.zeros((25, 3))
        joint_pos_pinch[0] = [0, 0, 0]        # wrist
        joint_pos_pinch[4] = [0.04, 0.05, 0]  # thumb_tip - closer
        joint_pos_pinch[8] = [0.03, 0.05, 0]  # index_tip - closer
        joint_pos_pinch[12] = [0, 0.05, 0]    # middle_tip - closer
        
        vectors_pinch = joint_pos_pinch[task_indices, :] - joint_pos_pinch[origin_indices, :]
        dexpilot_pinch = hand_retargeting.left_retargeting.retarget(vectors_pinch, fixed_qpos)
        
        print("1. Joint Movement Analysis:")
        print(f"   Open hand (URDF order): {dexpilot_open}")
        print(f"   Pinch hand (URDF order): {dexpilot_pinch}")
        
        # Check that fixed joints remain the same
        fixed_joints_same = (abs(dexpilot_open[1] - dexpilot_pinch[1]) < 0.001 and 
                           abs(dexpilot_open[6] - dexpilot_pinch[6]) < 0.001)
        
        print(f"   Fixed joints consistent: {'✅' if fixed_joints_same else '❌'}")
        
        # Check that optimized joints move
        optimized_indices = [0, 2, 3, 4, 5]  # All joints except 1 and 6
        movement = dexpilot_pinch - dexpilot_open
        optimized_movement = movement[optimized_indices]
        
        print(f"   Optimized joint movement: {optimized_movement}")
        print(f"   Max movement: {abs(optimized_movement).max():.3f} rad ({abs(optimized_movement).max() * 180/np.pi:.1f}°)")
        
        # Check for responsive joints
        tolerance = 0.01  # ~0.6 degrees
        moving_joints = abs(optimized_movement) > tolerance
        responsive_count = np.sum(moving_joints)
        
        print(f"   Responsive joints: {responsive_count}/5")
        
        return fixed_joints_same and responsive_count >= 3
        
    except Exception as e:
        print(f"Error during movement test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing corrected fixed_qpos mapping for DexPilot...\n")
    
    success1 = test_fixed_qpos_mapping()
    success2 = test_joint_movement_response()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("🎉 SUCCESS: Fixed_qpos mapping is now working correctly!")
        print("   - Fixed joints are properly set to specified values")
        print("   - Hardware API mapping is correct")
        print("   - Optimized joints respond to hand pose changes")
        print("   - The DexPilot retargeting is ready for teleoperation!")
    else:
        print("❌ ISSUES DETECTED:")
        if not success1:
            print("   - Fixed_qpos mapping still has issues")
        if not success2:
            print("   - Joint movement response problems")
        print("   - Further investigation needed")
    
    print()
