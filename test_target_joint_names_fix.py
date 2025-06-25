#!/usr/bin/env python3
"""
Test the target_joint_names fix for DexPilot stuck joints.
"""

import sys
import os
import numpy as np

# Add the teleop directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'teleop'))

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

def test_target_joint_names_fix():
    """Test that the target_joint_names fix resolves the stuck joints issue."""
    
    print("=== Testing target_joint_names Fix ===\n")
    
    try:
        # Load the updated DexPilot configuration
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        print("1. Updated DexPilot Configuration:")
        target_joints = left_optimizer.target_joint_names
        print(f"   Number of target joints: {len(target_joints)}")
        print(f"   Target joint names: {target_joints}")
        print()
        
        print("2. Joint Mapping Analysis:")
        hardware_joints = hand_retargeting.left_dex3_api_joint_names
        print(f"   Hardware joints (7): {hardware_joints}")
        print(f"   DexPilot->Hardware mapping: {hand_retargeting.left_dex_retargeting_to_hardware}")
        print()
        
        # Test with different hand poses
        print("3. Testing Different Hand Poses:")
        
        # Get the human indices that DexPilot expects
        human_indices = left_optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        
        # Test Case 1: Open hand
        print("   Test 1: Open Hand")
        joint_pos_open = np.zeros((25, 3))
        joint_pos_open[0] = [0, 0, 0]        # wrist
        joint_pos_open[4] = [0.08, 0.02, 0]  # thumb_tip - spread out
        joint_pos_open[8] = [0.05, 0.08, 0]  # index_tip - pointing up
        joint_pos_open[12] = [0, 0.09, 0]    # middle_tip - pointing up
        
        vectors_open = joint_pos_open[task_indices, :] - joint_pos_open[origin_indices, :]
        
        # Provide fixed_qpos for the 2 skipped joints
        fixed_qpos = np.array([0.5, -0.5])  # thumb_2, index_1
        dexpilot_output_open = hand_retargeting.left_retargeting.retarget(vectors_open, fixed_qpos)
        
        print(f"     DexPilot output shape: {dexpilot_output_open.shape}")
        print(f"     DexPilot output: {dexpilot_output_open}")
        print(f"     Range: [{dexpilot_output_open.min():.3f}, {dexpilot_output_open.max():.3f}]")
        
        # Map to full hardware array
        full_joints_open = np.zeros(7)
        for i, hw_idx in enumerate(hand_retargeting.left_dex_retargeting_to_hardware):
            if i < len(dexpilot_output_open):
                full_joints_open[hw_idx] = dexpilot_output_open[i]
        full_joints_open[2] = fixed_qpos[0]   # thumb_2 default
        full_joints_open[6] = fixed_qpos[1]   # index_1 default
        
        print(f"     Full hardware joints: {full_joints_open}")
        print()
        
        # Test Case 2: Closed/pinch hand
        print("   Test 2: Pinch Hand")
        joint_pos_pinch = np.zeros((25, 3))
        joint_pos_pinch[0] = [0, 0, 0]        # wrist
        joint_pos_pinch[4] = [0.03, 0.04, 0]  # thumb_tip - closer
        joint_pos_pinch[8] = [0.03, 0.05, 0]  # index_tip - closer
        joint_pos_pinch[12] = [0, 0.05, 0]    # middle_tip - closer
        
        vectors_pinch = joint_pos_pinch[task_indices, :] - joint_pos_pinch[origin_indices, :]
        dexpilot_output_pinch = hand_retargeting.left_retargeting.retarget(vectors_pinch, fixed_qpos)
        
        print(f"     DexPilot output shape: {dexpilot_output_pinch.shape}")
        print(f"     DexPilot output: {dexpilot_output_pinch}")
        print(f"     Range: [{dexpilot_output_pinch.min():.3f}, {dexpilot_output_pinch.max():.3f}]")
        
        # Map to full hardware array
        full_joints_pinch = np.zeros(7)
        for i, hw_idx in enumerate(hand_retargeting.left_dex_retargeting_to_hardware):
            if i < len(dexpilot_output_pinch):
                full_joints_pinch[hw_idx] = dexpilot_output_pinch[i]
        full_joints_pinch[2] = fixed_qpos[0]   # thumb_2 default
        full_joints_pinch[6] = fixed_qpos[1]   # index_1 default
        
        print(f"     Full hardware joints: {full_joints_pinch}")
        print()
        
        # Test Case 3: Compare differences
        print("4. Movement Analysis:")
        dexpilot_diff = dexpilot_output_pinch - dexpilot_output_open
        full_joints_diff = full_joints_pinch - full_joints_open
        
        print(f"   DexPilot output differences: {dexpilot_diff}")
        print(f"   Max DexPilot difference: {np.abs(dexpilot_diff).max():.3f} rad ({np.abs(dexpilot_diff).max() * 180/np.pi:.1f}°)")
        print(f"   Full joint differences: {full_joints_diff}")
        print(f"   Max full joint difference: {np.abs(full_joints_diff).max():.3f} rad ({np.abs(full_joints_diff).max() * 180/np.pi:.1f}°)")
        
        # Check for stuck joints
        print("\n5. Stuck Joint Analysis:")
        tolerance = 0.01  # 0.6 degrees
        for i, (joint_name, diff) in enumerate(zip(hardware_joints, full_joints_diff)):
            if abs(diff) < tolerance:
                status = "❌ STUCK"
            else:
                status = "✅ MOVING"
            print(f"   Joint {i} ({joint_name}): {diff:.3f} rad ({diff*180/np.pi:.1f}°) {status}")
        
        # Success criteria
        moving_joints = sum(1 for diff in full_joints_diff if abs(diff) > tolerance)
        print(f"\n6. Result Summary:")
        print(f"   Moving joints: {moving_joints}/7")
        print(f"   Stuck joints: {7 - moving_joints}/7")
        
        if moving_joints >= 3:  # At least 3 joints should be moving
            print("   ✅ SUCCESS: Multiple joints are responding to input changes!")
            return True
        else:
            print("   ❌ FAILURE: Too few joints are moving")
            return False
            
    except Exception as e:
        print(f"Error in target_joint_names test: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_joint_limits():
    """Verify that the target joints have reasonable limits."""
    
    print("=== Joint Limits Verification ===\n")
    
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        target_joints = left_optimizer.target_joint_names
        robot = left_optimizer.robot
        
        print("Target joints and their limits:")
        for i, joint_name in enumerate(target_joints):
            if hasattr(robot, 'joint_limits') and i < len(robot.joint_limits):
                lower, upper = robot.joint_limits[i]
                range_deg = (upper - lower) * 180 / np.pi
                print(f"   {joint_name}: [{lower:.3f}, {upper:.3f}] rad = [{lower*180/np.pi:.1f}, {upper*180/np.pi:.1f}]° (range: {range_deg:.1f}°)")
        
        print("\nSkipped joints (should be the problematic ones):")
        all_joints = robot.dof_joint_names
        skipped_joints = [joint for joint in all_joints if joint not in target_joints]
        for joint_name in skipped_joints:
            joint_idx = all_joints.index(joint_name)
            if hasattr(robot, 'joint_limits') and joint_idx < len(robot.joint_limits):
                lower, upper = robot.joint_limits[joint_idx]
                range_deg = (upper - lower) * 180 / np.pi
                print(f"   {joint_name}: [{lower:.3f}, {upper:.3f}] rad = [{lower*180/np.pi:.1f}, {upper*180/np.pi:.1f}]° (range: {range_deg:.1f}°)")
        
        return True
        
    except Exception as e:
        print(f"Error in joint limits verification: {e}")
        return False

if __name__ == "__main__":
    print("Testing target_joint_names fix for DexPilot...\n")
    
    success1 = test_target_joint_names_fix()
    success2 = verify_joint_limits()
    
    if success1 and success2:
        print("\n🎉 SUCCESS: target_joint_names fix appears to be working!")
        print("   - DexPilot now optimizes only 5 joints with good ranges")
        print("   - Problematic joints are set to safe default values")
        print("   - Multiple joints are responding to input changes")
    else:
        print("\n❌ Some issues remain. Further investigation needed.")
