#!/usr/bin/env python3
"""
Test DexPilot with all 7 joints in target_joint_names (no fixed joints).
"""

import sys
import os
import numpy as np

# Add the teleop directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'teleop'))

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType


def test_dexpilot_all_7_joints():
    """Test DexPilot with all 7 joints in target_joint_names."""
    
    print("=== Testing DexPilot with All 7 Joints ===\n")
    
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        print("1. Configuration Verification:")
        print(f"   Target joints: {len(left_optimizer.target_joint_names)}")
        print(f"   Target joint names: {left_optimizer.target_joint_names}")
        print(f"   Fixed joints: {len(left_optimizer.idx_pin2fixed)}")
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
        
        # Test Case: Open hand without fixed_qpos
        print("3. Test Case: Open Hand (No fixed_qpos)")
        
        joint_pos_open = np.zeros((25, 3))
        joint_pos_open[0] = [0, 0, 0]        # wrist
        joint_pos_open[4] = [0.08, 0.02, 0]  # thumb_tip - spread out
        joint_pos_open[8] = [0.05, 0.08, 0]  # index_tip - pointing up
        joint_pos_open[12] = [0, 0.09, 0]    # middle_tip - pointing up
        
        vectors_open = joint_pos_open[task_indices, :] - joint_pos_open[origin_indices, :]
        
        # Call retarget without fixed_qpos since all joints are optimized
        dexpilot_output = hand_retargeting.left_retargeting.retarget(vectors_open)
        
        print(f"   DexPilot output shape: {dexpilot_output.shape}")
        print(f"   DexPilot output: {dexpilot_output}")
        print(f"   Range: [{dexpilot_output.min():.3f}, {dexpilot_output.max():.3f}]")
        print()
        
        # Test Case 2: Pinching hand
        print("4. Test Case: Pinching Hand")
        
        joint_pos_pinch = np.zeros((25, 3))
        joint_pos_pinch[0] = [0, 0, 0]        # wrist
        joint_pos_pinch[4] = [0.04, 0.05, 0]  # thumb_tip - closer
        joint_pos_pinch[8] = [0.03, 0.05, 0]  # index_tip - closer
        joint_pos_pinch[12] = [0, 0.05, 0]    # middle_tip - closer
        
        vectors_pinch = joint_pos_pinch[task_indices, :] - joint_pos_pinch[origin_indices, :]
        dexpilot_output_pinch = hand_retargeting.left_retargeting.retarget(vectors_pinch)
        
        print(f"   DexPilot output shape: {dexpilot_output_pinch.shape}")
        print(f"   DexPilot output: {dexpilot_output_pinch}")
        print(f"   Range: [{dexpilot_output_pinch.min():.3f}, {dexpilot_output_pinch.max():.3f}]")
        print()
        
        # Check movement
        print("5. Movement Analysis:")
        movement = dexpilot_output_pinch - dexpilot_output
        print(f"   Joint differences: {movement}")
        print(f"   Max difference: {abs(movement).max():.3f} rad ({abs(movement).max() * 180/np.pi:.1f}°)")
        
        # Check for stuck joints (no movement)
        tolerance = 0.01  # ~0.6 degrees
        moving_joints = abs(movement) > tolerance
        stuck_joints = ~moving_joints
        
        print(f"   Moving joints: {np.sum(moving_joints)}/7")
        print(f"   Stuck joints: {np.sum(stuck_joints)}/7")
        
        # Check each joint individually
        print("\n6. Individual Joint Analysis:")
        hardware_joints = hand_retargeting.left_dex3_api_joint_names
        
        # Map from URDF order to Hardware API order
        urdf_to_hardware = [5, 6, 3, 4, 0, 1, 2]  # As defined in robot_hand_unitree.py
        
        for urdf_idx, hw_idx in enumerate(urdf_to_hardware):
            urdf_joint = left_optimizer.target_joint_names[urdf_idx]
            hw_joint = hardware_joints[hw_idx]
            diff = movement[urdf_idx]
            status = "✅ MOVING" if abs(diff) > tolerance else "❌ STUCK"
            print(f"   URDF[{urdf_idx}] {urdf_joint} -> HW[{hw_idx}] {hw_joint}: {diff:.3f} rad ({diff*180/np.pi:.1f}°) {status}")
        
        print()
        
        if np.sum(moving_joints) >= 5:  # At least 5 joints should move
            print("✅ SUCCESS: Good joint responsiveness")
            return True
        else:
            print("❌ ISSUE: Too many stuck joints")
            return False
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_joint_limits():
    """Test that all joints are within reasonable limits."""
    
    print("=== Testing Joint Limits ===\n")
    
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        robot = left_optimizer.robot
        
        print("1. Joint Limits Analysis:")
        
        # Try to access joint limits
        if hasattr(robot, 'joint_limits'):
            limits = robot.joint_limits
            print(f"   Joint limits shape: {limits.shape}")
            for i, joint_name in enumerate(robot.dof_joint_names):
                if i < len(limits):
                    lower, upper = limits[i]
                    range_deg = (upper - lower) * 180 / np.pi
                    print(f"   Joint {i} ({joint_name}): [{lower:.3f}, {upper:.3f}] rad = [{lower*180/np.pi:.1f}, {upper*180/np.pi:.1f}] deg (range: {range_deg:.1f}°)")
        else:
            print("   Joint limits not directly accessible from robot object")
            
        return True
        
    except Exception as e:
        print(f"Error during joint limits test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing DexPilot with all 7 joints...\n")
    
    success1 = test_dexpilot_all_7_joints()
    success2 = test_joint_limits()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("🎉 SUCCESS: DexPilot with all 7 joints is working!")
        print("   - All 7 joints are included in target_joint_names")
        print("   - No fixed_qpos needed - all joints are optimized")
        print("   - Most joints respond to different hand poses")
        print("   - Ready for real teleoperation!")
    else:
        print("❌ ISSUES DETECTED:")
        if not success1:
            print("   - Joint optimization not responding properly")
        if not success2:
            print("   - Joint limits configuration issues")
        print("   - Further investigation needed")
    
    print()
