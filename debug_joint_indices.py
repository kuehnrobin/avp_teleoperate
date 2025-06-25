#!/usr/bin/env python3
"""
Debug the joint index mapping in DexPilot to understand why fixed_qpos isn't working.
"""

import sys
import os
import numpy as np

# Add the teleop directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'teleop'))

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType


def debug_joint_indices():
    """Debug the joint index mapping to understand the fixed_qpos issue."""
    
    print("=== Debugging Joint Index Mapping ===\n")
    
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        robot = left_optimizer.robot
        
        print("1. Robot Joint Information:")
        print(f"   Total robot DOF: {robot.dof}")
        print(f"   All robot joint names: {robot.dof_joint_names}")
        print()
        
        print("2. Target Joint Configuration:")
        print(f"   Target joint names: {left_optimizer.target_joint_names}")
        print(f"   Number of target joints: {len(left_optimizer.target_joint_names)}")
        print()
        
        print("3. Index Mappings:")
        print(f"   idx_pin2target: {left_optimizer.idx_pin2target}")
        print(f"   idx_pin2fixed: {left_optimizer.idx_pin2fixed}")
        print()
        
        print("4. Detailed Joint Analysis:")
        for i, joint_name in enumerate(robot.dof_joint_names):
            if i in left_optimizer.idx_pin2target:
                target_idx = list(left_optimizer.idx_pin2target).index(i)
                status = f"TARGET (position {target_idx} in optimization)"
            elif i in left_optimizer.idx_pin2fixed:
                fixed_idx = list(left_optimizer.idx_pin2fixed).index(i)
                status = f"FIXED (position {fixed_idx} in fixed_qpos)"
            else:
                status = "UNKNOWN"
            print(f"   Joint {i}: {joint_name} -> {status}")
        print()
        
        print("5. Hardware API Joint Mapping:")
        hardware_joints = hand_retargeting.left_dex3_api_joint_names
        print(f"   Hardware API joints: {hardware_joints}")
        
        for i, hw_joint in enumerate(hardware_joints):
            if hw_joint in robot.dof_joint_names:
                robot_idx = robot.dof_joint_names.index(hw_joint)
                print(f"   Hardware[{i}] {hw_joint} -> Robot joint {robot_idx}")
            else:
                print(f"   Hardware[{i}] {hw_joint} -> NOT FOUND in robot")
        print()
        
        print("6. Expected vs Actual Fixed Joints:")
        expected_fixed_joints = ['left_hand_thumb_2_joint', 'left_hand_index_1_joint']
        print(f"   Expected fixed joints: {expected_fixed_joints}")
        
        actual_fixed_joints = [robot.dof_joint_names[i] for i in left_optimizer.idx_pin2fixed]
        print(f"   Actual fixed joints: {actual_fixed_joints}")
        
        if set(expected_fixed_joints) == set(actual_fixed_joints):
            print("   ✅ Fixed joints match expectation")
        else:
            print("   ❌ Fixed joints do NOT match expectation")
            print(f"   Missing: {set(expected_fixed_joints) - set(actual_fixed_joints)}")
            print(f"   Extra: {set(actual_fixed_joints) - set(expected_fixed_joints)}")
        
        return left_optimizer
        
    except Exception as e:
        print(f"Error during debug: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_urdf_joint_order():
    """Check the URDF joint order to understand the mismatch."""
    
    print("=== URDF Joint Order Analysis ===\n")
    
    try:
        # Load URDF directly to see the actual joint order
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        robot = hand_retargeting.left_retargeting.optimizer.robot
        
        print("From URDF (robot.dof_joint_names):")
        for i, joint_name in enumerate(robot.dof_joint_names):
            print(f"   {i}: {joint_name}")
        print()
        
        print("Expected hardware API order:")
        expected = [
            'left_hand_thumb_0_joint',    # 0
            'left_hand_thumb_1_joint',    # 1  
            'left_hand_thumb_2_joint',    # 2 <- should be fixed
            'left_hand_middle_0_joint',   # 3
            'left_hand_middle_1_joint',   # 4
            'left_hand_index_0_joint',    # 5
            'left_hand_index_1_joint'     # 6 <- should be fixed
        ]
        
        for i, joint_name in enumerate(expected):
            print(f"   {i}: {joint_name}")
        print()
        
        # Check if they match
        matches = all(robot.dof_joint_names[i] == expected[i] for i in range(min(len(robot.dof_joint_names), len(expected))))
        
        if matches:
            print("✅ URDF joint order matches hardware API expectation")
        else:
            print("❌ URDF joint order does NOT match hardware API expectation")
            print("   This explains the fixed_qpos mapping issue!")
            
    except Exception as e:
        print(f"Error during URDF debug: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Debugging DexPilot joint index mapping...\n")
    
    optimizer = debug_joint_indices()
    if optimizer:
        debug_urdf_joint_order()
    
    print("\nDebug complete.")
