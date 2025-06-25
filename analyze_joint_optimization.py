#!/usr/bin/env python3
"""
Investigate which joints DexPilot is optimizing and identify the issue with target_joint_names.
"""

import sys
import os
import numpy as np

# Add the teleop directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'teleop'))

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

def analyze_joint_optimization():
    """Analyze which joints DexPilot is optimizing."""
    
    print("=== Joint Optimization Analysis ===\n")
    
    try:
        # Load DexPilot configuration
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        print("1. DexPilot Target Joints:")
        target_joints = left_optimizer.target_joint_names
        print(f"   Number of target joints: {len(target_joints)}")
        print(f"   Target joint names: {target_joints}")
        print()
        
        print("2. Robot DOF Information:")
        robot = left_optimizer.robot
        print(f"   Total robot DOF: {robot.dof}")
        print(f"   All joint names: {robot.dof_joint_names}")
        print()
        
        print("3. Joint Limits Analysis:")
        # Get joint limits from the robot model
        if hasattr(robot, 'joint_limits'):
            limits = robot.joint_limits
            print(f"   Joint limits shape: {limits.shape}")
            for i, joint_name in enumerate(robot.dof_joint_names):
                if i < len(limits):
                    lower, upper = limits[i]
                    range_deg = (upper - lower) * 180 / np.pi
                    print(f"   Joint {i} ({joint_name}): [{lower:.3f}, {upper:.3f}] rad = [{lower*180/np.pi:.1f}, {upper*180/np.pi:.1f}] deg (range: {range_deg:.1f}°)")
        else:
            print("   Joint limits not directly accessible")
        print()
        
        print("4. Index Mapping:")
        print(f"   idx_pin2target: {left_optimizer.idx_pin2target}")
        print(f"   idx_pin2fixed: {left_optimizer.idx_pin2fixed}")
        print()
        
        print("5. Comparison with Hardware Joint Order:")
        hardware_joints = hand_retargeting.left_dex3_api_joint_names
        print(f"   Hardware API joint order: {hardware_joints}")
        print()
        
        # Map target joints to hardware joints
        print("6. Target Joint to Hardware Mapping:")
        for i, target_joint in enumerate(target_joints):
            if target_joint in hardware_joints:
                hw_index = hardware_joints.index(target_joint)
                print(f"   Target joint {i} ({target_joint}) -> Hardware index {hw_index}")
            else:
                print(f"   Target joint {i} ({target_joint}) -> NOT FOUND in hardware joints!")
        
        return left_optimizer
        
    except Exception as e:
        print(f"Error in joint optimization analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_target_joint_names_config():
    """Test what happens if we add target_joint_names to our DexPilot config."""
    
    print("=== Testing target_joint_names Configuration ===\n")
    
    # Based on the URDF, let's identify which joints are most important
    # From the URDF we can see:
    # - left_hand_thumb_0_joint: thumb base rotation
    # - left_hand_thumb_1_joint: thumb bend
    # - left_hand_thumb_2_joint: thumb tip (this is joint 2 - stuck!)
    # - left_hand_middle_0_joint: middle base
    # - left_hand_middle_1_joint: middle tip  
    # - left_hand_index_0_joint: index base
    # - left_hand_index_1_joint: index tip (this is joint 6 - stuck!)
    
    print("URDF Joint Analysis:")
    print("   Joint 0: left_hand_thumb_0_joint (thumb base, -60° to +60°)")
    print("   Joint 1: left_hand_thumb_1_joint (thumb bend, -41° to +53°)")
    print("   Joint 2: left_hand_thumb_2_joint (thumb tip, 0° to +100°) ← STUCK AT UPPER LIMIT")
    print("   Joint 3: left_hand_middle_0_joint (middle base, -90° to 0°)")
    print("   Joint 4: left_hand_middle_1_joint (middle tip, -100° to 0°)")
    print("   Joint 5: left_hand_index_0_joint (index base, -90° to 0°)")
    print("   Joint 6: left_hand_index_1_joint (index tip, -100° to 0°) ← STUCK AT UPPER LIMIT")
    print()
    
    print("HYPOTHESIS:")
    print("   Joints 2 and 6 (thumb_tip and index_tip) might be hitting their upper limits")
    print("   because DexPilot is trying to optimize ALL joints, including these tip joints")
    print("   which have very restrictive ranges.")
    print()
    
    print("SOLUTION:")
    print("   Add target_joint_names to only optimize the base joints that have good range:")
    print("   - left_hand_thumb_0_joint (base rotation)")
    print("   - left_hand_thumb_1_joint (bend)")
    print("   - left_hand_middle_0_joint (base)")
    print("   - left_hand_middle_1_joint (tip) - has good range")
    print("   - left_hand_index_0_joint (base)")
    print("   Skip joint 2 (thumb_2) and joint 6 (index_1) which have limited ranges")
    
    return True

if __name__ == "__main__":
    print("Analyzing joint optimization for DexPilot...\n")
    
    optimizer = analyze_joint_optimization()
    if optimizer:
        test_target_joint_names_config()
    
    print("\nAnalysis complete.")
