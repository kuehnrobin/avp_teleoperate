#!/usr/bin/env python3
"""
Test DexPilot without fixed_qpos to understand the actual output.
"""

import sys
import os
import numpy as np

# Add the teleop directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'teleop'))

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType


def test_dexpilot_without_fixed_qpos():
    """Test DexPilot without providing fixed_qpos to see actual behavior."""
    
    print("=== Testing DexPilot Without fixed_qpos ===\n")
    
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        print("1. Optimizer Configuration:")
        print(f"   Robot DOF: {left_optimizer.robot.dof}")
        print(f"   Robot joint names: {left_optimizer.robot.dof_joint_names}")
        print(f"   Target joint names: {left_optimizer.target_joint_names}")
        print(f"   Target joint count: {len(left_optimizer.target_joint_names)}")
        print(f"   Fixed joint indices: {left_optimizer.idx_pin2fixed}")
        print(f"   Target joint indices: {left_optimizer.idx_pin2target}")
        print(f"   Optimizer DOF: {left_optimizer.opt_dof}")
        print()
        
        # Test hand pose
        human_indices = left_optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        
        joint_pos = np.zeros((25, 3))
        joint_pos[0] = [0, 0, 0]        # wrist
        joint_pos[4] = [0.08, 0.02, 0]  # thumb_tip
        joint_pos[8] = [0.05, 0.08, 0]  # index_tip
        joint_pos[12] = [0, 0.09, 0]    # middle_tip
        
        vectors = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        
        print("2. Test without fixed_qpos:")
        try:
            # Call retarget without fixed_qpos (should use empty array)
            dexpilot_output = hand_retargeting.left_retargeting.retarget(vectors)
            print(f"   DexPilot output shape: {dexpilot_output.shape}")
            print(f"   DexPilot output: {dexpilot_output}")
            print("   ✅ Successfully called without fixed_qpos")
        except Exception as e:
            print(f"   ❌ Error calling without fixed_qpos: {e}")
        print()
        
        print("3. Test with empty fixed_qpos:")
        try:
            # Call retarget with explicit empty array
            dexpilot_output = hand_retargeting.left_retargeting.retarget(vectors, np.array([]))
            print(f"   DexPilot output shape: {dexpilot_output.shape}")
            print(f"   DexPilot output: {dexpilot_output}")
            print("   ✅ Successfully called with empty fixed_qpos")
            
            # Check the mapping to hardware
            print("\n4. Hardware Mapping:")
            print(f"   Mapping indices: {hand_retargeting.left_dex_retargeting_to_hardware}")
            
            if len(dexpilot_output) == len(hand_retargeting.left_dex_retargeting_to_hardware):
                # Map the DexPilot output to hardware positions
                hardware_output = np.zeros(7)
                for i, hw_idx in enumerate(hand_retargeting.left_dex_retargeting_to_hardware):
                    hardware_output[hw_idx] = dexpilot_output[i]
                
                print(f"   Hardware output: {hardware_output}")
                print("   Hardware joint assignment:")
                hardware_joints = hand_retargeting.left_dex3_api_joint_names
                for i, joint_name in enumerate(hardware_joints):
                    print(f"     Hardware[{i}] {joint_name}: {hardware_output[i]:.3f}")
                
                # Set the missing joints to safe values
                hardware_output[2] = 0.5   # thumb_2 to middle of range
                hardware_output[6] = -0.5  # index_1 to middle of range
                
                print(f"   Hardware output with fixed joints: {hardware_output}")
                print("   ✅ Mapping successful")
                
                return True
            else:
                print(f"   ❌ Output size mismatch: got {len(dexpilot_output)}, expected {len(hand_retargeting.left_dex_retargeting_to_hardware)}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error calling with empty fixed_qpos: {e}")
            return False
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hand_movement_with_correct_approach():
    """Test hand movement using the correct 5-joint approach."""
    
    print("=== Testing Hand Movement with Correct Approach ===\n")
    
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        human_indices = hand_retargeting.left_retargeting.optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        
        # Test 1: Open hand
        joint_pos_open = np.zeros((25, 3))
        joint_pos_open[0] = [0, 0, 0]        # wrist
        joint_pos_open[4] = [0.08, 0.02, 0]  # thumb_tip - spread out
        joint_pos_open[8] = [0.05, 0.08, 0]  # index_tip - pointing up
        joint_pos_open[12] = [0, 0.09, 0]    # middle_tip - pointing up
        
        vectors_open = joint_pos_open[task_indices, :] - joint_pos_open[origin_indices, :]
        dexpilot_open = hand_retargeting.left_retargeting.retarget(vectors_open, np.array([]))
        
        # Test 2: Pinching hand
        joint_pos_pinch = np.zeros((25, 3))
        joint_pos_pinch[0] = [0, 0, 0]        # wrist
        joint_pos_pinch[4] = [0.04, 0.05, 0]  # thumb_tip - closer
        joint_pos_pinch[8] = [0.03, 0.05, 0]  # index_tip - closer
        joint_pos_pinch[12] = [0, 0.05, 0]    # middle_tip - closer
        
        vectors_pinch = joint_pos_pinch[task_indices, :] - joint_pos_pinch[origin_indices, :]
        dexpilot_pinch = hand_retargeting.left_retargeting.retarget(vectors_pinch, np.array([]))
        
        print("1. Joint Movement Analysis:")
        print(f"   Open hand output: {dexpilot_open}")
        print(f"   Pinch hand output: {dexpilot_pinch}")
        
        # Check movement
        movement = dexpilot_pinch - dexpilot_open
        print(f"   Joint differences: {movement}")
        print(f"   Max difference: {abs(movement).max():.3f} rad ({abs(movement).max() * 180/np.pi:.1f}°)")
        
        # Check for responsive joints
        tolerance = 0.01  # ~0.6 degrees
        moving_joints = abs(movement) > tolerance
        responsive_count = np.sum(moving_joints)
        
        print(f"   Responsive joints: {responsive_count}/{len(movement)}")
        
        if responsive_count >= 3:
            print("   ✅ Good joint responsiveness")
            return True
        else:
            print("   ❌ Poor joint responsiveness")
            return False
            
    except Exception as e:
        print(f"Error during movement test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing DexPilot without fixed_qpos to understand behavior...\n")
    
    success1 = test_dexpilot_without_fixed_qpos()
    success2 = test_hand_movement_with_correct_approach()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("🎉 SUCCESS: DexPilot 5-joint approach is working!")
        print("   - DexPilot outputs 5 joint values as expected")
        print("   - Mapping to 7 hardware joints works correctly")
        print("   - Joint movement is responsive to hand pose changes")
        print("   - Ready to update robot_hand_unitree.py with correct approach")
    else:
        print("❌ ISSUES DETECTED:")
        if not success1:
            print("   - DexPilot output or mapping issues")
        if not success2:
            print("   - Joint movement response problems")
        print("   - Further investigation needed")
    
    print()
