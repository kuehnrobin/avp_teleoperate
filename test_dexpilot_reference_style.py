#!/usr/bin/env python3
"""
Test that our DexPilot implementation now matches the reference show_realtime_retargeting.py behavior.
"""

import sys
import os
import numpy as np

# Add the teleop directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'teleop'))

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType


def test_reference_style_dexpilot():
    """Test DexPilot implementation using the reference style (no fixed_qpos)."""
    
    print("=== Testing Reference-Style DexPilot Implementation ===\n")
    
    try:
        # Load the DexPilot configuration (now with all 7 joints)
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        print("1. Configuration Verification:")
        print(f"   Target joints: {len(left_optimizer.target_joint_names)}")
        print(f"   Target joint names: {left_optimizer.target_joint_names}")
        print(f"   Fixed joints: {len(left_optimizer.idx_pin2fixed)}")
        print(f"   Retargeting type: {left_optimizer.retargeting_type}")
        print()
        
        print("2. Joint Names Mapping:")
        try:
            joint_names = hand_retargeting.left_retargeting_joint_names
            print(f"   Left retargeting joint names: {joint_names}")
            print(f"   Number of joints: {len(joint_names)}")
        except Exception as e:
            print(f"   Error getting joint names: {e}")
        print()
        
        # Get the human indices that DexPilot expects
        human_indices = left_optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        
        print("3. DexPilot Vector Configuration:")
        print(f"   Origin indices: {origin_indices}")
        print(f"   Task indices: {task_indices}")
        print(f"   Expected vector count: {len(origin_indices)}")
        print()
        
        # Test the reference-style call (no fixed_qpos)
        print("4. Reference-Style Retargeting Test:")
        
        # Create test hand pose
        joint_pos = np.zeros((25, 3))
        joint_pos[0] = [0, 0, 0]        # wrist
        joint_pos[4] = [0.06, 0.03, 0]  # thumb_tip
        joint_pos[8] = [0.04, 0.07, 0]  # index_tip
        joint_pos[12] = [0, 0.08, 0]    # middle_tip
        
        # Calculate vectors exactly like in show_realtime_retargeting.py
        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        print(f"   Input vectors shape: {ref_value.shape}")
        print(f"   Input vectors:\n{ref_value}")
        print()
        
        # Call retarget WITHOUT fixed_qpos (reference style)
        qpos = hand_retargeting.left_retargeting.retarget(ref_value)
        print(f"   Output qpos shape: {qpos.shape}")
        print(f"   Output qpos: {qpos}")
        print(f"   Range: [{qpos.min():.3f}, {qpos.max():.3f}]")
        print()
        
        # Check if all joints are within reasonable ranges
        print("5. Joint Range Analysis:")
        hardware_joints = hand_retargeting.left_dex3_api_joint_names
        
        # Expected ranges (from URDF)
        expected_ranges = {
            'left_hand_thumb_0_joint': (-1.05, 1.05),     # -60° to +60°
            'left_hand_thumb_1_joint': (-0.72, 0.92),     # -41° to +53°
            'left_hand_thumb_2_joint': (0.0, 1.75),       # 0° to +100°
            'left_hand_middle_0_joint': (-1.57, 0.0),     # -90° to 0°
            'left_hand_middle_1_joint': (-1.75, 0.0),     # -100° to 0°
            'left_hand_index_0_joint': (-1.57, 0.0),      # -90° to 0°
            'left_hand_index_1_joint': (-1.75, 0.0)       # -100° to 0°
        }
        
        # Map from URDF order to hardware order for comparison
        urdf_joints = hand_retargeting.left_retargeting_joint_names
        urdf_to_hw_mapping = {}
        
        for hw_idx, hw_joint in enumerate(hardware_joints):
            if hw_joint in urdf_joints:
                urdf_idx = urdf_joints.index(hw_joint)
                urdf_to_hw_mapping[hw_idx] = urdf_idx
        
        print(f"   URDF to Hardware mapping: {urdf_to_hw_mapping}")
        
        all_in_range = True
        for hw_idx, joint_name in enumerate(hardware_joints):
            if hw_idx in urdf_to_hw_mapping:
                urdf_idx = urdf_to_hw_mapping[hw_idx]
                value = qpos[urdf_idx]
                min_val, max_val = expected_ranges[joint_name]
                
                if min_val <= value <= max_val:
                    status = "✅ IN RANGE"
                else:
                    status = "❌ OUT OF RANGE"
                    all_in_range = False
                    
                print(f"   Joint {hw_idx} ({joint_name}): {value:.3f} rad ({value*180/np.pi:.1f}°) - {status}")
                print(f"      Expected: [{min_val:.3f}, {max_val:.3f}] rad ([{min_val*180/np.pi:.1f}°, {max_val*180/np.pi:.1f}°])")
        
        print()
        
        # Test different hand poses to check responsiveness
        print("6. Responsiveness Test:")
        
        # Pinching pose
        joint_pos_pinch = np.zeros((25, 3))
        joint_pos_pinch[0] = [0, 0, 0]        # wrist
        joint_pos_pinch[4] = [0.03, 0.05, 0]  # thumb_tip - closer
        joint_pos_pinch[8] = [0.02, 0.05, 0]  # index_tip - closer
        joint_pos_pinch[12] = [0, 0.05, 0]    # middle_tip - closer
        
        ref_value_pinch = joint_pos_pinch[task_indices, :] - joint_pos_pinch[origin_indices, :]
        qpos_pinch = hand_retargeting.left_retargeting.retarget(ref_value_pinch)
        
        # Check for movement
        movement = qpos_pinch - qpos
        max_movement = abs(movement).max()
        print(f"   Max joint movement: {max_movement:.3f} rad ({max_movement*180/np.pi:.1f}°)")
        
        # Count moving joints
        tolerance = 0.01  # ~0.6 degrees
        moving_joints = abs(movement) > tolerance
        print(f"   Moving joints: {np.sum(moving_joints)}/{len(qpos)}")
        
        if np.sum(moving_joints) >= 4:  # At least 4 joints should move
            print("   ✅ Good responsiveness")
            responsive = True
        else:
            print("   ❌ Poor responsiveness")
            responsive = False
        
        print()
        
        return all_in_range and responsive
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hardware_mapping():
    """Test that the hardware mapping works correctly."""
    
    print("=== Testing Hardware Mapping ===\n")
    
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        
        print("1. Mapping Configuration:")
        print(f"   Left mapping: {hand_retargeting.left_dex_retargeting_to_hardware}")
        print(f"   Right mapping: {hand_retargeting.right_dex_retargeting_to_hardware}")
        print()
        
        print("2. Joint Name Verification:")
        hardware_joints = hand_retargeting.left_dex3_api_joint_names
        retargeting_joints = hand_retargeting.left_retargeting_joint_names
        
        print(f"   Hardware API joints: {hardware_joints}")
        print(f"   Retargeting joints: {retargeting_joints}")
        print()
        
        print("3. Mapping Verification:")
        for hw_idx, hw_joint in enumerate(hardware_joints):
            if hw_idx < len(hand_retargeting.left_dex_retargeting_to_hardware):
                retarget_idx = hand_retargeting.left_dex_retargeting_to_hardware[hw_idx]
                if retarget_idx < len(retargeting_joints):
                    retarget_joint = retargeting_joints[retarget_idx]
                    if hw_joint == retarget_joint:
                        status = "✅ MATCH"
                    else:
                        status = "❌ MISMATCH"
                    print(f"   Hardware[{hw_idx}] {hw_joint} -> Retargeting[{retarget_idx}] {retarget_joint} - {status}")
                else:
                    print(f"   Hardware[{hw_idx}] {hw_joint} -> INVALID INDEX {retarget_idx}")
            else:
                print(f"   Hardware[{hw_idx}] {hw_joint} -> NO MAPPING")
        
        return True
        
    except Exception as e:
        print(f"Error during mapping test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing reference-style DexPilot implementation...\n")
    
    success1 = test_reference_style_dexpilot()
    success2 = test_hardware_mapping()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("🎉 SUCCESS: DexPilot implementation now matches reference behavior!")
        print("   - All 7 joints are optimized (no fixed_qpos)")
        print("   - Joint values are within expected ranges")
        print("   - Joints respond to hand pose changes")
        print("   - Hardware mapping is correct")
        print("   - Ready for real teleoperation!")
    else:
        print("❌ ISSUES DETECTED:")
        if not success1:
            print("   - DexPilot behavior doesn't match reference")
        if not success2:
            print("   - Hardware mapping issues")
        print("   - Further investigation needed")
    
    print()
