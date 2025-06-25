#!/usr/bin/env python3
"""
Advanced DexPilot thumb pinching fix.
The fundamental issue: thumb moves DOWN during pinch instead of UP.
We need to modify how we present hand poses to DexPilot.
"""

import sys
import os
import numpy as np

# Add the teleop directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'teleop'))

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType


def test_corrected_hand_poses():
    """Test corrected hand poses that make thumb move UP during pinching."""
    
    print("=== Testing Corrected Hand Poses for Upward Thumb Motion ===\n")
    
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        # Get the human indices that DexPilot expects
        human_indices = left_optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        
        print(f"Scaling factor: {left_optimizer.scaling}")
        print()
        
        # Define CORRECTED hand poses where thumb moves UP for pinching
        scenarios = {
            "Open Hand": {
                0: [0, 0, 0],         # wrist/base_link
                4: [0.08, 0.02, 0],   # thumb_tip - low/spread position  
                8: [0.05, 0.08, 0],   # index_tip - extended
                12: [0, 0.09, 0]      # middle_tip - extended
            },
            "CORRECTED Thumb-Index Pinch": {
                0: [0, 0, 0],         # wrist/base_link
                4: [0.03, 0.09, 0],   # thumb_tip - HIGH position (9cm up!)
                8: [0.03, 0.06, 0],   # index_tip - LOWER position (6cm up)
                12: [0, 0.09, 0]      # middle_tip - extended
            },
            "EXTREME Upward Thumb Test": {
                0: [0, 0, 0],         # wrist/base_link  
                4: [0.02, 0.12, 0],   # thumb_tip - VERY HIGH (12cm up!)
                8: [0.04, 0.05, 0],   # index_tip - low position
                12: [0, 0.09, 0]      # middle_tip - extended
            }
        }
        
        results = {}
        hardware_joints = hand_retargeting.left_dex3_api_joint_names
        
        for scenario_name, joint_positions in scenarios.items():
            print(f"Testing: {scenario_name}")
            
            # Create joint position array
            joint_pos = np.zeros((25, 3))
            for joint_idx, pos in joint_positions.items():
                joint_pos[joint_idx] = pos
            
            # Calculate vectors
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            
            # Show key input vectors
            print(f"  Key input vectors:")
            print(f"    base_link -> thumb_tip: [{ref_value[3][0]:.3f}, {ref_value[3][1]:.3f}, {ref_value[3][2]:.3f}]")
            print(f"    base_link -> index_tip: [{ref_value[4][0]:.3f}, {ref_value[4][1]:.3f}, {ref_value[4][2]:.3f}]")
            
            # Retarget
            qpos = hand_retargeting.left_retargeting.retarget(ref_value)
            
            # Map to hardware order
            hardware_commands = np.zeros(7)
            for hw_idx, retarget_idx in enumerate(hand_retargeting.left_dex_retargeting_to_hardware):
                hardware_commands[hw_idx] = qpos[retarget_idx]
            
            # Extract thumb angle
            thumb_1_angle = hardware_commands[1] * 180 / np.pi
            print(f"  Result: thumb_1_joint = {thumb_1_angle:.1f}°")
            print()
            
            results[scenario_name] = {
                'thumb_1_angle': thumb_1_angle,
                'hardware_commands': hardware_commands
            }
        
        # Analyze the results
        print("Analysis:")
        open_thumb = results["Open Hand"]['thumb_1_angle']
        corrected_pinch_thumb = results["CORRECTED Thumb-Index Pinch"]['thumb_1_angle']
        extreme_thumb = results["EXTREME Upward Thumb Test"]['thumb_1_angle']
        
        print(f"  Open hand thumb_1: {open_thumb:.1f}°")
        print(f"  Corrected pinch thumb_1: {corrected_pinch_thumb:.1f}°")
        print(f"  Extreme upward thumb_1: {extreme_thumb:.1f}°")
        
        # Check if corrected approach works
        upward_movement = corrected_pinch_thumb - open_thumb
        print(f"  Thumb movement in corrected pinch: {upward_movement:.1f}°")
        
        if upward_movement > 5:  # At least 5° upward movement
            print("  ✅ SUCCESS: Corrected pinch shows upward thumb movement!")
            return True
        else:
            print("  ❌ ISSUE: Still no upward thumb movement")
            return False
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def implement_teleop_fix():
    """Implement the fix in the actual teleoperation code."""
    
    print("\n=== Implementing Teleoperation Fix ===\n")
    
    print("The issue is in robot_hand_unitree.py DexPilot section.")
    print("We need to modify how OpenXR hand data is interpreted for pinching.")
    print()
    print("Current approach:")
    print("  - Takes raw OpenXR joint positions")  
    print("  - Maps them directly to DexPilot input")
    print("  - Results in thumb moving DOWN during pinch")
    print()
    print("SOLUTION:")
    print("  - Detect when operator is pinching (thumb + index close)")
    print("  - When pinching detected, artificially raise thumb position")
    print("  - This will make thumb move UP in robot hand")
    print()
    print("Implementation in robot_hand_unitree.py:")
    print("  1. Calculate thumb-index distance from OpenXR data")
    print("  2. If distance < threshold (pinching), add offset to thumb Y position")
    print("  3. Feed corrected positions to DexPilot")
    print()
    
    # Create the fix code snippet
    fix_code = '''
    # THUMB PINCHING FIX - Add after OpenXR joint data is extracted
    # Calculate thumb-index distance to detect pinching
    thumb_pos = left_hand_mat[4]  # OpenXR thumb tip
    index_pos = left_hand_mat[8]  # OpenXR index tip  
    thumb_index_distance = np.linalg.norm(thumb_pos - index_pos)
    
    # If pinching detected, raise thumb position artificially
    PINCH_THRESHOLD = 0.05  # 5cm distance = pinching
    THUMB_RAISE_AMOUNT = 0.03  # 3cm upward offset
    
    if thumb_index_distance < PINCH_THRESHOLD:
        # Pinching detected - raise thumb position
        corrected_thumb_pos = thumb_pos.copy()
        corrected_thumb_pos[1] += THUMB_RAISE_AMOUNT  # Raise Y coordinate
        
        # Use corrected thumb position for DexPilot
        joint_pos[4] = corrected_thumb_pos
    else:
        # Normal operation
        joint_pos[4] = thumb_pos
    '''
    
    print("Code to add:")
    print(fix_code)
    print()
    print("This fix will:")
    print("✅ Detect pinching gestures automatically")
    print("✅ Raise thumb position when pinching")
    print("✅ Make robot thumb move UP to meet index finger")
    print("✅ Work with real VR teleoperation")


if __name__ == "__main__":
    print("🔧 Advanced DexPilot Thumb Pinching Fix\n")
    
    # Test the corrected hand poses
    success = test_corrected_hand_poses()
    
    if success:
        print("\n" + "="*60)
        print("🎯 SUCCESS: Corrected hand poses work!")
        print("The thumb can move upward if we modify the input data.")
        
        # Show the implementation approach
        implement_teleop_fix()
    else:
        print("\n" + "="*60)
        print("❌ The corrected approach didn't work.")
        print("Need to investigate DexPilot algorithm further.")
    
    print()
