#!/usr/bin/env python3
"""
DexPilot fine-tuning test to solve pinching issues:
1. Thumb too low for index pinching
2. Thumb range limitation (60° instead of 90°)  
3. Middle finger insufficient closure
"""

import sys
import os
import numpy as np

# Add the teleop directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'teleop'))

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType


def analyze_current_behavior():
    """Analyze current DexPilot behavior for pinching scenarios."""
    
    print("=== Analyzing Current DexPilot Pinching Behavior ===\n")
    
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        # Get the human indices that DexPilot expects
        human_indices = left_optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        
        print("1. Current Configuration:")
        print(f"   Scaling factor: {left_optimizer.scaling}")
        print(f"   Vector indices - Origin: {origin_indices}, Task: {task_indices}")
        
        # Test different pinching scenarios
        scenarios = {
            "Open Hand": {
                0: [0, 0, 0],         # wrist
                4: [0.08, 0.02, 0],   # thumb_tip - spread
                8: [0.05, 0.08, 0],   # index_tip - extended
                12: [0, 0.09, 0]      # middle_tip - extended
            },
            "Thumb-Index Pinch": {
                0: [0, 0, 0],         # wrist
                4: [0.04, 0.06, 0],   # thumb_tip - up and forward for pinch
                8: [0.04, 0.06, 0],   # index_tip - down to meet thumb
                12: [0, 0.08, 0]      # middle_tip - extended
            },
            "Thumb-Middle Pinch": {
                0: [0, 0, 0],         # wrist
                4: [0.03, 0.06, 0],   # thumb_tip - up and forward
                8: [0.05, 0.08, 0],   # index_tip - extended
                12: [0.03, 0.06, 0]   # middle_tip - forward to meet thumb
            },
            "Thumb 90 Degree": {
                0: [0, 0, 0],         # wrist
                4: [0.02, 0.08, 0],   # thumb_tip - straight up (90°)
                8: [0.05, 0.08, 0],   # index_tip - extended
                12: [0, 0.09, 0]      # middle_tip - extended
            }
        }
        
        results = {}
        hardware_joints = hand_retargeting.left_dex3_api_joint_names
        
        for scenario_name, joint_positions in scenarios.items():
            print(f"\n2. Testing: {scenario_name}")
            
            # Create joint position array
            joint_pos = np.zeros((25, 3))
            for joint_idx, pos in joint_positions.items():
                joint_pos[joint_idx] = pos
            
            # Calculate vectors
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            
            # Retarget
            qpos = hand_retargeting.left_retargeting.retarget(ref_value)
            
            # Map to hardware order
            hardware_commands = np.zeros(7)
            for hw_idx, retarget_idx in enumerate(hand_retargeting.left_dex_retargeting_to_hardware):
                hardware_commands[hw_idx] = qpos[retarget_idx]
            
            print(f"   Hardware joint commands:")
            thumb_angles = {}
            for i, joint_name in enumerate(hardware_joints):
                angle_deg = hardware_commands[i] * 180 / np.pi
                print(f"     [{i}] {joint_name}: {hardware_commands[i]:.3f} rad ({angle_deg:.1f}°)")
                
                if 'thumb' in joint_name:
                    thumb_angles[joint_name] = angle_deg
            
            results[scenario_name] = {
                'hardware_commands': hardware_commands,
                'thumb_angles': thumb_angles
            }
        
        print("\n3. Analysis of Issues:")
        
        # Issue 1: Thumb-Index pinch height
        open_thumb1 = results["Open Hand"]['thumb_angles']['left_hand_thumb_1_joint']
        pinch_thumb1 = results["Thumb-Index Pinch"]['thumb_angles']['left_hand_thumb_1_joint']
        print(f"   Issue 1 - Thumb height for index pinch:")
        print(f"     Open hand thumb_1: {open_thumb1:.1f}°")
        print(f"     Pinch thumb_1: {pinch_thumb1:.1f}°")
        print(f"     Change: {pinch_thumb1 - open_thumb1:.1f}°")
        if pinch_thumb1 - open_thumb1 < 30:  # Need at least 30° upward movement
            print(f"     ❌ ISSUE: Insufficient upward thumb movement for pinching")
        else:
            print(f"     ✅ OK: Sufficient upward thumb movement")
        
        # Issue 2: Thumb 90° range
        thumb90_thumb1 = results["Thumb 90 Degree"]['thumb_angles']['left_hand_thumb_1_joint']
        print(f"\n   Issue 2 - Thumb 90° range:")
        print(f"     Target: 90° thumb position")
        print(f"     Actual thumb_1: {thumb90_thumb1:.1f}°")
        if abs(thumb90_thumb1) < 45:  # Should reach at least 45° for 90° gesture
            print(f"     ❌ ISSUE: Thumb doesn't reach sufficient angle for 90° gesture")
        else:
            print(f"     ✅ OK: Thumb reaches good angle for 90° gesture")
        
        # Issue 3: Middle finger closure
        open_middle1 = results["Open Hand"]['hardware_commands'][4]  # middle_1
        pinch_middle1 = results["Thumb-Middle Pinch"]['hardware_commands'][4]  # middle_1
        middle_closure = abs(pinch_middle1 - open_middle1) * 180 / np.pi
        print(f"\n   Issue 3 - Middle finger closure:")
        print(f"     Open middle_1: {open_middle1 * 180/np.pi:.1f}°")
        print(f"     Pinch middle_1: {pinch_middle1 * 180/np.pi:.1f}°")
        print(f"     Closure amount: {middle_closure:.1f}°")
        if middle_closure < 40:  # Need at least 40° closure for pinching
            print(f"     ❌ ISSUE: Insufficient middle finger closure")
        else:
            print(f"     ✅ OK: Sufficient middle finger closure")
        
        return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_scaling_solutions():
    """Test different scaling factors to improve pinching."""
    
    print("\n=== Testing Scaling Factor Solutions ===\n")
    
    # We'll modify the DexPilot config temporarily to test different scaling factors
    original_config = "/home/robin/humanoid/humanoid_ws/src/avp_teleoperate/assets/unitree_hand/unitree_dex3_left_dexpilot.yml"
    
    scaling_factors = [0.8, 1.0, 1.2, 1.5]
    
    for scaling in scaling_factors:
        print(f"Testing scaling factor: {scaling}")
        
        # Create modified config content
        config_content = f"""retargeting:
  type: DexPilot
  urdf_path: unitree_hand/unitree_dex3_left.urdf
  
  target_joint_names: [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint", 
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint"
  ]
  wrist_link_name: "base_link"
  finger_tip_link_names: ["thumb_tip", "index_tip", "middle_tip"]
  scaling_factor: {scaling}
  
  project_dist: 0.03
  escape_dist: 0.05
  low_pass_alpha: 0.2
"""
        
        # Write temporary config
        temp_config = "/tmp/test_dexpilot_config.yml"
        with open(temp_config, 'w') as f:
            f.write(config_content)
        
        # TODO: Load and test with temporary config
        # This would require modifying the HandRetargeting class to accept config path
        # For now, we'll print what we would test
        print(f"   Would test scaling {scaling} with thumb 90° gesture")
        print(f"   Expected behavior: {'Better' if scaling > 1.0 else 'Worse'} thumb range")
        print()


def propose_solutions():
    """Propose specific solutions for the pinching issues."""
    
    print("=== Proposed Solutions ===\n")
    
    print("1. SOLUTION for Thumb too low (Issue 1):")
    print("   Problem: thumb_1_joint range is -41° to +53°, limited positive range")
    print("   Solutions:")
    print("   a) Increase scaling_factor to 1.3-1.5 to amplify thumb movements")
    print("   b) Modify URDF to expand thumb_1_joint upper limit from +53° to +70°")
    print("   c) Add offset to thumb_1 calculations in retargeting")
    print()
    
    print("2. SOLUTION for Thumb 90° range (Issue 2):")
    print("   Problem: Scaling factor 1.0 doesn't utilize full thumb joint ranges")
    print("   Solutions:")
    print("   a) Increase overall scaling_factor to 1.4")
    print("   b) Use per-joint scaling with higher factors for thumb joints")
    print("   c) Adjust DexPilot vector calculation for thumb-specific movements")
    print()
    
    print("3. SOLUTION for Middle finger closure (Issue 3):")
    print("   Problem: middle_1_joint may not be closing enough for pinching")
    print("   Solutions:")
    print("   a) Increase scaling_factor affects all fingers including middle")
    print("   b) Modify DexPilot to give higher weight to finger-tip vectors")
    print("   c) Adjust human hand pose interpretation for middle finger")
    print()
    
    print("4. RECOMMENDED APPROACH:")
    print("   Start with scaling_factor: 1.4 (affects all joints proportionally)")
    print("   If needed, modify URDF limits for thumb_1_joint: upper='1.22' (+70°)")
    print("   Test and iterate based on real teleoperation feedback")


if __name__ == "__main__":
    print("DexPilot Fine-tuning Analysis for Pinching Issues...\n")
    
    results = analyze_current_behavior()
    if results:
        test_scaling_solutions()
        propose_solutions()
        
        print("\n" + "="*60)
        print("🔧 NEXT STEPS:")
        print("1. Increase scaling_factor to 1.4 in DexPilot configs")
        print("2. Test with real VR teleoperation")
        print("3. If thumb still too low, modify URDF thumb_1 upper limit")
        print("4. Fine-tune based on operator feedback")
    
    print()
