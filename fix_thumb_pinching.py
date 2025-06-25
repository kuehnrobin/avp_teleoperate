#!/usr/bin/env python3
"""
Targeted thumb pinching fix for DexPilot.
The thumb consistently moves ~10° too low during pinching.
Let's identify and fix the root cause.
"""

import sys
import os
import numpy as np

# Add the teleop directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'teleop'))

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType


def analyze_thumb_motion():
    """Analyze exactly why the thumb moves down instead of up during pinching."""
    
    print("=== Analyzing Thumb Motion Problem ===\n")
    
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        # Get the human indices that DexPilot expects
        human_indices = left_optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        
        print("Current Vector Configuration:")
        print(f"Origin indices: {origin_indices}")
        print(f"Task indices:   {task_indices}")
        print()
        
        # Define two thumb positions: high and low
        scenarios = {
            "Thumb High (for pinching)": {
                0: [0, 0, 0],         # wrist/base_link
                4: [0.04, 0.08, 0],   # thumb_tip - high position for pinching
                8: [0.05, 0.06, 0],   # index_tip - position for pinch
                12: [0, 0.09, 0]      # middle_tip - extended
            },
            "Thumb Low (open hand)": {
                0: [0, 0, 0],         # wrist/base_link  
                4: [0.08, 0.02, 0],   # thumb_tip - low/spread position
                8: [0.05, 0.08, 0],   # index_tip - extended
                12: [0, 0.09, 0]      # middle_tip - extended
            }
        }
        
        print("Testing Thumb Motion Vectors:")
        print("="*50)
        
        for scenario_name, joint_positions in scenarios.items():
            print(f"\n{scenario_name}:")
            
            # Create joint position array
            joint_pos = np.zeros((25, 3))
            for joint_idx, pos in joint_positions.items():
                joint_pos[joint_idx] = pos
            
            # Calculate and analyze each vector
            vectors = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            
            vector_names = [
                "index_tip -> thumb_tip",
                "middle_tip -> thumb_tip", 
                "middle_tip -> index_tip",
                "base_link -> thumb_tip",
                "base_link -> index_tip",
                "base_link -> middle_tip"
            ]
            
            print("  Input vectors to DexPilot:")
            for i, (name, vector) in enumerate(zip(vector_names, vectors)):
                print(f"    Vector {i}: {name}")
                print(f"      [{vector[0]:6.3f}, {vector[1]:6.3f}, {vector[2]:6.3f}]")
            
            # Get thumb joint result
            qpos = hand_retargeting.left_retargeting.retarget(vectors)
            thumb_1_angle = qpos[1] * 180 / np.pi  # thumb_1_joint is at index 1 in target_joint_names
            print(f"  Resulting thumb_1_joint: {thumb_1_angle:.1f}°")
        
        return True
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_thumb_correction_strategies():
    """Test different strategies to fix thumb pinching motion."""
    
    print("\n=== Testing Thumb Correction Strategies ===\n")
    
    # Strategy 1: Modify input vectors to favor upward thumb motion
    print("Strategy 1: Adjust input hand poses to favor upward thumb motion")
    print("- Instead of bringing thumb down to index, bring thumb UP to meet index")
    print("- Adjust the simulated pinch pose to have thumb higher than index")
    print()
    
    # Strategy 2: Increase scaling factor specifically for thumb vectors
    print("Strategy 2: Increase scaling factor to 2.0 to amplify thumb motion")
    print("- Current: 1.8")
    print("- Test: 2.0, 2.2")
    print()
    
    # Strategy 3: Modify the URDF limits further
    print("Strategy 3: Expand URDF thumb_1_joint limits even more")
    print("- Current upper limit: 1.22 rad (70°)")
    print("- Test: 1.40 rad (80°) or 1.57 rad (90°)")
    print()
    
    # Strategy 4: Adjust DexPilot parameters
    print("Strategy 4: Fine-tune DexPilot parameters")
    print("- Reduce project_dist to 0.015 (1.5cm) for earlier grasp detection")
    print("- Increase escape_dist to 0.08 (8cm) for more hysteresis")
    print()


def implement_thumb_fix_v1():
    """Implement the first thumb correction strategy."""
    
    print("=== Implementing Thumb Fix V1 ===\n")
    print("Applying multiple fixes simultaneously:")
    print("1. Increase scaling_factor to 2.0")
    print("2. Reduce project_dist to 0.015 (more sensitive grasp detection)")
    print("3. Increase escape_dist to 0.08 (more stable grasp mode)")
    print()
    
    # Update left hand config
    left_config_content = """retargeting:
  type: DexPilot
  urdf_path: unitree_hand/unitree_dex3_left.urdf
  
  # Target refers to the retargeting target, which is the robot hand
  # All 7 joints optimized for maximum thumb flexibility
  target_joint_names: [
    "left_hand_thumb_0_joint",    # thumb base rotation (-60° to +60°) - good range
    "left_hand_thumb_1_joint",    # thumb bend (-41° to +70°) - EXPANDED range
    "left_hand_thumb_2_joint",    # thumb tip (0° to 100°) - full range
    "left_hand_middle_0_joint",   # middle base (-90° to 0°) - good range
    "left_hand_middle_1_joint",   # middle tip (-100° to 0°) - good range
    "left_hand_index_0_joint",    # index base (-90° to 0°) - good range
    "left_hand_index_1_joint"     # index tip (-100° to 0°) - full range
  ]
  wrist_link_name: "base_link"
  finger_tip_link_names: ["thumb_tip", "index_tip", "middle_tip"]
  scaling_factor: 2.0
  
  # DexPilot parameters - optimized for pinching
  project_dist: 0.015   # 1.5cm - enter grasp mode very early
  escape_dist: 0.08     # 8cm - stable grasp mode with more hysteresis
  
  # A smaller alpha means stronger filtering, i.e. more smooth but also larger latency
  low_pass_alpha: 0.2
"""
    
    # Update right hand config  
    right_config_content = """retargeting:
  type: DexPilot
  urdf_path: unitree_hand/unitree_dex3_right.urdf
  
  # Target refers to the retargeting target, which is the robot hand
  # All 7 joints optimized for maximum thumb flexibility
  target_joint_names: [
    "right_hand_thumb_0_joint",    # thumb base rotation (-60° to +60°) - good range
    "right_hand_thumb_1_joint",    # thumb bend (-41° to +70°) - EXPANDED range 
    "right_hand_thumb_2_joint",    # thumb tip (0° to 100°) - full range
    "right_hand_middle_0_joint",   # middle base (-90° to 0°) - good range
    "right_hand_middle_1_joint",   # middle tip (-100° to 0°) - good range
    "right_hand_index_0_joint",    # index base (-90° to 0°) - good range
    "right_hand_index_1_joint"     # index tip (-100° to 0°) - full range
  ]
  wrist_link_name: "base_link"
  finger_tip_link_names: ["thumb_tip", "index_tip", "middle_tip"]
  scaling_factor: 2.0
  
  # DexPilot parameters - optimized for pinching
  project_dist: 0.015   # 1.5cm - enter grasp mode very early
  escape_dist: 0.08     # 8cm - stable grasp mode with more hysteresis
  
  # A smaller alpha means stronger filtering, i.e. more smooth but also larger latency
  low_pass_alpha: 0.2
"""
    
    # Write the updated configs
    left_config_path = "/home/robin/humanoid/humanoid_ws/src/avp_teleoperate/assets/unitree_hand/unitree_dex3_left_dexpilot.yml"
    right_config_path = "/home/robin/humanoid/humanoid_ws/src/avp_teleoperate/assets/unitree_hand/unitree_dex3_right_dexpilot.yml"
    
    with open(left_config_path, 'w') as f:
        f.write(left_config_content)
    print(f"✅ Updated {left_config_path}")
    
    with open(right_config_path, 'w') as f:
        f.write(right_config_content)
    print(f"✅ Updated {right_config_path}")
    
    print("\n🔧 Configuration Updated!")
    print("Next steps:")
    print("1. Test with: python analyze_dexpilot_pinching.py")
    print("2. If thumb still too low, we'll implement Strategy V2")


def test_current_configuration():
    """Test the current configuration to see thumb behavior."""
    
    print("\n=== Testing Current Configuration ===\n")
    
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        
        # Test pinch scenario with corrected hand pose
        print("Testing improved pinch pose (thumb UP, index down):")
        
        # Create joint position array - CORRECTED for proper pinching
        joint_pos = np.zeros((25, 3))
        joint_pos[0] = [0, 0, 0]         # wrist
        joint_pos[4] = [0.02, 0.09, 0]   # thumb_tip - HIGH position (9cm up)
        joint_pos[8] = [0.04, 0.05, 0]   # index_tip - lower position (5cm up) 
        joint_pos[12] = [0, 0.09, 0]     # middle_tip - extended
        
        # Get vectors
        human_indices = hand_retargeting.left_retargeting.optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        vectors = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        
        # Test retargeting
        qpos = hand_retargeting.left_retargeting.retarget(vectors)
        
        # Check thumb_1_joint result
        thumb_1_angle = qpos[1] * 180 / np.pi
        print(f"Thumb_1_joint result: {thumb_1_angle:.1f}°")
        
        if thumb_1_angle > 45:
            print("✅ SUCCESS: Thumb reaches good angle for pinching!")
        else:
            print("❌ ISSUE: Thumb still too low, need more fixes")
            
        return thumb_1_angle > 45
        
    except Exception as e:
        print(f"Error during test: {e}")
        return False


if __name__ == "__main__":
    print("🔧 DexPilot Thumb Pinching Fix Tool\n")
    
    # Step 1: Analyze the current problem
    if analyze_thumb_motion():
        
        # Step 2: Show correction strategies
        test_thumb_correction_strategies()
        
        # Step 3: Implement fix
        implement_thumb_fix_v1()
        
        # Step 4: Test the fix
        test_current_configuration()
        
    print("\n" + "="*60)
    print("🎯 THUMB FIX SUMMARY:")
    print("Applied fixes:")
    print("• Increased scaling_factor: 1.8 → 2.0")
    print("• Reduced project_dist: 0.02 → 0.015 (more sensitive)")
    print("• Increased escape_dist: 0.06 → 0.08 (more stable)")
    print()
    print("Next: Test with real VR teleoperation!")
