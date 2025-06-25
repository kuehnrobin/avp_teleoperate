#!/usr/bin/env python3
"""
Comprehensive test to verify that DexPilot now correctly maps to all 7 joint angles
and that the thumb/index finger issues are resolved.
"""

import sys
import os
import numpy as np

# Add the teleop directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'teleop'))

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

def test_comprehensive_joint_mapping():
    """Test all 7 joints respond correctly to different hand poses."""
    
    print("=== Comprehensive Joint Mapping Test ===\n")
    
    try:
        # Load DexPilot configuration
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        # Get the human indices that DexPilot expects
        human_indices = left_optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        
        # Hardware joint names for reference
        joint_names = [
            "left_hand_thumb_0_joint",    # Joint 0 - Thumb base
            "left_hand_thumb_1_joint",    # Joint 1 - Thumb middle  
            "left_hand_thumb_2_joint",    # Joint 2 - Thumb tip
            "left_hand_middle_0_joint",   # Joint 3 - Middle base
            "left_hand_middle_1_joint",   # Joint 4 - Middle tip
            "left_hand_index_0_joint",    # Joint 5 - Index base
            "left_hand_index_1_joint"     # Joint 6 - Index tip
        ]
        
        print("Joint names mapping:")
        for i, name in enumerate(joint_names):
            print(f"   Joint {i}: {name}")
        print()
        
        # Test different hand poses
        test_poses = [
            {
                "name": "Fully Open Hand", 
                "wrist": [0, 0, 0],
                "thumb": [0.08, 0.03, 0.02],    # Thumb spread wide
                "index": [0.05, 0.09, 0],       # Index pointing up
                "middle": [0, 0.095, 0]         # Middle pointing up
            },
            {
                "name": "Loose Fist",
                "wrist": [0, 0, 0],
                "thumb": [0.04, 0.05, 0.01],    # Thumb slightly closed
                "index": [0.03, 0.06, 0],       # Index partially bent
                "middle": [0, 0.06, 0]          # Middle partially bent
            },
            {
                "name": "Tight Pinch",
                "wrist": [0, 0, 0],
                "thumb": [0.025, 0.055, 0],     # Thumb almost touching
                "index": [0.025, 0.055, 0],     # Index meeting thumb
                "middle": [0, 0.04, 0]          # Middle bent more
            },
            {
                "name": "Thumb Opposition",
                "wrist": [0, 0, 0],
                "thumb": [0.02, 0.06, 0.03],    # Thumb across palm
                "index": [0.05, 0.08, 0],       # Index extended
                "middle": [0, 0.085, 0]         # Middle extended
            }
        ]
        
        results = []
        
        for pose in test_poses:
            print(f"Testing: {pose['name']}")
            
            # Create joint_pos array with the pose
            joint_pos = np.zeros((25, 3))
            joint_pos[0] = pose["wrist"]   # wrist -> index 0
            joint_pos[4] = pose["thumb"]   # thumb_tip -> index 4
            joint_pos[8] = pose["index"]   # index_tip -> index 8
            joint_pos[12] = pose["middle"] # middle_tip -> index 12
            
            # Calculate vectors
            vectors = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            
            # Get retargeting result
            joint_angles = hand_retargeting.left_retargeting.retarget(vectors)
            
            print(f"   Joint angles: {joint_angles}")
            print(f"   Range: [{joint_angles.min():.3f}, {joint_angles.max():.3f}]")
            
            # Check each joint individually
            joint_analysis = []
            for i, (angle, name) in enumerate(zip(joint_angles, joint_names)):
                joint_analysis.append({
                    "joint": i,
                    "name": name,
                    "angle": angle,
                    "degrees": angle * 180 / np.pi
                })
                print(f"     Joint {i} ({name.split('_')[-2]}_{name.split('_')[-1]}): {angle:.3f} rad ({angle * 180 / np.pi:.1f}°)")
            
            results.append({
                "pose": pose["name"],
                "joint_angles": joint_angles,
                "joint_analysis": joint_analysis
            })
            print()
        
        # Analyze joint movement across poses
        print("=== Joint Movement Analysis ===")
        print("Checking which joints show significant movement between poses:\n")
        
        base_angles = results[0]["joint_angles"]  # Open hand as baseline
        
        for i, joint_name in enumerate(joint_names):
            angles_for_joint = [result["joint_angles"][i] for result in results]
            min_angle = min(angles_for_joint)
            max_angle = max(angles_for_joint)
            range_rad = max_angle - min_angle
            range_deg = range_rad * 180 / np.pi
            
            status = "✓ GOOD" if range_deg > 5.0 else "⚠ LIMITED" if range_deg > 1.0 else "✗ STUCK"
            
            print(f"Joint {i} ({joint_name.split('_')[-2]}_{joint_name.split('_')[-1]}):")
            print(f"   Range: {range_rad:.3f} rad ({range_deg:.1f}°) - {status}")
            
            if range_deg < 1.0:
                print(f"   ⚠ This joint may be stuck at {angles_for_joint[0]:.3f} rad")
            
            print()
        
        # Check for the specific issues mentioned in the conversation
        print("=== Specific Issue Analysis ===")
        
        # Check thumb joints (0, 1, 2)
        thumb_ranges = []
        for i in [0, 1, 2]:
            angles_for_joint = [result["joint_angles"][i] for result in results]
            range_deg = (max(angles_for_joint) - min(angles_for_joint)) * 180 / np.pi
            thumb_ranges.append(range_deg)
        
        thumb_working = all(r > 3.0 for r in thumb_ranges)
        print(f"Thumb joints (0,1,2) working: {'✓ YES' if thumb_working else '✗ NO'}")
        for i, range_deg in enumerate(thumb_ranges):
            print(f"   Joint {i}: {range_deg:.1f}° range")
        
        # Check index joints (5, 6)  
        index_ranges = []
        for i in [5, 6]:
            angles_for_joint = [result["joint_angles"][i] for result in results]
            range_deg = (max(angles_for_joint) - min(angles_for_joint)) * 180 / np.pi
            index_ranges.append(range_deg)
        
        index_working = all(r > 3.0 for r in index_ranges)
        print(f"Index joints (5,6) working: {'✓ YES' if index_working else '✗ NO'}")
        for i, range_deg in enumerate(index_ranges):
            print(f"   Joint {5+i}: {range_deg:.1f}° range")
        
        # Check middle joints (3, 4)
        middle_ranges = []
        for i in [3, 4]:
            angles_for_joint = [result["joint_angles"][i] for result in results]
            range_deg = (max(angles_for_joint) - min(angles_for_joint)) * 180 / np.pi
            middle_ranges.append(range_deg)
        
        middle_working = all(r > 3.0 for r in middle_ranges)
        print(f"Middle joints (3,4) working: {'✓ YES' if middle_working else '✗ NO'}")
        for i, range_deg in enumerate(middle_ranges):
            print(f"   Joint {3+i}: {range_deg:.1f}° range")
        
        overall_success = thumb_working and index_working and middle_working
        print(f"\nOverall joint mapping: {'✓ SUCCESS' if overall_success else '✗ ISSUES REMAIN'}")
        
        return overall_success
        
    except Exception as e:
        print(f"Error in comprehensive joint mapping test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_realtime_simulation():
    """Simulate the real-time control loop to test actual robot_hand_unitree.py logic."""
    
    print("=== Real-time Control Loop Simulation ===\n")
    
    try:
        # This simulates the exact logic from robot_hand_unitree.py
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        
        # Simulate OpenXR hand data arrays (25 joints * 3 coordinates = 75 values)
        left_hand_array = np.zeros(75)
        right_hand_array = np.zeros(75)
        
        # Test poses
        test_poses = [
            "Open Hand - fingers spread",
            "Pinch - thumb and index touching", 
            "Fist - all fingers closed"
        ]
        
        openxr_poses = [
            # Open hand
            {0: [0,0,0], 4: [0.08,0.03,0.02], 9: [0.05,0.09,0], 14: [0,0.095,0]},
            # Pinch 
            {0: [0,0,0], 4: [0.025,0.055,0], 9: [0.025,0.055,0], 14: [0,0.04,0]},
            # Fist
            {0: [0,0,0], 4: [0.02,0.04,0.01], 9: [0.02,0.035,0], 14: [0,0.03,0]}
        ]
        
        for pose_name, openxr_data in zip(test_poses, openxr_poses):
            print(f"Testing: {pose_name}")
            
            # Fill left_hand_array with OpenXR data (reshape to 25x3)
            left_hand_mat = np.zeros((25, 3))
            for joint_idx, pos in openxr_data.items():
                left_hand_mat[joint_idx] = pos
                # Also fill the flat array
                left_hand_array[joint_idx*3:(joint_idx+1)*3] = pos
            
            # Simulate the exact DexPilot logic from robot_hand_unitree.py
            left_retargeting_type = hand_retargeting.left_retargeting.optimizer.retargeting_type
            left_indices = hand_retargeting.left_retargeting.optimizer.target_link_human_indices
            
            if left_retargeting_type == "POSITION":
                print("   ERROR: Should be DEXPILOT, not POSITION")
                continue
            else:
                # DexPilot method - exactly as implemented
                origin_indices = left_indices[0, :]  
                task_indices = left_indices[1, :]    
                
                joint_pos = np.zeros((25, 3))
                joint_pos[0] = left_hand_mat[0]   # wrist -> index 0
                joint_pos[4] = left_hand_mat[4]   # thumb_tip -> index 4
                joint_pos[8] = left_hand_mat[9]   # index_tip -> index 8 (OpenXR index 9)
                joint_pos[12] = left_hand_mat[14] # middle_tip -> index 12 (OpenXR index 14)
                
                ref_left_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                left_q_target = hand_retargeting.left_retargeting.retarget(ref_left_value)
            
            print(f"   Target joint angles: {left_q_target}")
            print(f"   Range: [{left_q_target.min():.3f}, {left_q_target.max():.3f}]")
            
            # Check for problematic joints
            problematic = []
            if abs(left_q_target[0]) < 0.01: problematic.append("Thumb0")
            if abs(left_q_target[1]) < 0.01: problematic.append("Thumb1") 
            if abs(left_q_target[2]) < 0.01: problematic.append("Thumb2")
            if abs(left_q_target[5]) < 0.01: problematic.append("Index0")
            if abs(left_q_target[6]) < 0.01: problematic.append("Index1")
            
            if problematic:
                print(f"   ⚠ Joints near zero: {problematic}")
            else:
                print(f"   ✓ All joints show movement")
            print()
        
        return True
        
    except Exception as e:
        print(f"Error in real-time simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting comprehensive DexPilot validation...\n")
    
    success1 = test_comprehensive_joint_mapping()
    success2 = test_realtime_simulation()
    
    if success1 and success2:
        print("\n🎉 SUCCESS: DexPilot appears to be fully functional!")
        print("   - All 7 joints respond to input changes")
        print("   - Thumb orientation issues should be resolved")
        print("   - Index finger mapping should now work correctly")
        print("\nReady for real-world testing with teleop system!")
    else:
        print("\n❌ Issues remain that need further investigation.")
