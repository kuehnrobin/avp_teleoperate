#!/usr/bin/env python3
"""
Debug script to understand DexPilot's internal mapping and expected input format.
"""

import sys
import os
import numpy as np

# Add the teleop directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'teleop'))
sys.path.append('/home/robin/humanoid/humanoid_ws/src/avp_teleoperate/dex-retargeting/src')

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from dex_retargeting.retargeting_config import RetargetingConfig

def analyze_dexpilot_mapping():
    """Analyze DexPilot's internal mapping structure."""
    
    print("=== DexPilot Mapping Analysis ===\n")
    
    # Load DexPilot configuration
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        
        # Get the optimizer from left hand retargeting
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        print("1. Basic Configuration:")
        print(f"   Number of fingers: {left_optimizer.num_fingers}")
        print(f"   Finger tip link names: {left_optimizer.task_link_names}")
        print(f"   Origin link names: {left_optimizer.origin_link_names}")
        print()
        
        print("2. Generated Link Indices:")
        origin_indices, task_indices = left_optimizer.generate_link_indices(left_optimizer.num_fingers)
        print(f"   Origin link indices: {origin_indices}")
        print(f"   Task link indices: {task_indices}")
        print()
        
        print("3. Target Link Human Indices:")
        human_indices = left_optimizer.target_link_human_indices
        print(f"   Shape: {human_indices.shape}")
        print(f"   Content:\n{human_indices}")
        print()
        
        print("4. Expected Input Structure:")
        print("   DexPilot expects vectors between pairs of links:")
        for i, (origin_idx, task_idx) in enumerate(zip(origin_indices, task_indices)):
            origin_name = left_optimizer.origin_link_names[i] if i < len(left_optimizer.origin_link_names) else f"link_{origin_idx}"
            task_name = left_optimizer.task_link_names[i] if i < len(left_optimizer.task_link_names) else f"link_{task_idx}"
            print(f"   Vector {i}: {origin_name} -> {task_name}")
        print()
        
        print("5. Human Joint Mapping Analysis:")
        print("   DexPilot's default human indices calculation:")
        print(f"   Origin indices * 4: {np.array(origin_indices) * 4}")
        print(f"   Task indices * 4: {np.array(task_indices) * 4}")
        print()
        
        print("6. OpenXR to DexPilot Mapping Issue:")
        print("   OpenXR fingertip indices: [0=wrist, 4=thumb, 9=index, 14=middle]")
        print("   DexPilot expects: [0=wrist, 4=thumb, 8=index, 12=middle] (4-unit spacing)")
        print("   Current mapping problem: we're using [0, 4, 8, 12] but getting [0, 4, 9, 14]")
        print()
        
        # Let's see what the actual computed indices are
        print("7. Computed Link Indices:")
        print(f"   Computed link names: {left_optimizer.computed_link_names}")
        print(f"   Computed link indices: {left_optimizer.computed_link_indices}")
        print()
        
        print("8. Origin and Task Mapping in Optimizer:")
        print(f"   Origin link indices tensor: {left_optimizer.origin_link_indices}")
        print(f"   Task link indices tensor: {left_optimizer.task_link_indices}")
        print()
        
        return left_optimizer
        
    except Exception as e:
        print(f"Error loading DexPilot configuration: {e}")
        return None

def test_input_vectors():
    """Test different input vector configurations."""
    
    print("=== Testing Input Vector Configurations ===\n")
    
    try:
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        left_optimizer = hand_retargeting.left_retargeting.optimizer
        
        # Test 1: Current approach with incorrect mapping
        print("Test 1: Current mapping approach")
        joint_pos_current = np.zeros((25, 3))
        # This is what we're currently doing (WRONG)
        joint_pos_current[0] = [0, 0, 0]      # wrist
        joint_pos_current[4] = [0.1, 0, 0]   # thumb
        joint_pos_current[8] = [0, 0.1, 0]   # index (mapped from OpenXR[9])
        joint_pos_current[12] = [0, 0, 0.1]  # middle (mapped from OpenXR[14])
        
        human_indices = left_optimizer.target_link_human_indices
        origin_indices = human_indices[0, :]
        task_indices = human_indices[1, :]
        
        vectors_current = joint_pos_current[task_indices, :] - joint_pos_current[origin_indices, :]
        print(f"   Input vectors shape: {vectors_current.shape}")
        print(f"   Vectors:\n{vectors_current}")
        print()
        
        # Test 2: Corrected approach
        print("Test 2: Corrected mapping approach")
        # We need to understand what indices DexPilot actually expects
        
        # Let's manually create the correct mapping
        print("   Manual vector creation based on understanding:")
        print("   Expected vectors for 3-finger hand (thumb, index, middle):")
        
        # For a 3-finger hand, DexPilot generates these connections:
        # Between fingers: index->thumb, middle->thumb, middle->index  
        # To base: wrist->thumb, wrist->index, wrist->middle
        
        wrist_pos = np.array([0, 0, 0])
        thumb_pos = np.array([0.1, 0, 0])
        index_pos = np.array([0, 0.1, 0]) 
        middle_pos = np.array([0, 0, 0.1])
        
        # Based on generate_link_indices for 3 fingers:
        # origin_indices = [2, 3, 3, 0, 0, 0] 
        # task_indices = [1, 1, 2, 1, 2, 3]
        # But with link names: [wrist, thumb, index, middle] -> indices [0, 1, 2, 3]
        
        manual_vectors = []
        print("   Expected vector pairs:")
        for i, (orig_idx, task_idx) in enumerate(zip(origin_indices, task_indices)):
            print(f"     Vector {i}: origin={orig_idx}, task={task_idx}")
        
        return left_optimizer
        
    except Exception as e:
        print(f"Error in vector testing: {e}")
        return None

def debug_joint_mapping():
    """Debug the joint mapping issue specifically."""
    
    print("=== Joint Mapping Debug ===\n")
    
    try:
        # Load both vector and dexpilot for comparison
        vector_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'vector')
        dexpilot_retargeting = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        
        print("1. Vector Retargeting Joint Names:")
        print(f"   Left: {vector_retargeting.left_retargeting_joint_names}")
        print(f"   Right: {vector_retargeting.right_retargeting_joint_names}")
        print()
        
        print("2. DexPilot Target Joint Names:")
        left_joint_names = dexpilot_retargeting.left_retargeting.joint_names
        right_joint_names = dexpilot_retargeting.right_retargeting.joint_names
        print(f"   Left: {left_joint_names}")
        print(f"   Right: {right_joint_names}")
        print()
        
        print("3. Hardware Joint Names:")
        print(f"   Left API joints: {dexpilot_retargeting.left_dex3_api_joint_names}")
        print(f"   Right API joints: {dexpilot_retargeting.right_dex3_api_joint_names}")
        print()
        
        print("4. Mapping Arrays:")
        print(f"   Left retargeting->hardware: {dexpilot_retargeting.left_dex_retargeting_to_hardware}")
        print(f"   Right retargeting->hardware: {dexpilot_retargeting.right_dex_retargeting_to_hardware}")
        print()
        
        return dexpilot_retargeting
        
    except Exception as e:
        print(f"Error in joint mapping debug: {e}")
        return None

if __name__ == "__main__":
    print("Starting DexPilot mapping analysis...\n")
    
    optimizer = analyze_dexpilot_mapping()
    if optimizer:
        test_input_vectors()
        debug_joint_mapping()
    
    print("\nAnalysis complete.")
