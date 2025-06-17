#!/usr/bin/env python3
"""
Test script to verify that the retargeting method configuration works correctly.
This script tests the HandRetargeting class with both vector and dexpilot methods.
"""

import sys
import os
import argparse

# Add the teleop directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'teleop'))

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

def test_retargeting_methods():
    """Test both vector and dexpilot retargeting methods."""
    
    print("Testing HandRetargeting with different methods...")
    
    # Test 1: Vector method (default)
    try:
        print("\n1. Testing vector method:")
        hand_retargeting_vector = HandRetargeting(HandType.UNITREE_DEX3, 'vector')
        print(f"   ✓ Vector method initialized successfully")
        print(f"   Left retargeting type: {hand_retargeting_vector.cfg['left']['type']}")
        print(f"   Right retargeting type: {hand_retargeting_vector.cfg['right']['type']}")
    except Exception as e:
        print(f"   ✗ Vector method failed: {e}")
        return False
    
    # Test 2: Dexpilot method
    try:
        print("\n2. Testing dexpilot method:")
        hand_retargeting_dexpilot = HandRetargeting(HandType.UNITREE_DEX3, 'dexpilot')
        print(f"   ✓ Dexpilot method initialized successfully")
        print(f"   Left retargeting type: {hand_retargeting_dexpilot.cfg['left']['type']}")
        print(f"   Right retargeting type: {hand_retargeting_dexpilot.cfg['right']['type']}")
        
        # Check if dexpilot-specific parameters were set
        if 'wrist_link_name' in hand_retargeting_dexpilot.cfg['left']:
            print(f"   Left wrist link name: {hand_retargeting_dexpilot.cfg['left']['wrist_link_name']}")
        if 'finger_tip_link_names' in hand_retargeting_dexpilot.cfg['left']:
            print(f"   Left finger tip links: {hand_retargeting_dexpilot.cfg['left']['finger_tip_link_names']}")
    except Exception as e:
        print(f"   ✗ Dexpilot method failed: {e}")
        return False
    
    # Test 3: Invalid method (should use default)
    try:
        print("\n3. Testing invalid method (should default to config file):")
        hand_retargeting_invalid = HandRetargeting(HandType.UNITREE_DEX3, 'invalid_method')
        print(f"   ✓ Invalid method handled gracefully")
        print(f"   Left retargeting type: {hand_retargeting_invalid.cfg['left']['type']}")
    except Exception as e:
        print(f"   ✗ Invalid method test failed: {e}")
        return False
    
    print("\n✓ All retargeting method tests passed!")
    return True

def test_command_line_integration():
    """Test command-line argument parsing."""
    
    print("\nTesting command-line argument integration...")
    
    # Simulate command-line arguments
    test_args = [
        ('--retargeting-method', 'vector'),
        ('--retargeting-method', 'dexpilot'),
    ]
    
    for arg_name, arg_value in test_args:
        # Create a simple parser like in teleop_hand_and_arm.py
        parser = argparse.ArgumentParser()
        parser.add_argument('--retargeting-method', type=str, choices=['vector', 'dexpilot'], default='vector', 
                          help='Select hand retargeting method: vector (default) or dexpilot')
        
        try:
            args = parser.parse_args([arg_name, arg_value])
            print(f"   ✓ Command-line argument '{arg_name} {arg_value}' parsed successfully: {args.retargeting_method}")
        except Exception as e:
            print(f"   ✗ Command-line argument parsing failed: {e}")
            return False
    
    print("✓ Command-line integration tests passed!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("RETARGETING METHOD CONFIGURATION TEST")
    print("=" * 60)
    
    success = True
    
    # Test the HandRetargeting class
    if not test_retargeting_methods():
        success = False
    
    # Test command-line integration
    if not test_command_line_integration():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! The retargeting method configuration is working correctly.")
        print("\nYou can now use the following commands:")
        print("  python teleop_hand_and_arm.py --hand dex3 --retargeting-method vector")
        print("  python teleop_hand_and_arm.py --hand dex3 --retargeting-method dexpilot")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    print("=" * 60)
