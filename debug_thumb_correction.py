#!/usr/bin/env python3
"""
Debug the thumb correction application step by step.
"""

import numpy as np
import math
import sys
import os

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)

from teleop.robot_control.thumb_pinch_corrector import ThumbPinchCorrector


def debug_thumb_correction():
    """Debug the thumb correction step by step."""
    
    print("=== Debug Thumb Correction ===\n")
    
    # Initialize corrector
    corrector = ThumbPinchCorrector()
    
    # Test with simulated DexPilot output for pinching
    dexpilot_output = np.array([
        -0.24,   # thumb_0 (index 0)
        0.917,   # thumb_1 (index 1) ← main target
        1.746,   # thumb_2 (index 2)
        -1.604,  # middle_0 (index 3)
        -1.538,  # middle_1 (index 4)
        -1.556,  # index_0 (index 5)
        -1.638   # index_1 (index 6)
    ])
    
    # Simulated fingertip positions for pinching
    thumb_tip_pos = np.array([0.03, 0.04, 0])
    index_tip_pos = np.array([0.03, 0.05, 0])
    middle_tip_pos = np.array([0, 0.05, 0])
    
    print("1. Input Data:")
    print(f"   DexPilot output: {dexpilot_output}")
    print(f"   Thumb angles: [{dexpilot_output[0]*180/math.pi:.1f}°, "
          f"{dexpilot_output[1]*180/math.pi:.1f}°, {dexpilot_output[2]*180/math.pi:.1f}°]")
    print(f"   Thumb tip position: {thumb_tip_pos}")
    print(f"   Index tip position: {index_tip_pos}")
    print(f"   Middle tip position: {middle_tip_pos}")
    print()
    
    # Step 1: Detect pinching
    is_pinching, pinch_strength, closest_dist = corrector.detect_pinching(
        thumb_tip_pos, index_tip_pos, middle_tip_pos
    )
    
    print("2. Pinch Detection:")
    print(f"   Is pinching: {is_pinching}")
    print(f"   Pinch strength: {pinch_strength:.3f}")
    print(f"   Closest distance: {closest_dist:.3f}m")
    print()
    
    # Step 2: Calculate correction
    thumb_angles = dexpilot_output[0:3]
    corrections = corrector.calculate_thumb_correction(pinch_strength, thumb_angles)
    
    print("3. Correction Calculation:")
    print(f"   Thumb angles (input): {thumb_angles}")
    print(f"   Corrections: {corrections}")
    print(f"   Corrections (degrees): [{corrections[0]*180/math.pi:.1f}°, "
          f"{corrections[1]*180/math.pi:.1f}°, {corrections[2]*180/math.pi:.1f}°]")
    print()
    
    # Step 3: Apply correction
    corrected_output = corrector.apply_correction(
        dexpilot_output, thumb_tip_pos, index_tip_pos, middle_tip_pos
    )
    
    print("4. Final Result:")
    print(f"   Original output: {dexpilot_output}")
    print(f"   Corrected output: {corrected_output}")
    print(f"   Difference: {corrected_output - dexpilot_output}")
    print()
    
    print("5. Thumb Joint Comparison:")
    print(f"   Original thumb_0: {dexpilot_output[0]:.3f} rad ({dexpilot_output[0]*180/math.pi:.1f}°)")
    print(f"   Corrected thumb_0: {corrected_output[0]:.3f} rad ({corrected_output[0]*180/math.pi:.1f}°)")
    print(f"   Change: {(corrected_output[0] - dexpilot_output[0])*180/math.pi:.1f}°")
    print()
    print(f"   Original thumb_1: {dexpilot_output[1]:.3f} rad ({dexpilot_output[1]*180/math.pi:.1f}°)")
    print(f"   Corrected thumb_1: {corrected_output[1]:.3f} rad ({corrected_output[1]*180/math.pi:.1f}°)")
    print(f"   Change: {(corrected_output[1] - dexpilot_output[1])*180/math.pi:.1f}°")
    print()
    print(f"   Original thumb_2: {dexpilot_output[2]:.3f} rad ({dexpilot_output[2]*180/math.pi:.1f}°)")
    print(f"   Corrected thumb_2: {corrected_output[2]:.3f} rad ({corrected_output[2]*180/math.pi:.1f}°)")
    print(f"   Change: {(corrected_output[2] - dexpilot_output[2])*180/math.pi:.1f}°")


if __name__ == "__main__":
    debug_thumb_correction()
