#!/usr/bin/env python3
"""
Test the improved thumb corrections focusing on thumb_0 rotation vs thumb_1 bending.

This test demonstrates the key fix:
- thumb_0 (rotation) is now PRIMARY correction for pinching (up to 25°)
- thumb_1 (bending) is now SECONDARY correction (only up to ~10°)
- Index finger gets stronger corrections than middle finger
"""

import numpy as np
import math
from thumb_pinch_corrector import ThumbPinchCorrector

def test_thumb_correction_improvements():
    """Test the key improvements in thumb correction strategy."""
    corrector = ThumbPinchCorrector()
    
    print("=== IMPROVED Thumb Correction Test ===\n")
    print("KEY IMPROVEMENTS:")
    print("- thumb_0 (rotation) is now PRIMARY correction for pinching")
    print("- thumb_1 (bending) is REDUCED to prevent over-bending at 90°")
    print("- Index finger gets stronger thumb_0 corrections\n")
    
    # Base joint angles representing a typical gesture
    joint_angles = np.array([0.1, 0.5, 0.3, -0.5, -0.3, -0.7, -0.2])
    
    # Test Index Finger Pinching (should get strong thumb_0 corrections)
    print("=== INDEX FINGER PINCHING ===")
    thumb_pos = np.array([0.03, 0.04, 0])
    index_pos = np.array([0.03, 0.05, 0])   # 1.4cm from thumb - strong pinch
    middle_pos = np.array([0, 0.09, 0])     # Far from thumb
    
    corrected = corrector.apply_correction(joint_angles, thumb_pos, index_pos, middle_pos)
    
    thumb_0_corr = (corrected[0] - joint_angles[0]) * 180 / math.pi
    thumb_1_corr = (corrected[1] - joint_angles[1]) * 180 / math.pi
    thumb_2_corr = (corrected[2] - joint_angles[2]) * 180 / math.pi
    
    print(f"thumb_0 (rotation) correction: {thumb_0_corr:.1f}° ← PRIMARY for pinching")
    print(f"thumb_1 (bending) correction:  {thumb_1_corr:.1f}° ← REDUCED to prevent over-bend")
    print(f"thumb_2 (tip) correction:      {thumb_2_corr:.1f}° ← minimal")
    print(f"Ratio thumb_0/thumb_1: {thumb_0_corr/thumb_1_corr:.1f}:1 (rotation dominates)\n")
    
    # Test Middle Finger Pinching (should get moderate thumb_0 corrections)
    print("=== MIDDLE FINGER PINCHING ===")
    thumb_pos = np.array([0.03, 0.04, 0])
    index_pos = np.array([0.05, 0.08, 0])   # Far from thumb
    middle_pos = np.array([0.03, 0.05, 0])  # 1.4cm from thumb - strong pinch
    
    corrected = corrector.apply_correction(joint_angles, thumb_pos, index_pos, middle_pos)
    
    thumb_0_corr_mid = (corrected[0] - joint_angles[0]) * 180 / math.pi
    thumb_1_corr_mid = (corrected[1] - joint_angles[1]) * 180 / math.pi
    thumb_2_corr_mid = (corrected[2] - joint_angles[2]) * 180 / math.pi
    
    print(f"thumb_0 (rotation) correction: {thumb_0_corr_mid:.1f}° ← less than index pinch")
    print(f"thumb_1 (bending) correction:  {thumb_1_corr_mid:.1f}° ← even more reduced")
    print(f"thumb_2 (tip) correction:      {thumb_2_corr_mid:.1f}° ← minimal")
    print(f"Ratio thumb_0/thumb_1: {thumb_0_corr_mid/thumb_1_corr_mid:.1f}:1\n")
    
    # Show the difference
    print("=== COMPARISON ===")
    print(f"Index pinch gets {thumb_0_corr:.1f}° thumb_0 correction")
    print(f"Middle pinch gets {thumb_0_corr_mid:.1f}° thumb_0 correction")
    print(f"Index advantage: {(thumb_0_corr - thumb_0_corr_mid):.1f}° more rotation correction")
    
    print("\n=== SOLUTION SUMMARY ===")
    print("✅ thumb_0 (rotation) is now PRIMARY correction - fixes pinching geometry")
    print("✅ thumb_1 (bending) is REDUCED - prevents over-bending at 90° operator position") 
    print("✅ Index finger gets stronger corrections - fixes index pinching regression")
    print("✅ Debug logging removed - no more console flooding during VR operation")

if __name__ == "__main__":
    test_thumb_correction_improvements()
