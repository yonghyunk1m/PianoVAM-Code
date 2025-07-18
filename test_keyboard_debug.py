#!/usr/bin/env python3

# Simple test to debug keyboard coordinate generation
import sys
sys.path.append('FingeringDetection')

from floatinghands_torch_pure import generate_keyboard_from_keystone_points

# Create some sample keystone points for testing
# These represent the corners of white keys across the keyboard
sample_keystone_points = {
    # A0 (leftmost key)
    "A0_upper": [0.1, 0.3],
    "A0_lower": [0.1, 0.7],
    
    # C1 
    "C1_upper": [0.15, 0.3],
    "C1_lower": [0.15, 0.7],
    
    # C2
    "C2_upper": [0.25, 0.3], 
    "C2_lower": [0.25, 0.7],
    
    # C3
    "C3_upper": [0.35, 0.3],
    "C3_lower": [0.35, 0.7],
    
    # C4 (middle C)
    "C4_upper": [0.45, 0.3],
    "C4_lower": [0.45, 0.7],
    
    # C5
    "C5_upper": [0.55, 0.3],
    "C5_lower": [0.55, 0.7],
    
    # C6
    "C6_upper": [0.65, 0.3],
    "C6_lower": [0.65, 0.7],
    
    # C7
    "C7_upper": [0.75, 0.3],
    "C7_lower": [0.75, 0.7],
    
    # C8 (rightmost key)
    "C8_upper": [0.9, 0.3],
    "C8_lower": [0.9, 0.7],
}

print("🔧 TESTING KEYBOARD COORDINATE GENERATION")
print("=" * 60)

try:
    bottompoints, toppoints = generate_keyboard_from_keystone_points(sample_keystone_points)
    print(f"\n✅ Generated {len(bottompoints)} bottom points and {len(toppoints)} top points")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 