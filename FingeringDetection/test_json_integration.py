#!/usr/bin/env python3
"""
Test script to verify JSON format integration with floating hands detection and MIDI comparison
"""

import os
import sys
import json
from floatinghands_torch_pure import *

def test_json_keyboard_loading():
    """Test loading keyboard data from JSON format"""
    print("🧪 Testing JSON keyboard loading...")
    
    # Check if pixel_points directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pixel_points_dir = os.path.join(script_dir, "pixel_points")
    
    if not os.path.exists(pixel_points_dir):
        print("❌ pixel_points directory not found")
        return False
    
    # Find JSON files
    json_files = [f for f in os.listdir(pixel_points_dir) if f.endswith('.json')]
    
    if not json_files:
        print("❌ No JSON files found in pixel_points directory")
        return False
    
    print(f"✅ Found {len(json_files)} JSON files")
    
    # Test loading each JSON file
    for json_file in json_files[:3]:  # Test first 3 files
        json_path = os.path.join(pixel_points_dir, json_file)
        print(f"\n📁 Testing: {json_file}")
        
        try:
            # Load keyboard directly from JSON pixel points
            keyboard, black_key_data = load_keyboard_from_json_pixel_points(json_path)
            if keyboard is None:
                print(f"   ❌ Failed to load keyboard data")
                return False
            print(f"   ✅ Keyboard loaded: {len(keyboard)} keys")
            if black_key_data:
                print(f"   🎹 Black keys: {len(black_key_data.get('black_key_indices', []))}")
            print(f"   ✅ Keyboard generated: {len(keyboard)} keys")
            if black_key_data:
                print(f"   ✅ Black key data: {len(black_key_data.get('black_key_indices', []))} black keys")
            
            # Test keyboard structure
            white_keys = 0
            black_keys = 0
            for i, key in enumerate(keyboard):
                if i % 12 in [0, 2, 3, 5, 7, 8, 10]:  # White keys
                    white_keys += 1
                else:  # Black keys
                    black_keys += 1
            
            print(f"   ✅ Key distribution: {white_keys} white keys, {black_keys} black keys")
            
            # Test polygon data
            polygons_with_data = sum(1 for key in keyboard if isinstance(key, dict) and 'polygon' in key and len(key['polygon']) > 0)
            print(f"   ✅ Keys with polygon data: {polygons_with_data}/{len(keyboard)}")
            
        except Exception as e:
            print(f"   ❌ Error loading {json_file}: {e}")
            return False
    
    return True

def test_midi_comparison_compatibility():
    """Test MIDI comparison compatibility with new keyboard format"""
    print("\n🧪 Testing MIDI comparison compatibility...")
    
    try:
        from midicomparison import keydistance
        
        # Create a simple test keyboard
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pixel_points_dir = os.path.join(script_dir, "pixel_points")
        json_files = [f for f in os.listdir(pixel_points_dir) if f.endswith('.json')]
        
        if not json_files:
            print("❌ No JSON files available for testing")
            return False
        
        # Load first JSON file for testing
        json_path = os.path.join(pixel_points_dir, json_files[0])
        keyboard, black_key_data = load_keyboard_from_json_pixel_points(json_path)
        if keyboard is None:
            print("❌ Failed to load keyboard data for testing")
            return False
        
        # Test keydistance function with new format
        test_point = [0.5, 0.5]  # Center point
        distances = []
        
        for i in range(min(5, len(keyboard))):  # Test first 5 keys
            try:
                distance = keydistance(keyboard, i, test_point)
                distances.append(distance)
                print(f"   ✅ Key {i} distance: {distance:.4f}")
            except Exception as e:
                print(f"   ❌ Error calculating distance for key {i}: {e}")
                return False
        
        print(f"   ✅ Successfully calculated distances for {len(distances)} keys")
        return True
        
    except ImportError as e:
        print(f"❌ Could not import midicomparison: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in MIDI comparison test: {e}")
        return False

def test_floating_hands_integration():
    """Test floating hands detection integration with new keyboard format"""
    print("\n🧪 Testing floating hands integration...")
    
    try:
        # Test that the functions can be called with new keyboard format
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pixel_points_dir = os.path.join(script_dir, "pixel_points")
        json_files = [f for f in os.listdir(pixel_points_dir) if f.endswith('.json')]
        
        if not json_files:
            print("❌ No JSON files available for testing")
            return False
        
        # Load keyboard data
        json_path = os.path.join(pixel_points_dir, json_files[0])
        keyboard, black_key_data = load_keyboard_from_json_pixel_points(json_path)
        if keyboard is None:
            print("❌ Failed to load keyboard data for testing")
            return False
        
        # Test handpositiondetector function signature
        # Create dummy data for testing
        dummy_handsinfo = []
        dummy_floatingframes = []
        
        # Test that handpositiondetector can handle the new keyboard format
        try:
            result = handpositiondetector(dummy_handsinfo, dummy_floatingframes, keyboard)
            print("   ✅ handpositiondetector accepts new keyboard format")
        except Exception as e:
            print(f"   ❌ handpositiondetector error: {e}")
            return False
        
        print("   ✅ Floating hands functions compatible with new keyboard format")
        return True
        
    except Exception as e:
        print(f"❌ Error in floating hands integration test: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 Testing JSON format integration with floating hands detection and MIDI comparison")
    print("=" * 80)
    
    tests = [
        ("JSON Keyboard Loading", test_json_keyboard_loading),
        ("MIDI Comparison Compatibility", test_midi_comparison_compatibility),
        ("Floating Hands Integration", test_floating_hands_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! JSON format integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 